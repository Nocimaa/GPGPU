from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
import shutil
import subprocess
import tempfile

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from stream_engine import StreamLib, StreamParams

app = FastAPI(title="Stream Diff API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
lib = StreamLib()


DEFAULT_LIVE_PARAMS = {
    "opening_size": 3,
    "th_low": 10,
    "th_high": 50,
    "bg_sampling_rate": 500,
    "bg_number_frame": 10,
    "cpu_simd": True,
}


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y", "t")
    return bool(value)


def _write_images_to_disk(image_a: UploadFile, image_b: UploadFile, tmpdir: Path) -> tuple[Path, tuple[Path, Path]]:
    original_a = tmpdir / "baseline.png"
    original_b = tmpdir / "current.png"
    original_a.write_bytes(image_a.file.read())
    original_b.write_bytes(image_b.file.read())
    return tmpdir, (original_a, original_b)


def _normalize_live_params(params: dict | None) -> dict:
    normalized = DEFAULT_LIVE_PARAMS.copy()
    if not isinstance(params, dict):
        return normalized
    for key in ("opening_size", "th_low", "th_high", "bg_sampling_rate", "bg_number_frame"):
        if key in params:
            try:
                normalized[key] = int(params[key])
            except (TypeError, ValueError):
                pass
    if "cpu_simd" in params:
        normalized["cpu_simd"] = _parse_bool(params["cpu_simd"])
    return normalized


def _decode_data_url(data_url: str) -> bytes:
    if "," in data_url:
        _, payload = data_url.split(",", 1)
    else:
        payload = data_url
    return base64.b64decode(payload)


def _write_bytes_to_frames(frame_a: bytes, frame_b: bytes, tmpdir: Path) -> tuple[Path, Path]:
    baseline = tmpdir / "baseline.png"
    current = tmpdir / "current.png"
    baseline.write_bytes(frame_a)
    current.write_bytes(frame_b)
    return baseline, current


def _build_temp_video(frames: tuple[Path, Path], tmpdir: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg is required to build the temporary video.")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "1",
        "-i",
        str(tmpdir / "frame_%d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(tmpdir / "stream_input.mp4"),
    ]

    for idx, frame in enumerate(frames):
        shutil.copy(frame, tmpdir / f"frame_{idx}.png")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create temporary video: {exc}") from exc

    return tmpdir / "stream_input.mp4"


def _process_live_frames_sync(frame_a: bytes, frame_b: bytes, mode: str, params: dict) -> bytes:
    tmpdir = Path(tempfile.mkdtemp(prefix="stream-live-"))
    try:
        print("Processing live frames...")
        baseline, current = _write_bytes_to_frames(frame_a, frame_b, tmpdir)
        video_path = _build_temp_video((baseline, current), tmpdir)
        normalized = _normalize_live_params(params)
        stream_params = StreamParams(
            output=str(tmpdir / "diff_output.mp4"),
            opening_size=normalized["opening_size"],
            th_low=normalized["th_low"],
            th_high=normalized["th_high"],
            bg_sampling_rate=normalized["bg_sampling_rate"],
            bg_number_frame=normalized["bg_number_frame"],
            cpu_simd=normalized["cpu_simd"],
        )

        run_mode = mode.lower()
        if run_mode not in ("cpu", "gpu"):
            run_mode = "cpu"

        result = lib.run(run_mode, str(video_path), params=stream_params)
        if result != 0:
            raise RuntimeError("Stream engine returned an error.")

        diff_image = tmpdir / "diff_overlay.png"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            stream_params.output,
            "-vf",
            "select=eq(n\\,1)",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(diff_image),
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(diff_image)
        if not diff_image.exists():
            raise RuntimeError("Failed to generate overlay image.")

        print(diff_image)
        return diff_image.read_bytes()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _process_live_frames(frame_a: bytes, frame_b: bytes, mode: str, params: dict) -> bytes:
    return await asyncio.to_thread(_process_live_frames_sync, frame_a, frame_b, mode, params)


async def _send_overlay_response(
    websocket: WebSocket,
    frame_a: bytes,
    frame_b: bytes,
    mode: str,
    params: dict | None,
) -> bool:
    """Run the engine on a pair of frames and send the overlay back to the client."""
    try:
        overlay_bytes = await _process_live_frames(frame_a, frame_b, mode, params or {})
    except Exception as exc:
        print("Frame processing error:", exc)
        await websocket.send_json({"error": str(exc)})
        return False

    overlay_data_url = f"data:image/png;base64,{base64.b64encode(overlay_bytes).decode()}"
    await websocket.send_json({"overlay": overlay_data_url})
    return True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compare")
def compare_images(
    background_tasks: BackgroundTasks,
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    mode: str = Form("cpu"),
    output: str | None = Form(None),
    opening_size: int = Form(3),
    th_low: int = Form(3),
    th_high: int = Form(30),
    bg_sampling_rate: int = Form(500),
    bg_number_frame: int = Form(10),
    cpu_simd: bool = Form(True),
) -> FileResponse:
    """Send two frames to the native stream engine and return the produced diff video."""
    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="stream-api-"))
        _write_images_to_disk(image_a, image_b, tmpdir)
        video_path = _build_temp_video((tmpdir / "baseline.png", tmpdir / "current.png"), tmpdir)

        params = StreamParams(
            output=output or str(tmpdir / "diff_output.mp4"),
            opening_size=opening_size,
            th_low=th_low,
            th_high=th_high,
            bg_sampling_rate=bg_sampling_rate,
            bg_number_frame=bg_number_frame,
            cpu_simd=_parse_bool(cpu_simd),
        )

        start = time.perf_counter()
        result = lib.run(mode, str(video_path), params=params)
        duration = time.perf_counter() - start
        if result != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="Stream engine returned an error.")

        diff_image = tmpdir / "diff_overlay.png"
        # Grab the second frame from the diff video so the result mirrors the "current" image.
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            params.output,
            "-vf",
            "select=eq(n\\,1)",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(diff_image),
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Failed to extract overlay image: {exc}") from exc

        background_tasks.add_task(shutil.rmtree, tmpdir, ignore_errors=True)
        return FileResponse(
            diff_image,
            media_type="image/png",
            filename="diff.png",
            background=background_tasks,
            headers={"X-Execution-Time": f"{duration:.6f}"},
        )
    except Exception as exc:
        print(exc)
        raise HTTPException(status_code=500, detail=f"An error occurred: {exc}") from exc

@app.post("/compare/video")
def compare_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    mode: str = Form("cpu"),
    opening_size: int = Form(3),
    th_low: int = Form(3),
    th_high: int = Form(30),
    bg_sampling_rate: int = Form(500),
    bg_number_frame: int = Form(10),
    cpu_simd: bool = Form(True),
) -> FileResponse:
    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="stream-video-"))
        video_path = tmpdir / "input_video"
        suffix = Path(video.filename).suffix or ".mp4"
        video_path = video_path.with_suffix(suffix)
        video_path.write_bytes(video.file.read())

        params = StreamParams(
            output=str(tmpdir / "diff_video.mp4"),
            opening_size=opening_size,
            th_low=th_low,
            th_high=th_high,
            bg_sampling_rate=bg_sampling_rate,
            bg_number_frame=bg_number_frame,
            cpu_simd=_parse_bool(cpu_simd),
        )

        start = time.perf_counter()
        result = lib.run(mode, str(video_path), params=params)
        duration = time.perf_counter() - start
        if result != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="Stream engine returned an error.")

        background_tasks.add_task(shutil.rmtree, tmpdir, ignore_errors=True)
        return FileResponse(
            params.output,
            media_type="video/mp4",
            filename="diff_video.mp4",
            background=background_tasks,
            headers={"X-Execution-Time": f"{duration:.6f}"},
        )
    except Exception as exc:
        print(exc)
        raise HTTPException(status_code=500, detail=f"An error occurred: {exc}") from exc

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    previous_frame: bytes | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                print("JSON decode error:", exc)
                await websocket.send_json({"error": "Invalid JSON payload"})
                continue

            payload_type = payload.get("type")
            if payload_type == "frame":
                frame_data = payload.get("frame")
                if not isinstance(frame_data, str):
                    await websocket.send_json({"error": "Frame must be a base64 data URL"})
                    continue
                try:
                    current_frame = _decode_data_url(frame_data)
                except Exception as exc:
                    print("Data URL decode error:", exc)
                    await websocket.send_json({"error": f"Unable to decode frame: {exc}"})
                    continue
                if previous_frame is None:
                    previous_frame = current_frame
                    await websocket.send_json({"status": "baseline stored"})
                    continue

                mode = str(payload.get("mode", "cpu") or "cpu")
                params = payload.get("params")
                await _send_overlay_response(websocket, previous_frame, current_frame, mode, params)
                previous_frame = current_frame
                continue

            if payload_type == "frames":
                print("Received frames payload")
                frames = payload.get("frames")
                if not isinstance(frames, list) or len(frames) != 2:
                    await websocket.send_json({"error": "Two frames are required"})
                    continue
                try:
                    decoded = [_decode_data_url(frame) for frame in frames]
                except Exception as exc:
                    print("Data URL decode error:", exc)
                    await websocket.send_json({"error": f"Unable to decode frames: {exc}"})
                    continue

                mode = str(payload.get("mode", "cpu") or "cpu")
                params = payload.get("params")
                await _send_overlay_response(websocket, decoded[0], decoded[1], mode, params)
                previous_frame = decoded[1]
                continue

            await websocket.send_json({"error": "Unsupported payload type"})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        pass
