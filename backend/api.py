from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
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


def _write_images_to_disk(image_a: UploadFile, image_b: UploadFile, tmpdir: Path) -> tuple[Path, tuple[Path, Path]]:
    original_a = tmpdir / "baseline.png"
    original_b = tmpdir / "current.png"
    original_a.write_bytes(image_a.file.read())
    original_b.write_bytes(image_b.file.read())
    return tmpdir, (original_a, original_b)


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
        )

        result = lib.run(mode, str(video_path), params=params)
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
        return FileResponse(diff_image, media_type="image/png", filename="diff.png", background=background_tasks)
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
        )

        result = lib.run(mode, str(video_path), params=params)
        if result != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="Stream engine returned an error.")

        background_tasks.add_task(shutil.rmtree, tmpdir, ignore_errors=True)
        return FileResponse(params.output, media_type="video/mp4", filename="diff_video.mp4", background=background_tasks)
    except Exception as exc:
        print(exc)
        raise HTTPException(status_code=500, detail=f"An error occurred: {exc}") from exc
