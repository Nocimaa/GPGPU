#!/usr/bin/env python3
"""Run a CPU grid search between two captures and save every resulting overlay."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

ROOT = Path(__file__).resolve().parent.parent

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - user may not have tqdm installed
    tqdm = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    baseline: Path
    current: Path


@dataclass(frozen=True)
class BenchConfig:
    gpgpu_dir: Path = ROOT / "gpgpu-cuda"
    datasets: tuple[DatasetSpec, ...] = (
        DatasetSpec(
            "car",
            ROOT / "benchmarks" / "images" / "car" / "baseline.png",
            ROOT / "benchmarks" / "images" / "car" / "current.png",
        ),
        DatasetSpec(
            "staline",
            ROOT / "benchmarks" / "images" / "staline" / "baseline.png",
            ROOT / "benchmarks" / "images" / "staline" / "current.png",
        ),
    )
    output_dir: Path = ROOT / "outputs" / "cpu_image_bench"
    results: Path = ROOT / "results" / "cpu_image_bench.json"
    opening_sizes: tuple[int, ...] = (1, 3, 5, 9, 13)
    th_low_values: tuple[int, ...] = (10, 20, 30, 40, 50)
    th_high_values: tuple[int, ...] = (35, 45, 55, 65, 75)
    bg_sampling_values: tuple[int, ...] = (100, 200, 300, 400, 500)
    bg_number_frame_values: tuple[int, ...] = (5, 10, 15, 20, 25)
    extra: tuple[str, ...] = ()


def default_config() -> BenchConfig:
    return BenchConfig()


def build_temp_video(image_a: Path, image_b: Path, tmpdir: Path) -> Path:
    frames = [tmpdir / "frame_0.png", tmpdir / "frame_1.png"]
    shutil.copy(image_a, frames[0])
    shutil.copy(image_b, frames[1])

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to build the temporary video.")

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

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmpdir / "stream_input.mp4"


def run_stream(
    stream_binary: Path,
    video: Path,
    combo: dict[str, int],
    extra_flags: list[str],
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(stream_binary), "--mode=cpu", str(video), "--output", str(output_path)]
    for name, value in combo.items():
        cmd.append(f"--{name}={value}")
    cmd.extend(extra_flags)
    return subprocess.run(cmd, capture_output=True, text=True)


def extract_overlay(video_path: Path, overlay_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(overlay_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    cfg = default_config()
    stream_exe = cfg.gpgpu_dir / "build" / "stream"
    if not stream_exe.exists():
        raise SystemExit(f"{stream_exe} not found; build the CUDA project first.")

    opening = cfg.opening_sizes
    th_low = cfg.th_low_values
    th_high = cfg.th_high_values
    bg_sampling = cfg.bg_sampling_values
    bg_number = cfg.bg_number_frame_values

    combos = list(itertools.product(opening, th_low, th_high, bg_sampling, bg_number))
    runs = list(itertools.product(cfg.datasets, combos))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.results.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    iterator = tqdm(runs, desc="CPU image grid", unit="combo") if tqdm is not None else runs

    for dataset, combo in iterator:
        combo_dict = {
            "opening_size": combo[0],
            "th_low": combo[1],
            "th_high": combo[2],
            "bg_sampling_rate": combo[3],
            "bg_number_frame": combo[4],
        }
        encoded = "_".join(f"{name}{value}" for name, value in combo_dict.items())

        final_image = cfg.output_dir / f"{dataset.name}__{encoded}.png"
        tmpdir = Path(tempfile.mkdtemp(prefix="cpu-image-bench-"))
        duration = 0.0
        overlay_error: str | None = None
        result: subprocess.CompletedProcess[str] | None = None
        try:
            video_path = build_temp_video(dataset.baseline, dataset.current, tmpdir)
            diff_output = tmpdir / "diff_output.mp4"

            start = time.perf_counter()
            result = run_stream(stream_exe, video_path, combo_dict, cfg.extra, diff_output)
            print(result.stdout)
            duration = time.perf_counter() - start

            overlay = tmpdir / "diff_overlay.png"
            if result.returncode == 0 and diff_output.exists():
                try:
                    extract_overlay(diff_output, overlay)
                    shutil.copy(overlay, final_image)
                except subprocess.CalledProcessError as exc:
                    print(exc)
                    overlay_error = str(exc)
            else:
                overlay_error = (
                    "stream binary failed"
                    if result.returncode != 0
                    else "diff output missing"
                )
        except Exception as exc:
            print(f"[error] dataset={dataset.name} combo={combo_dict} exception: {exc}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        record = {
            "dataset": dataset.name,
            "combo": combo_dict,
            "output_image": str(final_image),
            "command": " ".join(result.args) if result else "",
            "duration_seconds": duration,
            "returncode": result.returncode if result else -1,
            "stdout": result.stdout.strip() if result else "",
            "stderr": result.stderr.strip() if result else "",
            "overlay_error": overlay_error,
        }
        results.append(record)
        cfg.results.write_text(json.dumps(results, indent=2))

        status = "success" if result and result.returncode == 0 else "error"
        print(f"[{status}] dataset={dataset.name} combo={combo_dict} duration={duration:.2f}s image={final_image.name}")

    cfg.results.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} records to {cfg.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
