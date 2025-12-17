#!/usr/bin/env python3
"""GPU benchmark helper that grid-searches stream optimizations."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search GPU optimization toggles for stream.")
    parser.add_argument(
        "--gpgpu-dir",
        type=Path,
        default=Path("gpgpu-cuda"),
        help="Path to the CUDA project containing the `build/stream` binary.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("gpgpu-cuda/samples/ACET.mp4"),
        help="Video file used for every benchmark iteration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark"),
        help="Optional directory where output videos are written (makes debugging easier).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/benchmark.json"),
        help="JSON file that will collect timing and memory data.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra command-line flags passed to the stream binary.",
    )
    parser.add_argument(
        "--toggles",
        nargs="+",
        default=[
            "gpu-diff",
            "gpu-hysteresis",
            "gpu-morphology",
            "gpu-background",
            "gpu-overlay",
        ],
        help="Boolean GPU flags to permute (0 or 1).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Interval (in seconds) between GPU memory samples.",
    )
    return parser.parse_args()


def build_command(
    stream_binary: Path,
    video: Path,
    combo: dict[str, int],
    extra_flags: Sequence[str],
    output_dir: Path | None,
) -> list[str]:
    cmd = [
        str(stream_binary),
        "--mode=gpu",
        str(video),
    ]

    for name, value in combo.items():
        cmd.append(f"--{name}={value}")

    cmd.extend(str(flag) for flag in extra_flags)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        encoded = "_".join(f"{name}{value}" for name, value in combo.items())
        timestamp = int(time.time())
        filename = output_dir / f"bench_{timestamp}_{encoded}.mp4"
        cmd.extend(["--output", str(filename)])

    return cmd


def query_gpu_memory(pid: int) -> int:
    if shutil.which("nvidia-smi") is None:
        return 0

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0

    if result.returncode != 0:
        return 0

    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) != 2:
            continue
        try:
            recorded_pid = int(parts[0])
            memory_used = int(parts[1])
        except ValueError:
            continue
        if recorded_pid == pid:
            return memory_used

    return 0


def run_with_gpu_monitor(cmd: Sequence[str], interval: float) -> tuple[int, float, int, str, str]:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start = time.perf_counter()
    max_gpu_memory = 0
    while process.poll() is None:
        memory = query_gpu_memory(process.pid)
        if memory > max_gpu_memory:
            max_gpu_memory = memory
        time.sleep(interval)

    duration = time.perf_counter() - start
    stdout, stderr = process.communicate()
    max_gpu_memory = max(max_gpu_memory, query_gpu_memory(process.pid))
    return process.returncode, duration, max_gpu_memory, stdout.strip(), stderr.strip()


def main() -> int:
    args = parse_args()
    stream_exe = args.gpgpu_dir / "build" / "stream"
    if not stream_exe.exists():
        raise SystemExit(f"{stream_exe} not found; build the CUDA project first.")

    toggle_names = args.toggles
    combos = list(itertools.product([0, 1], repeat=len(toggle_names)))
    results: list[dict[str, object]] = []

    for combo in combos:
        combo_values = dict(zip(toggle_names, combo))
        cmd = build_command(
            stream_exe,
            args.video,
            combo_values,
            args.extra,
            args.output_dir,
        )

        returncode, duration, max_gpu_mem, stdout, stderr = run_with_gpu_monitor(cmd, args.interval)

        record = {
            "toggle_combo": combo_values,
            "command": " ".join(cmd),
            "duration_seconds": duration,
            "max_gpu_memory_mb": max_gpu_mem,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        results.append(record)

        status = "ok" if returncode == 0 else "failed"
        print(f"[{status}] combo={combo_values} duration={duration:.2f}s max_gpu={max_gpu_mem}MB")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} records to {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
