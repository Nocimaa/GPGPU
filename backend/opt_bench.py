#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path


def build_command(
    binary: Path,
    video: Path,
    output_dir: Path,
    toggle_combo: dict[str, int],
    extra_flags: list[str],
) -> list[str]:
    cmd = [
        str(binary),
        "--mode=gpu",
        "--gpu-diff=1",
        "--gpu-hysteresis=1",
        "--gpu-morphology=1",
        "--gpu-background=1",
        "--gpu-overlay=1",
        "--kernel-fusion=0",
        str(video),
    ]

    for name, value in toggle_combo.items():
        cmd.append(f"--{name}={value}")

    cmd.extend(extra_flags)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"optbench_{'_'.join(f'{k}{v}' for k,v in toggle_combo.items())}.mp4"
        cmd.extend(["--output", str(output_path)])

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPU toggles for Compute")
    parser.add_argument(
        "--gpgpu-dir",
        type=Path,
        default=Path("gpgpu-cuda"),
        help="Path to the gpgpu-cuda project (contains build/stream)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("gpgpu-cuda/samples/ACET.mp4"),
        help="Input video used for each benchmark iteration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/opt_bench"),
        help="Where to keep the encoded outputs",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra command line flags passed to the stream binary",
    )
    parser.add_argument(
        "--toggle-flags",
        nargs="+",
        default=["gpu-diff", "gpu-hysteresis", "gpu-morphology", "gpu-background", "gpu-overlay"],
        help="Which boolean GPU toggles to permute",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/opt_bench.json"),
        help="JSON file that will collect timing data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stream_exe = Path(args.gpgpu_dir) / "build" / "stream"
    if not stream_exe.exists() and not args.reuse_library:
        raise SystemExit(f"{stream_exe} was not found; run build.sh first")

    toggle_names = args.toggle_flags
    combos = list(itertools.product([0, 1], repeat=len(toggle_names)))
    stream_cmd_base = [str(stream_exe)]

    results: list[dict[str, object]] = []
    for combo in combos:
        combo_dict = dict(zip(toggle_names, combo))
        cmd = build_command(
            stream_exe,
            args.video,
            args.output_dir,
            combo_dict,
            args.extra,
        )

        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.perf_counter() - start

        record = {
            "toggle_combo": combo_dict,
            "cmd": " ".join(cmd),
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        results.append(record)

        print(f"Completed combo={combo_dict} rc={result.returncode} duration={duration:.2f}s")
        if result.returncode != 0:
            print(result.stderr)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} records to {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
