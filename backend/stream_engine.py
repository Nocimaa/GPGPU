from __future__ import annotations

from dataclasses import dataclass
from ctypes import CDLL, c_char_p, c_int
from pathlib import Path
from typing import Optional

_LIB_NAME = "_stream_engine.so"


def _bytes(value: Optional[str]) -> bytes:
    return value.encode() if value else b""


def _library_path(custom: Optional[Path]) -> Path:
    candidate = custom or Path(__file__).with_name(_LIB_NAME)
    if not candidate.exists():
        raise RuntimeError(f"Native library {candidate} not found; run the CMake build first.")
    return candidate


@dataclass
class StreamParams:
    output: Optional[str] = None
    background: Optional[str] = None
    opening_size: int = 3
    th_low: int = 3
    th_high: int = 30
    bg_sampling_rate: int = 500
    bg_number_frame: int = 10
    cpu_simd: bool = False


class StreamLib:
    """Minimal wrapper around the `_stream_engine.so` native binary."""

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        self._lib = CDLL(str(_library_path(lib_path)))
        self._lib.run_stream_c.argtypes = [
            c_char_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
        ]
        self._lib.run_stream_c.restype = c_int

    def run(self, mode: str, filename: str, params: Optional[StreamParams] = None) -> int:
        target = params or StreamParams()
        return self._lib.run_stream_c(
            mode.encode(),
            filename.encode(),
            _bytes(target.output),
            _bytes(target.background),
            target.opening_size,
            target.th_low,
            target.th_high,
            target.bg_sampling_rate,
            target.bg_number_frame,
            1 if target.cpu_simd else 0,
        )


_stream_lib = StreamLib()


def run_stream(
    mode: str,
    filename: str,
    *,
    output: Optional[str] = None,
    background: Optional[str] = None,
    opening_size: int = 3,
    th_low: int = 3,
    th_high: int = 30,
    bg_sampling_rate: int = 500,
    bg_number_frame: int = 10,
    cpu_simd: bool = False,
) -> int:
    """Run the streaming pipeline through the shared library."""

    params = StreamParams(
        output=output,
        background=background,
        opening_size=opening_size,
        th_low=th_low,
        th_high=th_high,
        bg_sampling_rate=bg_sampling_rate,
        bg_number_frame=bg_number_frame,
        cpu_simd=cpu_simd,
    )
    return _stream_lib.run(mode, filename, params=params)
