from __future__ import annotations

import os
from ctypes import CDLL, c_char_p, c_int
from pathlib import Path

_LIB_NAME = "_stream_engine.so"
_LIB_PATH = Path(__file__).with_name(_LIB_NAME)

if not _LIB_PATH.exists():
    raise RuntimeError(f"Native library {_LIB_PATH} not found; run the CMake build first.")

_lib = CDLL(str(_LIB_PATH))
_lib.run_stream_c.argtypes = [
    c_char_p,
    c_char_p,
    c_char_p,
    c_char_p,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
]
_lib.run_stream_c.restype = c_int


def run_stream(
    mode: str,
    filename: str,
    *,
    output: str | None = None,
    background: str | None = None,
    opening_size: int = 3,
    th_low: int = 3,
    th_high: int = 30,
    bg_sampling_rate: int = 500,
    bg_number_frame: int = 10,
) -> int:
    """Run the streaming pipeline through the shared library."""

    def _bytes(value: str | None) -> bytes:
        return value.encode() if value else b""

    return _lib.run_stream_c(
        mode.encode(),
        filename.encode(),
        _bytes(output),
        _bytes(background),
        opening_size,
        th_low,
        th_high,
        bg_sampling_rate,
        bg_number_frame,
    )
