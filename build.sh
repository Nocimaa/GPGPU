#!/usr/bin/env sh
set -eu

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
GP_DIR="$SCRIPT_ROOT/gpgpu-cuda"
BUILD_DIR="$GP_DIR/build"

echo "Configuring GPGPU CUDA project (Debug)..."
cmake -S "$GP_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug

echo "Building GPGPU CUDA project..."
cmake --build "$BUILD_DIR"

PYTHON_BACKEND_DIR="$SCRIPT_ROOT/python-backend"
PYTHON_ENGINE_LIB="$PYTHON_BACKEND_DIR/_stream_engine.so"
BUILD_ENGINE_LIB="$BUILD_DIR/_stream_engine.so"

if [ ! -f "$BUILD_ENGINE_LIB" ]; then
  echo "Error: expected library at $BUILD_ENGINE_LIB" >&2
  exit 1
fi

cp "$BUILD_ENGINE_LIB" "$PYTHON_ENGINE_LIB"
echo "Synced python backend engine to $PYTHON_ENGINE_LIB"

echo "✅ build complete. You can run ./gpgpu-cuda/build/stream --mode=[cpu,gpu] ..."
