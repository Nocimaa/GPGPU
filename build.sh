#/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GP_DIR="$SCRIPT_ROOT/gpgpu-cuda"
BUILD_DIR="$GP_DIR/build"

echo "Configuring GPGPU CUDA project (Debug)..."
cmake -S "$GP_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug

echo "Building GPGPU CUDA project..."
cmake --build "$BUILD_DIR"

echo "✅ build complete. You can run ./gpgpu-cuda/build/stream --mode=[cpu,gpu] ..."
