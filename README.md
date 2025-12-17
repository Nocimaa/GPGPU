# GPGPU Stream Demo

Instructions to build the native pipeline, run the backend API, and launch the frontend.

## Prerequisites
- CUDA toolkit with a compatible GPU (for `--mode=gpu`).
- GStreamer runtime with the common plugins (`gstreamer1.0-tools`, `-plugins-base/good/bad/ugly`, `-libav`) for decode/encode.
- Node.js 18+ for the frontend.

## Build native components
From the repo root:
```bash
sh build.sh
```
This configures and builds `gpgpu-cuda`, then copies the shared library to `backend/_stream_engine.so`.

## Run the backend (API)
```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
The API exposes:
- `POST /compare` for two images (baseline/current).
- `POST /compare/video` for a video file.
- `GET /health` for a quick status check.

## Run the frontend (Vite)
```bash
cd frontend
npm install
npm run dev -- --host --port 5173
```
Set `VITE_BACKEND_URL` in `.env` if the backend is on a different host/port.

## Run the native CLI directly
```bash
# CPU
./gpgpu-cuda/build/stream --mode=cpu ./gpgpu-cuda/samples/ACET.mp4 --output=out.mp4
# GPU (requires CUDA + GStreamer plugins)
./gpgpu-cuda/build/stream --mode=gpu ./gpgpu-cuda/samples/ACET.mp4 --output=out-gpu.mp4
```
Optional flags (both CPU/GPU): `--opening_size`, `--th_low`, `--th_high`, `--bg_sampling_rate`, `--bg_number_frame`, `--cpu-simd`. GPU-only toggles: `--gpu-diff`, `--gpu-hysteresis`, `--gpu-morphology`, `--gpu-background`, `--gpu-overlay`, `--kernel-fusion`.

## Notes
- For live cam in the frontend, a webcam and browser permissions are required.
- If you see GStreamer errors about missing elements (e.g., `x264enc`, `mp4mux`), install the plugin packages listed in prerequisites.
