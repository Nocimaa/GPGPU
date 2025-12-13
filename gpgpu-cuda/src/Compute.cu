#include "Compute.hpp"
#include "Image.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(cmd)                                                     \
  do {                                                                       \
    cudaError_t error = (cmd);                                               \
    if (error != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(error));                               \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

namespace {

struct DeviceState
{
  Image<float> background;
  Image<uint8_t> mask;
  Image<uint8_t> temp;
  size_t frame_count = 0;
};

static DeviceState g_device;

inline void ensure_device_buffers(ImageView<rgb8> in)
{
  if (g_device.background.width != in.width || g_device.background.height != in.height)
  {
    g_device.background = Image<float>(in.width, in.height, true);
    g_device.mask = Image<uint8_t>(in.width, in.height, true);
    g_device.temp = Image<uint8_t>(in.width, in.height, true);
    g_device.frame_count = 0;
  }
}

__device__ inline float luminance(const rgb8& px)
{
  return px.r * 0.299f + px.g * 0.587f + px.b * 0.114f;
}

__device__ inline float* background_row(const ImageView<float>& view, int y)
{
  return reinterpret_cast<float*>((uint8_t*)view.buffer + y * view.stride);
}

__device__ inline uint8_t* mask_row(const ImageView<uint8_t>& view, int y)
{
  return reinterpret_cast<uint8_t*>((uint8_t*)view.buffer + y * view.stride);
}

__global__ void update_background(ImageView<rgb8> frame, ImageView<float> background, float alpha, bool initialize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= frame.width || y >= frame.height)
    return;

  auto row = reinterpret_cast<rgb8*>((uint8_t*)frame.buffer + y * frame.stride);
  auto bg_row = background_row(background, y);

  float lum = luminance(row[x]);
  if (initialize)
    bg_row[x] = lum;
  else
    bg_row[x] = bg_row[x] * (1.f - alpha) + lum * alpha;
}

__global__ void build_mask(ImageView<rgb8> frame, ImageView<float> background, ImageView<uint8_t> mask, float threshold)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= frame.width || y >= frame.height)
    return;

  auto row = reinterpret_cast<rgb8*>((uint8_t*)frame.buffer + y * frame.stride);
  auto bg_row = background_row(background, y);
  auto mask_line = mask_row(mask, y);

  float diff = std::fabs(luminance(row[x]) - bg_row[x]);
  mask_line[x] = diff > threshold ? 255 : 0;
}

__device__ inline uint8_t max_u8(uint8_t a, uint8_t b)
{
  return a > b ? a : b;
}

__device__ inline uint8_t min_u8(uint8_t a, uint8_t b)
{
  return a < b ? a : b;
}

__global__ void morph(ImageView<uint8_t> src, ImageView<uint8_t> dst, bool dilate)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= src.width || y >= src.height)
    return;

  uint8_t value = dilate ? 0 : 255;
  for (int dy = -1; dy <= 1; ++dy)
  {
    int sy = y + dy;
    if (sy < 0 || sy >= src.height)
      continue;
    auto src_line = mask_row(src, sy);
    for (int dx = -1; dx <= 1; ++dx)
    {
      int sx = x + dx;
      if (sx < 0 || sx >= src.width)
        continue;
      value = dilate ? max_u8(value, src_line[sx]) : min_u8(value, src_line[sx]);
    }
  }
  auto dst_line = mask_row(dst, y);
  dst_line[x] = value;
}

__global__ void apply_mask(ImageView<rgb8> frame, ImageView<uint8_t> mask)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= frame.width || y >= frame.height)
    return;

  auto row = reinterpret_cast<rgb8*>((uint8_t*)frame.buffer + y * frame.stride);
  auto mask_line = mask_row(mask, y);
  if (mask_line[x] > 0)
  {
    row[x].r = 255;
    row[x].g = 0;
    row[x].b = 0;
  }
}

} // namespace

void compute_cu(ImageView<rgb8> in)
{
  ensure_device_buffers(in);

  Image<rgb8> device_frame(in.width, in.height, true);
  CUDA_CHECK(cudaMemcpy2D(device_frame.buffer, device_frame.stride, in.buffer, in.stride,
                           in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

  bool initialize = g_device.frame_count == 0;
  float alpha = 0.05f;
  float threshold = 20.f;

  ImageView<float> background_view{g_device.background.buffer, in.width, in.height, g_device.background.stride};
  ImageView<uint8_t> mask_view{g_device.mask.buffer, in.width, in.height, g_device.mask.stride};
  ImageView<uint8_t> temp_view{g_device.temp.buffer, in.width, in.height, g_device.temp.stride};
  ImageView<rgb8> frame_view{device_frame.buffer, in.width, in.height, device_frame.stride};

  update_background<<<grid, block>>>(frame_view, background_view, alpha, initialize);
  CUDA_KERNEL_CHECK();

  build_mask<<<grid, block>>>(frame_view, background_view, mask_view, threshold);
  CUDA_KERNEL_CHECK();

  morph<<<grid, block>>>(mask_view, temp_view, false);
  CUDA_KERNEL_CHECK();
  morph<<<grid, block>>>(temp_view, mask_view, true);
  CUDA_KERNEL_CHECK();
  morph<<<grid, block>>>(mask_view, temp_view, true);
  CUDA_KERNEL_CHECK();
  morph<<<grid, block>>>(temp_view, mask_view, false);
  CUDA_KERNEL_CHECK();

  apply_mask<<<grid, block>>>(frame_view, mask_view);
  CUDA_KERNEL_CHECK();

  CUDA_CHECK(cudaMemcpy2D(in.buffer, in.stride, device_frame.buffer, device_frame.stride,
                           in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost));

  g_device.frame_count++;
}
