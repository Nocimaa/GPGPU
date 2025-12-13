#include "Compute.hpp"
#include "Image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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
  Image<uint8_t> prev_mask;
  size_t frame_count = 0;
  bool manual_background = false;
  bool background_initialized = false;
  std::chrono::steady_clock::time_point last_bg_update;
};

static DeviceState g_device;

inline bool has_background_uri()
{
  return g_params.bg_uri != nullptr && g_params.bg_uri[0] != '\0';
}

inline int normalized_radius(int opening_size)
{
  int width = std::max(1, opening_size);
  int radius = width / 2;
  return std::max(1, radius);
}

void ensure_device_buffers(ImageView<rgb8> in)
{
  if (g_device.background.width != in.width || g_device.background.height != in.height)
  {
    g_device.background = Image<float>(in.width, in.height, true);
    g_device.mask = Image<uint8_t>(in.width, in.height, true);
    g_device.temp = Image<uint8_t>(in.width, in.height, true);
    g_device.prev_mask = Image<uint8_t>(in.width, in.height, true);
    g_device.frame_count = 0;
    g_device.manual_background = false;
    g_device.background_initialized = false;
    g_device.last_bg_update = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemset(g_device.mask.buffer, 0, g_device.mask.stride * g_device.mask.height));
    CUDA_CHECK(cudaMemset(g_device.prev_mask.buffer, 0, g_device.prev_mask.stride * g_device.prev_mask.height));
    CUDA_CHECK(cudaMemset(g_device.temp.buffer, 0, g_device.temp.stride * g_device.temp.height));
  }
}

bool should_update_background(bool& initialize)
{
  if (g_device.manual_background)
  {
    initialize = false;
    return false;
  }

  const int interval_ms = std::max(1, g_params.bg_sampling_rate_ms);
  const auto interval = std::chrono::milliseconds(interval_ms);
  auto now = std::chrono::steady_clock::now();

  if (!g_device.background_initialized)
  {
    initialize = true;
    g_device.background_initialized = true;
    g_device.last_bg_update = now;
    return true;
  }

  initialize = false;
  if (now - g_device.last_bg_update >= interval)
  {
    g_device.last_bg_update = now;
    return true;
  }

  return false;
}

void load_manual_background(ImageView<rgb8> in)
{
  if (g_device.manual_background || !has_background_uri())
    return;

  Image<rgb8> bg_image(g_params.bg_uri);
  if (!bg_image.buffer || bg_image.width != in.width || bg_image.height != in.height)
    return;

  Image<float> host_background(in.width, in.height);
  for (int y = 0; y < in.height; ++y)
  {
    auto src_line = (rgb8*)((std::byte*)bg_image.buffer + y * bg_image.stride);
    auto dst_line = (float*)((std::byte*)host_background.buffer + y * host_background.stride);
    for (int x = 0; x < in.width; ++x)
      dst_line[x] = src_line[x].r * 0.299f + src_line[x].g * 0.587f + src_line[x].b * 0.114f;
  }

  CUDA_CHECK(cudaMemcpy2D(g_device.background.buffer, g_device.background.stride,
                           host_background.buffer, host_background.stride,
                           in.width * sizeof(float), in.height,
                           cudaMemcpyHostToDevice));
  g_device.manual_background = true;
  g_device.background_initialized = true;
  g_device.last_bg_update = std::chrono::steady_clock::now();
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

__global__ void build_mask(ImageView<rgb8> frame, ImageView<float> background, ImageView<uint8_t> mask, ImageView<uint8_t> prev_mask, int th_low, int th_high, bool has_prev)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= frame.width || y >= frame.height)
    return;

  auto row = reinterpret_cast<rgb8*>((uint8_t*)frame.buffer + y * frame.stride);
  auto bg_row = background_row(background, y);
  auto mask_line = mask_row(mask, y);
  auto prev_line = mask_row(prev_mask, y);

  float diff = fabsf(luminance(row[x]) - bg_row[x]);
  uint8_t prev_value = has_prev ? prev_line[x] : 0;
  if (diff >= th_high)
    mask_line[x] = 255;
  else if (diff <= th_low)
    mask_line[x] = 0;
  else
    mask_line[x] = prev_value;
}

__device__ inline uint8_t max_u8(uint8_t a, uint8_t b)
{
  return a > b ? a : b;
}

__device__ inline uint8_t min_u8(uint8_t a, uint8_t b)
{
  return a < b ? a : b;
}

__global__ void morph(ImageView<uint8_t> src, ImageView<uint8_t> dst, int radius, bool dilate)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= src.width || y >= src.height)
    return;

  uint8_t value = dilate ? 0 : 255;
  for (int dy = -radius; dy <= radius; ++dy)
  {
    int sy = y + dy;
    if (sy < 0 || sy >= src.height)
      continue;
    auto src_line = mask_row(src, sy);
    for (int dx = -radius; dx <= radius; ++dx)
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

void copy_mask_gpu(ImageView<uint8_t> src, ImageView<uint8_t> dst)
{
  CUDA_CHECK(cudaMemcpy2D(dst.buffer, dst.stride, src.buffer, src.stride, src.width, src.height, cudaMemcpyDeviceToDevice));
}

} // namespace

void compute_cu(ImageView<rgb8> in)
{
  ensure_device_buffers(in);
  load_manual_background(in);

  Image<rgb8> device_frame(in.width, in.height, true);
  CUDA_CHECK(cudaMemcpy2D(device_frame.buffer, device_frame.stride, in.buffer, in.stride,
                           in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

  bool initialize = false;
  bool update_bg = should_update_background(initialize);
  const float alpha = 1.f / std::max(1, g_params.bg_number_frame);
  const int th_low = std::max(0, g_params.th_low);
  const int th_high = std::max(th_low, g_params.th_high);
  const int radius = normalized_radius(g_params.opening_size);

  ImageView<float> background_view{g_device.background.buffer, in.width, in.height, g_device.background.stride};
  ImageView<uint8_t> mask_view{g_device.mask.buffer, in.width, in.height, g_device.mask.stride};
  ImageView<uint8_t> temp_view{g_device.temp.buffer, in.width, in.height, g_device.temp.stride};
  ImageView<uint8_t> prev_view{g_device.prev_mask.buffer, in.width, in.height, g_device.prev_mask.stride};
  ImageView<rgb8> frame_view{device_frame.buffer, in.width, in.height, device_frame.stride};

  if (update_bg)
  {
    update_background<<<grid, block>>>(frame_view, background_view, alpha, initialize);
    CUDA_KERNEL_CHECK();
  }

  build_mask<<<grid, block>>>(frame_view, background_view, mask_view, prev_view, th_low, th_high, g_device.frame_count > 0);
  CUDA_KERNEL_CHECK();

  morph<<<grid, block>>>(mask_view, temp_view, radius, false);
  CUDA_KERNEL_CHECK();
  morph<<<grid, block>>>(temp_view, mask_view, radius, true);
  CUDA_KERNEL_CHECK();
  morph<<<grid, block>>>(mask_view, temp_view, radius, true);
  CUDA_KERNEL_CHECK();
  morph<<<grid, block>>>(temp_view, mask_view, radius, false);
  CUDA_KERNEL_CHECK();

  copy_mask_gpu(mask_view, prev_view);

  apply_mask<<<grid, block>>>(frame_view, mask_view);
  CUDA_KERNEL_CHECK();

  CUDA_CHECK(cudaMemcpy2D(in.buffer, in.stride, device_frame.buffer, device_frame.stride,
                           in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost));

  g_device.frame_count++;
}
