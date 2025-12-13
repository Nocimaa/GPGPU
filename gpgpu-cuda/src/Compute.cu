#include "Compute.hpp"
#include "Image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <curand_kernel.h>
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

constexpr int kReservoirSize = 4;

struct DeviceState
{
  Image<uint8_t> mask;
  Image<uint8_t> temp;
  Image<uint8_t> prev_mask;
  Image<uint8_t> clean_prev;
  Image<uint8_t> host_mask;
  Image<float> reservoir_values;
  Image<float> reservoir_weights;
  Image<curandState> rng_states;
  Image<unsigned int> change_flag;
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

__global__ void init_rng_states(ImageView<curandState> states, unsigned long long seed);

void ensure_device_buffers(ImageView<rgb8> in)
{
  if (g_device.mask.width != in.width || g_device.mask.height != in.height)
  {
    int reservoir_width = in.width * kReservoirSize;
    g_device.mask = Image<uint8_t>(in.width, in.height, true);
    g_device.temp = Image<uint8_t>(in.width, in.height, true);
    g_device.prev_mask = Image<uint8_t>(in.width, in.height, true);
    g_device.clean_prev = Image<uint8_t>(in.width, in.height, true);
    g_device.host_mask = Image<uint8_t>(in.width, in.height);
    g_device.reservoir_values = Image<float>(reservoir_width, in.height, true);
    g_device.reservoir_weights = Image<float>(reservoir_width, in.height, true);
    g_device.rng_states = Image<curandState>(in.width, in.height, true);
    g_device.change_flag = Image<unsigned int>(1, 1, true);
    g_device.frame_count = 0;
    g_device.manual_background = false;
    g_device.background_initialized = false;
    g_device.last_bg_update = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemset(g_device.mask.buffer, 0, g_device.mask.stride * g_device.mask.height));
    CUDA_CHECK(cudaMemset(g_device.prev_mask.buffer, 0, g_device.prev_mask.stride * g_device.prev_mask.height));
    CUDA_CHECK(cudaMemset(g_device.temp.buffer, 0, g_device.temp.stride * g_device.temp.height));
    CUDA_CHECK(cudaMemset(g_device.clean_prev.buffer, 0, g_device.clean_prev.stride * g_device.clean_prev.height));
    CUDA_CHECK(cudaMemset(g_device.reservoir_values.buffer, 0, g_device.reservoir_values.stride * g_device.reservoir_values.height));
    CUDA_CHECK(cudaMemset(g_device.reservoir_weights.buffer, 0, g_device.reservoir_weights.stride * g_device.reservoir_weights.height));
    CUDA_CHECK(cudaMemset(g_device.change_flag.buffer, 0, g_device.change_flag.stride * g_device.change_flag.height));
    std::memset(g_device.host_mask.buffer, 0, g_device.host_mask.stride * g_device.host_mask.height);

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    ImageView<curandState> rng_view{g_device.rng_states.buffer, in.width, in.height, g_device.rng_states.stride};
    unsigned long long seed = std::chrono::steady_clock::now().time_since_epoch().count();
    init_rng_states<<<grid, block>>>(rng_view, seed);
    CUDA_KERNEL_CHECK();
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

  int reservoir_width = in.width * kReservoirSize;
  Image<float> host_reservoir(reservoir_width, in.height);
  Image<float> host_weights(reservoir_width, in.height);

  for (int y = 0; y < in.height; ++y)
  {
    auto src_line = (rgb8*)((std::byte*)bg_image.buffer + y * bg_image.stride);
    auto reservoir_line = (float*)((std::byte*)host_reservoir.buffer + y * host_reservoir.stride);
    auto weight_line = (float*)((std::byte*)host_weights.buffer + y * host_weights.stride);
    for (int x = 0; x < in.width; ++x)
    {
      float lum = src_line[x].r * 0.299f + src_line[x].g * 0.587f + src_line[x].b * 0.114f;
      for (int k = 0; k < kReservoirSize; ++k)
      {
        reservoir_line[x * kReservoirSize + k] = lum;
        weight_line[x * kReservoirSize + k] = 1.f;
      }
    }
  }

  ImageView<float> reservoir_view{g_device.reservoir_values.buffer, reservoir_width, in.height, g_device.reservoir_values.stride};
  ImageView<float> weight_view{g_device.reservoir_weights.buffer, reservoir_width, in.height, g_device.reservoir_weights.stride};

  CUDA_CHECK(cudaMemcpy2D(reservoir_view.buffer, reservoir_view.stride,
                           host_reservoir.buffer, host_reservoir.stride,
                           reservoir_width * sizeof(float), in.height,
                           cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy2D(weight_view.buffer, weight_view.stride,
                           host_weights.buffer, host_weights.stride,
                           reservoir_width * sizeof(float), in.height,
                           cudaMemcpyHostToDevice));

  g_device.manual_background = true;
  g_device.background_initialized = true;
  g_device.last_bg_update = std::chrono::steady_clock::now();
}

__device__ inline float luminance(const rgb8& px)
{
  return px.r * 0.299f + px.g * 0.587f + px.b * 0.114f;
}

__device__ inline float* reservoir_row(const ImageView<float>& view, int y)
{
  return reinterpret_cast<float*>((uint8_t*)view.buffer + y * view.stride);
}

__device__ inline uint8_t* mask_row(const ImageView<uint8_t>& view, int y)
{
  return reinterpret_cast<uint8_t*>((uint8_t*)view.buffer + y * view.stride);
}

__device__ inline curandState* rng_row(const ImageView<curandState>& view, int y)
{
  return reinterpret_cast<curandState*>((uint8_t*)view.buffer + y * view.stride);
}

__global__ void init_rng_states(ImageView<curandState> states, unsigned long long seed)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= states.width || y >= states.height)
    return;

  int idx = y * states.width + x;
  curandState* row = rng_row(states, y);
  curand_init(seed, idx, 0, &row[x]);
}

__global__ void update_reservoir_background(ImageView<rgb8> frame,
                                             ImageView<float> reservoir,
                                             ImageView<float> weights,
                                             ImageView<curandState> rng,
                                             int reservoir_size,
                                             float match_radius,
                                             bool initialize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= frame.width || y >= frame.height)
    return;

  auto frame_row = reinterpret_cast<rgb8*>((uint8_t*)frame.buffer + y * frame.stride);
  float lum = luminance(frame_row[x]);
  float* reservoir_row_ptr = reservoir_row(reservoir, y);
  float* weight_row_ptr = reservoir_row(weights, y);
  curandState* rng_row_ptr = rng_row(rng, y);
  curandState* state = &rng_row_ptr[x];
  int base = x * reservoir_size;

  if (initialize)
  {
    for (int i = 0; i < reservoir_size; ++i)
    {
      reservoir_row_ptr[base + i] = lum;
      weight_row_ptr[base + i] = 1.f;
    }
    return;
  }

  float total_weight = 0.f;
  int match_idx = -1;
  for (int i = 0; i < reservoir_size; ++i)
  {
    float value = reservoir_row_ptr[base + i];
    float diff = fabsf(lum - value);
    total_weight += weight_row_ptr[base + i];
    if (diff <= match_radius)
      match_idx = i;
  }

  if (match_idx >= 0)
  {
    float* entry = &reservoir_row_ptr[base + match_idx];
    float* weight = &weight_row_ptr[base + match_idx];
    *entry = *entry * 0.85f + lum * 0.15f;
    *weight = fminf(*weight + 1.f, 255.f);
    return;
  }

  int target = 0;
  if (total_weight > 0.f)
  {
    float threshold = curand_uniform(state) * total_weight;
    float accumulated = 0.f;
    for (int i = 0; i < reservoir_size; ++i)
    {
      accumulated += weight_row_ptr[base + i];
      if (accumulated >= threshold)
      {
        target = i;
        break;
      }
    }
  }
  else
  {
    target = curand(state) % reservoir_size;
  }

  reservoir_row_ptr[base + target] = lum;
  weight_row_ptr[base + target] = 1.f;
}

__global__ void build_mask(ImageView<rgb8> frame, ImageView<float> reservoir, ImageView<uint8_t> mask, ImageView<uint8_t> prev_mask, int th_low, int th_high, bool has_prev, int reservoir_size)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= frame.width || y >= frame.height)
    return;

  auto row = reinterpret_cast<rgb8*>((uint8_t*)frame.buffer + y * frame.stride);
  float* reservoir_line = reservoir_row(reservoir, y);
  auto mask_line = mask_row(mask, y);
  auto prev_line = mask_row(prev_mask, y);

  float lum = luminance(row[x]);
  float diff = FLT_MAX;
  int base = x * reservoir_size;
  for (int i = 0; i < reservoir_size; ++i)
  {
    float value = reservoir_line[base + i];
    float candidate = fabsf(lum - value);
    diff = diff < candidate ? diff : candidate;
  }
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

__device__ inline unsigned int* change_flag_cell(const ImageView<unsigned int>& view)
{
  return reinterpret_cast<unsigned int*>((uint8_t*)view.buffer);
}

__global__ void detect_mask_changes(ImageView<uint8_t> mask, ImageView<uint8_t> prev, ImageView<unsigned int> flag)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= mask.width || y >= mask.height)
    return;

  auto mask_line = mask_row(mask, y);
  auto prev_line = mask_row(prev, y);
  if (mask_line[x] != prev_line[x])
    atomicOr(change_flag_cell(flag), 1u);
}

void copy_mask_gpu(ImageView<uint8_t> src, ImageView<uint8_t> dst)
{
  CUDA_CHECK(cudaMemcpy2D(dst.buffer, dst.stride, src.buffer, src.stride, src.width, src.height, cudaMemcpyDeviceToDevice));
}

void apply_mask_host(ImageView<rgb8> image, ImageView<uint8_t> mask)
{
  for (int y = 0; y < image.height; ++y)
  {
    auto line = (rgb8*)((uint8_t*)image.buffer + y * image.stride);
    auto mask_line = (uint8_t*)((uint8_t*)mask.buffer + y * mask.stride);
    for (int x = 0; x < image.width; ++x)
    {
      if (mask_line[x] > 0)
      {
        line[x].r = 255;
        line[x].g = 0;
        line[x].b = 0;
      }
    }
  }
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

  int reservoir_width = in.width * kReservoirSize;
  ImageView<float> reservoir_view{g_device.reservoir_values.buffer, reservoir_width, in.height, g_device.reservoir_values.stride};
  ImageView<float> weight_view{g_device.reservoir_weights.buffer, reservoir_width, in.height, g_device.reservoir_weights.stride};
  ImageView<curandState> rng_view{g_device.rng_states.buffer, in.width, in.height, g_device.rng_states.stride};
  ImageView<uint8_t> mask_view{g_device.mask.buffer, in.width, in.height, g_device.mask.stride};
  ImageView<uint8_t> temp_view{g_device.temp.buffer, in.width, in.height, g_device.temp.stride};
  ImageView<uint8_t> prev_view{g_device.prev_mask.buffer, in.width, in.height, g_device.prev_mask.stride};
  ImageView<rgb8> frame_view{device_frame.buffer, in.width, in.height, device_frame.stride};

  if (update_bg)
  {
    float match_radius = std::max(1, g_params.th_low);
    update_reservoir_background<<<grid, block>>>(frame_view, reservoir_view, weight_view, rng_view,
                                                 kReservoirSize, match_radius, initialize);
    CUDA_KERNEL_CHECK();
  }

  build_mask<<<grid, block>>>(frame_view, reservoir_view, mask_view, prev_view, th_low, th_high,
                              g_device.frame_count > 0, kReservoirSize);
  CUDA_KERNEL_CHECK();

  ImageView<uint8_t> clean_prev_view{g_device.clean_prev.buffer, in.width, in.height, g_device.clean_prev.stride};
  copy_mask_gpu(mask_view, clean_prev_view);
  ImageView<unsigned int> change_flag_view{g_device.change_flag.buffer, 1, 1, g_device.change_flag.stride};
  constexpr int kMaxMaskCleaningIterations = 16;
  int cleaning_iteration = 0;
  bool mask_changed;
  do
  {
    morph<<<grid, block>>>(mask_view, temp_view, radius, false);
    CUDA_KERNEL_CHECK();
    morph<<<grid, block>>>(temp_view, mask_view, radius, true);
    CUDA_KERNEL_CHECK();
    morph<<<grid, block>>>(mask_view, temp_view, radius, true);
    CUDA_KERNEL_CHECK();
    morph<<<grid, block>>>(temp_view, mask_view, radius, false);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemset(change_flag_view.buffer, 0, change_flag_view.stride * change_flag_view.height));
    detect_mask_changes<<<grid, block>>>(mask_view, clean_prev_view, change_flag_view);
    CUDA_KERNEL_CHECK();

    unsigned int flag_host = 0;
    CUDA_CHECK(cudaMemcpy(&flag_host, change_flag_view.buffer, sizeof(flag_host), cudaMemcpyDeviceToHost));
    mask_changed = flag_host != 0;
    copy_mask_gpu(mask_view, clean_prev_view);
    cleaning_iteration++;
  } while (mask_changed && cleaning_iteration < kMaxMaskCleaningIterations);

  copy_mask_gpu(mask_view, prev_view);

  ImageView<uint8_t> host_mask_view{g_device.host_mask.buffer, in.width, in.height, g_device.host_mask.stride};
  CUDA_CHECK(cudaMemcpy2D(host_mask_view.buffer, host_mask_view.stride,
                           mask_view.buffer, mask_view.stride,
                           in.width, in.height, cudaMemcpyDeviceToHost));
  apply_mask_host(in, host_mask_view);

  g_device.frame_count++;
}
