#include "Compute.hpp"
#include "Image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

Parameters g_params = { e_device_t::CPU, nullptr, 3, 3, 30, 500, 10 };

namespace {

struct CpuState
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

static CpuState g_cpu;

inline float luminance(const rgb8& px)
{
  return px.r * 0.299f + px.g * 0.587f + px.b * 0.114f;
}

inline int normalized_radius(int opening_size)
{
  int width = std::max(1, opening_size);
  int radius = width / 2;
  return std::max(1, radius);
}

inline bool has_background_uri()
{
  return g_params.bg_uri != nullptr && g_params.bg_uri[0] != '\0';
}

void ensure_cpu_buffers(ImageView<rgb8> in)
{
  bool size_changed = g_cpu.background.width != in.width || g_cpu.background.height != in.height;
  if (size_changed)
  {
    g_cpu.background = Image<float>(in.width, in.height);
    g_cpu.mask = Image<uint8_t>(in.width, in.height);
    g_cpu.temp = Image<uint8_t>(in.width, in.height);
    g_cpu.prev_mask = Image<uint8_t>(in.width, in.height);
    g_cpu.frame_count = 0;
    g_cpu.manual_background = false;
    g_cpu.background_initialized = false;
    g_cpu.last_bg_update = std::chrono::steady_clock::now();
    std::memset(g_cpu.mask.buffer, 0, g_cpu.mask.height * g_cpu.mask.stride);
    std::memset(g_cpu.prev_mask.buffer, 0, g_cpu.prev_mask.height * g_cpu.prev_mask.stride);
    std::memset(g_cpu.temp.buffer, 0, g_cpu.temp.height * g_cpu.temp.stride);
  }
}

bool load_manual_background(ImageView<rgb8> in)
{
  if (g_cpu.manual_background || !has_background_uri())
    return g_cpu.manual_background;

  Image<rgb8> bg_image(g_params.bg_uri);
  if (!bg_image.buffer || bg_image.width != in.width || bg_image.height != in.height)
    return false;

  for (int y = 0; y < in.height; ++y)
  {
    auto src_line = (rgb8*)((std::byte*)bg_image.buffer + y * bg_image.stride);
    auto dst_line = (float*)((std::byte*)g_cpu.background.buffer + y * g_cpu.background.stride);
    for (int x = 0; x < in.width; ++x)
      dst_line[x] = luminance(src_line[x]);
  }

  g_cpu.manual_background = true;
  g_cpu.background_initialized = true;
  g_cpu.last_bg_update = std::chrono::steady_clock::now();
  return true;
}

bool should_update_background()
{
  if (g_cpu.manual_background)
    return false;

  const int interval_ms = std::max(1, g_params.bg_sampling_rate_ms);
  const auto interval = std::chrono::milliseconds(interval_ms);
  auto now = std::chrono::steady_clock::now();

  if (!g_cpu.background_initialized)
  {
    g_cpu.background_initialized = true;
    g_cpu.last_bg_update = now;
    return true;
  }

  if (now - g_cpu.last_bg_update >= interval)
  {
    g_cpu.last_bg_update = now;
    return true;
  }

  return false;
}

void update_background(ImageView<rgb8> in, float alpha, bool initialize)
{
  for (int y = 0; y < in.height; ++y)
  {
    auto src_line = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    auto bg_line = (float*)((std::byte*)g_cpu.background.buffer + y * g_cpu.background.stride);
    for (int x = 0; x < in.width; ++x)
    {
      float lum = luminance(src_line[x]);
      if (initialize)
        bg_line[x] = lum;
      else
        bg_line[x] = bg_line[x] * (1.f - alpha) + lum * alpha;
    }
  }
}

void erode_mask(const ImageView<uint8_t>& src, const ImageView<uint8_t>& dst, int radius)
{
  for (int y = 0; y < src.height; ++y)
  {
    auto dst_line = (uint8_t*)((std::byte*)dst.buffer + y * dst.stride);
    for (int x = 0; x < src.width; ++x)
    {
      uint8_t value = 255;
      for (int dy = -radius; dy <= radius; ++dy)
      {
        int sy = y + dy;
        if (sy < 0 || sy >= src.height)
          continue;
        auto src_line = (uint8_t*)((std::byte*)src.buffer + sy * src.stride);
        for (int dx = -radius; dx <= radius; ++dx)
        {
          int sx = x + dx;
          if (sx < 0 || sx >= src.width)
            continue;
          value = std::min<uint8_t>(value, src_line[sx]);
        }
      }
      dst_line[x] = value;
    }
  }
}

void dilate_mask(const ImageView<uint8_t>& src, const ImageView<uint8_t>& dst, int radius)
{
  for (int y = 0; y < src.height; ++y)
  {
    auto dst_line = (uint8_t*)((std::byte*)dst.buffer + y * dst.stride);
    for (int x = 0; x < src.width; ++x)
    {
      uint8_t value = 0;
      for (int dy = -radius; dy <= radius; ++dy)
      {
        int sy = y + dy;
        if (sy < 0 || sy >= src.height)
          continue;
        auto src_line = (uint8_t*)((std::byte*)src.buffer + sy * src.stride);
        for (int dx = -radius; dx <= radius; ++dx)
        {
          int sx = x + dx;
          if (sx < 0 || sx >= src.width)
            continue;
          value = std::max<uint8_t>(value, src_line[sx]);
        }
      }
      dst_line[x] = value;
    }
  }
}

void apply_mask(ImageView<rgb8> image, const ImageView<uint8_t>& mask)
{
  for (int y = 0; y < image.height; ++y)
  {
    auto line = (rgb8*)((std::byte*)image.buffer + y * image.stride);
    auto mask_line = (uint8_t*)((std::byte*)mask.buffer + y * mask.stride);
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

void copy_mask(ImageView<uint8_t> src, ImageView<uint8_t> dst)
{
  for (int y = 0; y < src.height; ++y)
  {
    auto src_line = (uint8_t*)((std::byte*)src.buffer + y * src.stride);
    auto dst_line = (uint8_t*)((std::byte*)dst.buffer + y * dst.stride);
    std::memcpy(dst_line, src_line, src.width);
  }
}

void compute_cpp(ImageView<rgb8> in)
{
  ensure_cpu_buffers(in);
  load_manual_background(in);

  const float alpha = 1.f / std::max(1, g_params.bg_number_frame);
  const int radius = normalized_radius(g_params.opening_size);
  const int th_low = std::max(0, g_params.th_low);
  const int th_high = std::max(th_low, g_params.th_high);

  if (should_update_background())
    update_background(in, alpha, g_cpu.frame_count == 0);

  auto mask_view = ImageView<uint8_t>{g_cpu.mask.buffer, in.width, in.height, g_cpu.mask.stride};
  auto prev_view = ImageView<uint8_t>{g_cpu.prev_mask.buffer, in.width, in.height, g_cpu.prev_mask.stride};

  for (int y = 0; y < in.height; ++y)
  {
    auto src_line = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    auto mask_line = (uint8_t*)((std::byte*)mask_view.buffer + y * mask_view.stride);
    auto prev_line = (uint8_t*)((std::byte*)prev_view.buffer + y * prev_view.stride);
    auto bg_row = (float*)((std::byte*)g_cpu.background.buffer + y * g_cpu.background.stride);
    for (int x = 0; x < in.width; ++x)
    {
      float diff = std::fabs(luminance(src_line[x]) - bg_row[x]);
      if (diff >= th_high)
        mask_line[x] = 255;
      else if (diff <= th_low)
        mask_line[x] = 0;
      else
        mask_line[x] = prev_line[x];
    }
  }

  erode_mask(mask_view, g_cpu.temp, radius);
  dilate_mask(g_cpu.temp, mask_view, radius);
  dilate_mask(mask_view, g_cpu.temp, radius);
  erode_mask(g_cpu.temp, mask_view, radius);

  copy_mask(mask_view, prev_view);
  apply_mask(in, mask_view);
  g_cpu.frame_count++;
}

} // namespace

/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

extern "C" {

  void cpt_init(Parameters* params)
  {
    g_params = *params;
  }

  void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
  {
    auto img = ImageView<rgb8>{(rgb8*)buffer, width, height, stride};
    if (g_params.device == e_device_t::CPU)
      compute_cpp(img);
    else if (g_params.device == e_device_t::GPU)
      compute_cu(img);
  }

}
