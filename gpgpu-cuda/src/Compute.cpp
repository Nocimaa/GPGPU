#include "Compute.hpp"
#include "Image.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>

namespace {

struct CpuState
{
  Image<float> background;
  Image<uint8_t> mask;
  Image<uint8_t> temp;
  size_t frame_count = 0;
};

static CpuState g_cpu;

inline float luminance(const rgb8& px)
{
  return px.r * 0.299f + px.g * 0.587f + px.b * 0.114f;
}

inline void ensure_cpu_buffers(ImageView<rgb8> in)
{
  if (g_cpu.background.width != in.width || g_cpu.background.height != in.height)
  {
    g_cpu.background = Image<float>(in.width, in.height);
    g_cpu.mask = Image<uint8_t>(in.width, in.height);
    g_cpu.temp = Image<uint8_t>(in.width, in.height);
    g_cpu.frame_count = 0;
  }
}

void erode_mask(const ImageView<uint8_t>& src, const ImageView<uint8_t>& dst)
{
  for (int y = 0; y < src.height; ++y)
  {
    auto dst_line = (uint8_t*)((std::byte*)dst.buffer + y * dst.stride);
    for (int x = 0; x < src.width; ++x)
    {
      uint8_t value = 255;
      for (int dy = -1; dy <= 1; ++dy)
      {
        int sy = y + dy;
        if (sy < 0 || sy >= src.height)
          continue;
        auto src_line = (uint8_t*)((std::byte*)src.buffer + sy * src.stride);
        for (int dx = -1; dx <= 1; ++dx)
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

void dilate_mask(const ImageView<uint8_t>& src, const ImageView<uint8_t>& dst)
{
  for (int y = 0; y < src.height; ++y)
  {
    auto dst_line = (uint8_t*)((std::byte*)dst.buffer + y * dst.stride);
    for (int x = 0; x < src.width; ++x)
    {
      uint8_t value = 0;
      for (int dy = -1; dy <= 1; ++dy)
      {
        int sy = y + dy;
        if (sy < 0 || sy >= src.height)
          continue;
        auto src_line = (uint8_t*)((std::byte*)src.buffer + sy * src.stride);
        for (int dx = -1; dx <= 1; ++dx)
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

void compute_cpp(ImageView<rgb8> in)
{
  ensure_cpu_buffers(in);

  const float alpha = 0.05f;
  const float threshold = 20.f;

  for (int y = 0; y < in.height; ++y)
  {
    auto src_line = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    auto bg_line = (float*)((std::byte*)g_cpu.background.buffer + y * g_cpu.background.stride);
    auto mask_line = (uint8_t*)((std::byte*)g_cpu.mask.buffer + y * g_cpu.mask.stride);
    for (int x = 0; x < in.width; ++x)
    {
      float lum = luminance(src_line[x]);
      if (g_cpu.frame_count == 0)
      {
        bg_line[x] = lum;
      }
      else
      {
        bg_line[x] = bg_line[x] * (1.f - alpha) + lum * alpha;
      }

      float diff = std::fabs(lum - bg_line[x]);
      mask_line[x] = diff > threshold ? 255 : 0;
    }
  }

  erode_mask(g_cpu.mask, g_cpu.temp);
  dilate_mask(g_cpu.temp, g_cpu.mask);
  dilate_mask(g_cpu.mask, g_cpu.temp);
  erode_mask(g_cpu.temp, g_cpu.mask);

  apply_mask(in, g_cpu.mask);
  g_cpu.frame_count++;
}

} // namespace

/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

extern "C" {

  static Parameters g_params;

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
