// Create a GStreamer pipeline to stream a video through our plugin

#include <gst/gst.h>
#include "StreamEngine.h"
#include "argh.h"


int main(int argc, char* argv[])
{
  argh::parser cmdl(argc, argv);
  if (cmdl[{"-h", "--help"}])
  {
    g_printerr("Usage: %s --mode=[gpu,cpu] <filename> [--output=output.mp4]\n", argv[0]);
    return 0;
  }

  Parameters params;
  auto method = cmdl("mode", "cpu").str();
  auto filename = cmdl(1).str();
  auto output = cmdl({"-o", "--output"}, "").str();
  if (method == "cpu")
    params.device = e_device_t::CPU;
  else if (method == "gpu")
    params.device = e_device_t::GPU;
  else
  {
    g_printerr("Invalid method: %s\n", method.c_str());
    return 1;
  }

  auto parse_bool_option = [&](const char* name, bool default_value)
  {
    auto value = cmdl(name, default_value ? "1" : "0").str();
    return value == "1" || value == "true";
  };
  auto parse_int_option = [&](const char* name, int default_value)
  {
    auto stream = cmdl(name, default_value);
    int value = default_value;
    if (stream && (stream >> value))
      return value;
    return default_value;
  };

  params.opt_gpu_diff = parse_bool_option("--gpu-diff", params.device == e_device_t::GPU);
  params.opt_gpu_hysteresis = parse_bool_option("--gpu-hysteresis", params.device == e_device_t::GPU);
  params.opt_gpu_morphology = parse_bool_option("--gpu-morphology", params.device == e_device_t::GPU);
  params.opt_gpu_background = parse_bool_option("--gpu-background", params.device == e_device_t::GPU);
  params.opt_gpu_overlay = parse_bool_option("--gpu-overlay", params.device == e_device_t::GPU);
  params.opt_kernel_fusion = parse_bool_option("--kernel-fusion", false);
  params.opt_cpu_simd = parse_bool_option("--cpu-simd", false);
  params.opening_size = parse_int_option("opening_size", 3);
  params.th_low = parse_int_option("th_low", 3);
  params.th_high = parse_int_option("th_high", 30);
  params.bg_sampling_rate = parse_int_option("bg_sampling_rate", 500);
  params.bg_number_frame = parse_int_option("bg_number_frame", 10);

  g_debug("Using method: %s", method);
  printf("Processing file: %s\n", method);
  return run_stream(params, method, filename, output, "");
}
