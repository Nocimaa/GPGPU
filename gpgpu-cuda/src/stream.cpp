#include <gst/gst.h>
#include "argh.h"

#include "StreamEngine.hpp"


int main(int argc, char* argv[])
{
  argh::parser cmdl(argc, argv);
  if (cmdl[{"-h", "--help"}])
  {
    g_printerr("Usage: %s --mode=[gpu,cpu] <filename> [--output=output.mp4]\n", argv[0]);
    return 0;
  }

  StreamOptions opts;
  auto method = cmdl("mode", "cpu").str();
  auto filename = cmdl(1).str();
  auto output = cmdl({"-o", "--output"}, "").str();

  if (method == "cpu")
    opts.params.device = e_device_t::CPU;
  else if (method == "gpu")
    opts.params.device = e_device_t::GPU;
  else
  {
    g_printerr("Invalid method: %s\n", method.c_str());
    return 1;
  }

  g_debug("Using method: %s", method.c_str());
  opts.video_path = filename;
  opts.output_path = output;

  return run_stream(opts);
}
