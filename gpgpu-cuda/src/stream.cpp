#include <gst/gst.h>
#include "StreamRunner.hpp"
#include "argh.h"

#include <string>

int main(int argc, char* argv[])
{
  argh::parser cmdl(argc, argv);
  if (cmdl[{"-h", "--help"}])
  {
    g_printerr("Usage: %s --mode=[gpu,cpu] <filename> [--output=output.mp4] bg=<path> opening_size=<size> th_low=<int> th_high=<int> bg_sampling_rate=<ms> bg_number_frame=<frames>\n", argv[0]);
    return 0;
  }

  auto parse_int = [&](const char* key, int def) {
    int value = def;
    auto stream = cmdl(key, def);
    if (!(stream >> value))
      value = def;
    return value;
  };

  const std::string filename = cmdl(1).str();
  if (filename.empty())
  {
    g_printerr("Missing video filename\n");
    return 1;
  }

  const std::string method = cmdl("mode", "cpu").str();
  if (method != "cpu" && method != "gpu")
  {
    g_printerr("Invalid method: %s\n", method.c_str());
    return 1;
  }

  StreamOptions opts;
  opts.mode = method;
  opts.filename = filename;
  opts.output = cmdl({"-o", "--output"}, "").str();
  opts.background = cmdl("bg", "").str();
  opts.opening_size = parse_int("opening_size", 3);
  opts.th_low = parse_int("th_low", 3);
  opts.th_high = parse_int("th_high", 30);
  opts.bg_sampling_rate = parse_int("bg_sampling_rate", 500);
  opts.bg_number_frame = parse_int("bg_number_frame", 10);

  g_debug("Using method: %s", method.c_str());
  return run_stream(opts);
}
