#pragma once

#include <string>

struct StreamOptions
{
  std::string mode = "cpu";
  std::string filename;
  std::string output;
  std::string background;
  int opening_size = 3;
  int th_low = 3;
  int th_high = 30;
  int bg_sampling_rate = 500;
  int bg_number_frame = 10;
};

int run_stream(const StreamOptions& options);
extern "C" int run_stream_c(const char* mode,
                            const char* filename,
                            const char* output,
                            const char* background,
                            int opening_size,
                            int th_low,
                            int th_high,
                            int bg_sampling_rate,
                            int bg_number_frame);
