#pragma once

#include "Compute.hpp"

#include <string>

/// Run the streaming pipeline using the provided parameters.
int run_stream(const Parameters& params,
               const std::string& mode,
               const std::string& filename,
               const std::string& output,
               const std::string& background);

extern "C" int run_stream_c(const char* mode,
                            const char* filename,
                            const char* output,
                            const char* background,
                            int opening_size,
                            int th_low,
                            int th_high,
                            int bg_sampling_rate,
                            int bg_number_frame);
