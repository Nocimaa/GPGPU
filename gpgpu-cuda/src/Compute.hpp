#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
    extern "C" {
#endif


// Execution parameters taken from the command line
typedef enum {
    CPU,
    GPU
} e_device_t;


typedef struct  {
    e_device_t device;
    bool opt_gpu_diff;
    bool opt_gpu_hysteresis;
    bool opt_gpu_morphology;
    bool opt_gpu_background;
    bool opt_gpu_overlay;
    bool opt_kernel_fusion;
    bool opt_cpu_simd;
    int opening_size;
    int th_low;
    int th_high;
    int bg_sampling_rate;
    int bg_number_frame;
} Parameters;

/// Global state initialization
/// This function is called once before any other cpt_* function at the beginning of the program
void cpt_init(Parameters* params);

/// Function called by gstreamer to process the incoming frame
void cpt_process_frame(uint8_t* pixels, int width, int height, int stride);
    

#ifdef __cplusplus
    }

#endif
