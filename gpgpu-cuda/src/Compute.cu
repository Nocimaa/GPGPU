#include "Compute.hpp"
#include "Image.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// ====================
// STATE & CONFIG
// ====================
struct OptimizationConfig
{
    bool use_gpu = false;
    bool gpu_diff = false;
    bool gpu_hysteresis = false;
    bool gpu_morphology = false;
    bool gpu_background_update = false;
    bool gpu_overlay = false;
    bool kernel_fusion = false;
};

struct Timing
{
    float diff_ms = 0.f;
    float hyst_ms = 0.f;
    float morph_ms = 0.f;
    float update_ms = 0.f;
};

struct ProcessingState
{
    bool initialized = false;
    int width = 0;
    int height = 0;

    std::vector<rgb8> background;
    std::vector<uint8_t> input_mask;
    std::vector<uint8_t> marker_mask;
    std::vector<uint8_t> motion_mask;
    std::vector<uint8_t> temp_mask;

    Image<rgb8> d_background;
    Image<rgb8> d_current;
    uint8_t* d_input_mask = nullptr;
    uint8_t* d_marker_mask = nullptr;
    uint8_t* d_temp_mask = nullptr;
    uint32_t* d_changed = nullptr;
    int frame_counter = 0;

    OptimizationConfig config;
    Timing timing;
};

static ProcessingState g_state;
static Parameters g_params;

__constant__ uint8_t c_low_threshold;
__constant__ uint8_t c_high_threshold;

using Clock = std::chrono::high_resolution_clock;
using DurationMs = std::chrono::duration<float, std::milli>;

static size_t total_pixels(const ProcessingState& state)
{
    return static_cast<size_t>(state.width) * static_cast<size_t>(state.height);
}

static void reset_device_buffers(ProcessingState& state)
{
    if (state.d_input_mask)
    {
        cudaFree(state.d_input_mask);
        state.d_input_mask = nullptr;
    }
    if (state.d_marker_mask)
    {
        cudaFree(state.d_marker_mask);
        state.d_marker_mask = nullptr;
    }
    if (state.d_temp_mask)
    {
        cudaFree(state.d_temp_mask);
        state.d_temp_mask = nullptr;
    }
    if (state.d_changed)
    {
        cudaFree(state.d_changed);
        state.d_changed = nullptr;
    }
}

static void init_state(ProcessingState& state, uint8_t* buffer, int width, int height, int stride)
{
    state.width = width;
    state.height = height;
    const size_t total = total_pixels(state);

    state.background.resize(total);
    state.input_mask.resize(total);
    state.marker_mask.resize(total);
    state.motion_mask.resize(total);
    state.temp_mask.resize(total);

    for (int y = 0; y < height; ++y)
    {
        auto* dst = state.background.data() + y * width;
        auto* src = reinterpret_cast<rgb8*>((std::byte*)buffer + y * stride);
        std::memcpy(dst, src, width * sizeof(rgb8));
    }

    state.d_background = Image<rgb8>(width, height, true);
    state.d_current = Image<rgb8>(width, height, true);
    reset_device_buffers(state);
    cudaMalloc(&state.d_input_mask, width * height);
    cudaMalloc(&state.d_marker_mask, width * height);
    cudaMalloc(&state.d_temp_mask, width * height);
    cudaMalloc(&state.d_changed, sizeof(uint32_t));

    cudaMemcpy2D(state.d_background.buffer,
                 state.d_background.stride,
                 state.background.data(),
                 width * sizeof(rgb8),
                 width * sizeof(rgb8),
                 height,
                 cudaMemcpyHostToDevice);

    state.initialized = true;
    state.frame_counter = 0;
}

// ====================
// CPU IMPLEMENTATIONS
// ====================
static void sync_motion_to_marker(ProcessingState& state)
{
    std::copy(state.motion_mask.begin(), state.motion_mask.end(), state.marker_mask.begin());
}

static bool should_update_background(const ProcessingState& state)
{
    return g_params.bg_sampling_rate <= 1 || (state.frame_counter % g_params.bg_sampling_rate == 0);
}

static void compute_diff_cpu(ProcessingState& state, uint8_t* buffer, int stride)
{
    const int width = state.width;
    const int height = state.height;
    const int low_threshold = g_params.th_low;
    const int high_threshold = g_params.th_high;
    for (int y = 0; y < height; ++y)
    {
        auto* curr_row = reinterpret_cast<rgb8*>((std::byte*)buffer + y * stride);
        for (int x = 0; x < width; ++x)
        {
            const int index = y * width + x;
            const rgb8 curr = curr_row[x];
            const rgb8 bg = state.background[index];
            int diff_r = curr.r - bg.r;
            if (diff_r < 0)
                diff_r = -diff_r;
            int diff_g = curr.g - bg.g;
            if (diff_g < 0)
                diff_g = -diff_g;
            int diff_b = curr.b - bg.b;
            if (diff_b < 0)
                diff_b = -diff_b;
            int total_diff = diff_r + diff_g + diff_b;
            state.input_mask[index] = total_diff > low_threshold ? 255 : 0;
            state.marker_mask[index] = total_diff > high_threshold ? 255 : 0;
        }
    }
}

static void hysteresis_cpu(ProcessingState& state)
{
    std::copy(state.marker_mask.begin(), state.marker_mask.end(), state.motion_mask.begin());
    bool changed;
    const int width = state.width;
    const int height = state.height;
    do
    {
        changed = false;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int index = y * width + x;
                if (state.motion_mask[index] != 0 || state.input_mask[index] == 0)
                    continue;
                bool neighbor_active = false;
                for (int dy = -1; dy <= 1 && !neighbor_active; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if (dy == 0 && dx == 0)
                            continue;
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                            continue;
                        if (state.motion_mask[ny * width + nx] != 0)
                        {
                            neighbor_active = true;
                            break;
                        }
                    }
                }
                if (neighbor_active)
                {
                    state.motion_mask[index] = 255;
                    changed = true;
                }
            }
        }
    } while (changed);
    sync_motion_to_marker(state);
}

static void morphology_cpu(ProcessingState& state)
{
    const int width = state.width;
    const int height = state.height;
    const size_t total = total_pixels(state);
    if (state.temp_mask.size() < total)
        state.temp_mask.resize(total);
    const uint8_t* src = state.motion_mask.data();
    uint8_t* temp = state.temp_mask.data();

    const int opening = std::max(1, g_params.opening_size);
    const int radius = opening / 2;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int index = y * width + x;
            bool keep = true;
            for (int dy = -radius; dy <= radius && keep; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                    {
                        keep = false;
                        break;
                    }
                    if (src[ny * width + nx] != 255)
                    {
                        keep = false;
                        break;
                    }
                }
            }
            temp[index] = keep ? 255 : 0;
        }
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int index = y * width + x;
            bool any = false;
            for (int dy = -radius; dy <= radius && !any; ++dy)
            {
                for (int dx = -radius; dx <= radius && !any; ++dx)
                {
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                        continue;
                    if (temp[ny * width + nx] == 255)
                    {
                        any = true;
                        break;
                    }
                }
            }
            state.motion_mask[index] = any ? 255 : 0;
        }
    }
    sync_motion_to_marker(state);
}

static void update_background_cpu(ProcessingState& state, uint8_t* buffer, int stride)
{
    const int width = state.width;
    const int height = state.height;
    const int blend_den = std::max(1, g_params.bg_number_frame);
    for (int y = 0; y < height; ++y)
    {
        const int row_offset = y * width;
        auto* curr_row = reinterpret_cast<rgb8*>((std::byte*)buffer + y * stride);
        for (int x = 0; x < width; ++x)
        {
            const int index = row_offset + x;
            if (state.motion_mask[index] == 0)
            {
                rgb8& bg = state.background[index];
                const rgb8 curr = curr_row[x];
                bg.r = (bg.r * (blend_den - 1) + curr.r) / blend_den;
                bg.g = (bg.g * (blend_den - 1) + curr.g) / blend_den;
                bg.b = (bg.b * (blend_den - 1) + curr.b) / blend_den;
            }
        }
    }
}

static void overlay_cpu(ProcessingState& state, uint8_t* buffer, int stride)
{
    const int width = state.width;
    const int height = state.height;
    for (int y = 0; y < height; ++y)
    {
        auto* curr_row = reinterpret_cast<rgb8*>((std::byte*)buffer + y * stride);
        for (int x = 0; x < width; ++x)
        {
            if (state.motion_mask[y * width + x] == 255)
            {
                curr_row[x].r = 255;
                curr_row[x].g = 0;
                curr_row[x].b = 0;
            }
        }
    }
}

// ====================
// CUDA KERNELS
// ====================
__global__ void hysteresis_propagate(const uint8_t* mask_in,
                                     uint8_t* mask_out,
                                     const uint8_t* low_mask,
                                     int width,
                                     int height,
                                     uint32_t* changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int index = y * width + x;
    const uint8_t current = mask_in[index];
    if (current == 255)
    {
        mask_out[index] = 255;
        return;
    }

    if (low_mask[index] == 0)
    {
        mask_out[index] = 0;
        return;
    }

    bool neighbor_active = false;
    for (int dy = -1; dy <= 1 && !neighbor_active; ++dy)
    {
        for (int dx = -1; dx <= 1 && !neighbor_active; ++dx)
        {
            if (dy == 0 && dx == 0)
                continue;
            int ny = y + dy;
            int nx = x + dx;
            if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                continue;
            neighbor_active = (mask_in[ny * width + nx] == 255);
        }
    }

    if (neighbor_active)
    {
        mask_out[index] = 255;
        atomicExch(changed, 1u);
    }
    else
    {
        mask_out[index] = 0;
    }
}

__global__ void erosion_kernel(uint8_t* src, uint8_t* dst, int width, int height, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    bool keep = true;
            for (int dy = -radius; dy <= radius && keep; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int ny = y + dy;
                    int nx = x + dx;
            if (ny < 0 || ny >= height || nx < 0 || nx >= width)
            {
                keep = false;
                break;
            }
            if (src[ny * width + nx] != 255)
            {
                keep = false;
                break;
            }
        }
    }
    dst[y * width + x] = keep ? 255 : 0;
}

__global__ void dilation_kernel(uint8_t* src, uint8_t* dst, int width, int height, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    bool any = false;
            for (int dy = -radius; dy <= radius && !any; ++dy)
            {
                for (int dx = -radius; dx <= radius && !any; ++dx)
                {
            int ny = y + dy;
            int nx = x + dx;
            if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                continue;
            if (src[ny * width + nx] == 255)
            {
                any = true;
            }
        }
    }
    dst[y * width + x] = any ? 255 : 0;
}

__global__ void update_background_kernel(ImageView<rgb8> current,
                                         ImageView<rgb8> background,
                                         ImageView<uint8_t> mask,
                                         int width,
                                         int height,
                                         int blend_num,
                                         int blend_den)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    auto* curr_row = (rgb8*)((std::byte*)current.buffer + y * current.stride);
    auto* bg_row = (rgb8*)((std::byte*)background.buffer + y * background.stride);
    auto* mask_row = (uint8_t*)((std::byte*)mask.buffer + y * mask.stride);

    if (mask_row[x] != 0)
        return;

    rgb8& bg_pixel = bg_row[x];
    const rgb8 curr_pixel = curr_row[x];
    bg_pixel.r = (bg_pixel.r * (blend_den - blend_num) + curr_pixel.r * blend_num) / blend_den;
    bg_pixel.g = (bg_pixel.g * (blend_den - blend_num) + curr_pixel.g * blend_num) / blend_den;
    bg_pixel.b = (bg_pixel.b * (blend_den - blend_num) + curr_pixel.b * blend_num) / blend_den;
}

__global__ void diff_mask_kernel(ImageView<rgb8> curr,
                                 ImageView<rgb8> background,
                                 ImageView<uint8_t> input_mask,
                                 ImageView<uint8_t> marker_mask,
                                 int width,
                                 int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    auto* curr_row = (rgb8*)((std::byte*)curr.buffer + y * curr.stride);
    auto* bg_row = (rgb8*)((std::byte*)background.buffer + y * background.stride);
    auto* input_row = (uint8_t*)((std::byte*)input_mask.buffer + y * input_mask.stride);
    auto* marker_row = (uint8_t*)((std::byte*)marker_mask.buffer + y * marker_mask.stride);

    const rgb8 curr_pixel = curr_row[x];
    const rgb8 bg_pixel = bg_row[x];

    int diff_r = curr_pixel.r - bg_pixel.r;
    if (diff_r < 0)
        diff_r = -diff_r;
    int diff_g = curr_pixel.g - bg_pixel.g;
    if (diff_g < 0)
        diff_g = -diff_g;
    int diff_b = curr_pixel.b - bg_pixel.b;
    if (diff_b < 0)
        diff_b = -diff_b;
    int total_diff = diff_r + diff_g + diff_b;

    input_row[x] = total_diff > c_low_threshold ? 255 : 0;
    marker_row[x] = total_diff > c_high_threshold ? 255 : 0;
}

__global__ void diff_overlay_kernel(ImageView<rgb8> curr,
                                    ImageView<rgb8> background,
                                    ImageView<uint8_t> input_mask,
                                    ImageView<uint8_t> marker_mask,
                                    int width,
                                    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    auto* curr_row = (rgb8*)((std::byte*)curr.buffer + y * curr.stride);
    auto* bg_row = (rgb8*)((std::byte*)background.buffer + y * background.stride);
    auto* input_row = (uint8_t*)((std::byte*)input_mask.buffer + y * input_mask.stride);
    auto* marker_row = (uint8_t*)((std::byte*)marker_mask.buffer + y * marker_mask.stride);

    const rgb8 curr_pixel = curr_row[x];
    const rgb8 bg_pixel = bg_row[x];

    int diff_r = curr_pixel.r - bg_pixel.r;
    if (diff_r < 0)
        diff_r = -diff_r;
    int diff_g = curr_pixel.g - bg_pixel.g;
    if (diff_g < 0)
        diff_g = -diff_g;
    int diff_b = curr_pixel.b - bg_pixel.b;
    if (diff_b < 0)
        diff_b = -diff_b;
    int total_diff = diff_r + diff_g + diff_b;

    input_row[x] = total_diff > c_low_threshold ? 255 : 0;
    marker_row[x] = total_diff > c_high_threshold ? 255 : 0;

    if (marker_row[x] == 255)
    {
        curr_row[x].r = 255;
        curr_row[x].g = 0;
        curr_row[x].b = 0;
    }
}

__global__ void overlay_kernel(ImageView<rgb8> current,
                               ImageView<uint8_t> mask,
                               int width,
                               int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    auto* curr_row = (rgb8*)((std::byte*)current.buffer + y * current.stride);
    auto* mask_row = (uint8_t*)((std::byte*)mask.buffer + y * mask.stride);

    if (mask_row[x] == 255)
    {
        curr_row[x].r = 255;
        curr_row[x].g = 0;
        curr_row[x].b = 0;
    }
}

// ====================
// GPU WRAPPERS
// ====================
static void upload_current_frame(ProcessingState& state, uint8_t* buffer, int stride)
{
    cudaMemcpy2D(state.d_current.buffer,
                 state.d_current.stride,
                 buffer,
                 stride,
                 state.width * sizeof(rgb8),
                 state.height,
                 cudaMemcpyHostToDevice);
}

static void download_diff_masks_from_device(ProcessingState& state)
{
    const size_t total = total_pixels(state);
    cudaMemcpy(state.input_mask.data(), state.d_input_mask, total, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.marker_mask.data(), state.d_marker_mask, total, cudaMemcpyDeviceToHost);
}

static void upload_diff_masks_to_device(ProcessingState& state)
{
    const size_t total = total_pixels(state);
    cudaMemcpy(state.d_input_mask, state.input_mask.data(), total, cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_marker_mask, state.marker_mask.data(), total, cudaMemcpyHostToDevice);
}

static void download_motion_mask_from_device(ProcessingState& state)
{
    const size_t total = total_pixels(state);
    cudaMemcpy(state.motion_mask.data(), state.d_marker_mask, total, cudaMemcpyDeviceToHost);
    std::copy(state.motion_mask.begin(), state.motion_mask.end(), state.marker_mask.begin());
}

static void upload_motion_mask_to_device(ProcessingState& state)
{
    const size_t total = total_pixels(state);
    cudaMemcpy(state.d_marker_mask, state.motion_mask.data(), total, cudaMemcpyHostToDevice);
}

static void compute_diff_gpu(ProcessingState& state, uint8_t* buffer, int stride)
{
    upload_current_frame(state, buffer, stride);

    const dim3 block(16, 16);
    const dim3 grid((state.width + block.x - 1) / block.x,
                    (state.height + block.y - 1) / block.y);
    if (state.config.kernel_fusion && state.config.gpu_overlay)
    {
        diff_overlay_kernel<<<grid, block>>>(
            ImageView<rgb8>{state.d_current.buffer, state.width, state.height, state.d_current.stride},
            ImageView<rgb8>{state.d_background.buffer, state.width, state.height, state.d_background.stride},
            ImageView<uint8_t>{state.d_input_mask, state.width, state.height, state.width},
            ImageView<uint8_t>{state.d_marker_mask, state.width, state.height, state.width},
            state.width,
            state.height);
    }
    else
    {
        diff_mask_kernel<<<grid, block>>>(
            ImageView<rgb8>{state.d_current.buffer, state.width, state.height, state.d_current.stride},
            ImageView<rgb8>{state.d_background.buffer, state.width, state.height, state.d_background.stride},
            ImageView<uint8_t>{state.d_input_mask, state.width, state.height, state.width},
            ImageView<uint8_t>{state.d_marker_mask, state.width, state.height, state.width},
            state.width,
            state.height);
    }
}

static void hysteresis_gpu(ProcessingState& state)
{
    const int width = state.width;
    const int height = state.height;
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    uint32_t host_changed = 0;
    const int max_iter = 10;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        host_changed = 0;
        cudaMemset(state.d_changed, 0, sizeof(uint32_t));
        hysteresis_propagate<<<grid, block>>>(state.d_marker_mask,
                                               state.d_temp_mask,
                                               state.d_input_mask,
                                               width,
                                               height,
                                               state.d_changed);
        cudaMemcpy(&host_changed, state.d_changed, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(state.d_marker_mask, state.d_temp_mask, width * height, cudaMemcpyDeviceToDevice);
        if (host_changed == 0)
            break;
    }
    download_motion_mask_from_device(state);
}

static void morphology_gpu(ProcessingState& state)
{
    const int width = state.width;
    const int height = state.height;
    const size_t total = total_pixels(state);

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    const int opening = std::max(1, g_params.opening_size);
    const int radius = opening / 2;
    erosion_kernel<<<grid, block>>>(state.d_marker_mask, state.d_temp_mask, width, height, radius);
    dilation_kernel<<<grid, block>>>(state.d_temp_mask, state.d_marker_mask, width, height, radius);

    cudaMemcpy(state.motion_mask.data(), state.d_marker_mask, total, cudaMemcpyDeviceToHost);
    sync_motion_to_marker(state);
}

static void update_background_gpu(ProcessingState& state)
{
    const dim3 block(16, 16);
    const dim3 grid((state.width + block.x - 1) / block.x,
                    (state.height + block.y - 1) / block.y);
    const int blend_den = std::max(1, g_params.bg_number_frame);
    update_background_kernel<<<grid, block>>>(
        ImageView<rgb8>{state.d_current.buffer, state.width, state.height, state.d_current.stride},
        ImageView<rgb8>{state.d_background.buffer, state.width, state.height, state.d_background.stride},
        ImageView<uint8_t>{state.d_marker_mask, state.width, state.height, state.width},
        state.width,
        state.height,
        1,
        blend_den);

    cudaMemcpy2D(state.background.data(),
                 state.width * sizeof(rgb8),
                 state.d_background.buffer,
                 state.d_background.stride,
                 state.width * sizeof(rgb8),
                 state.height,
                 cudaMemcpyDeviceToHost);
}

static void overlay_gpu(ProcessingState& state)
{
    const dim3 block(16, 16);
    const dim3 grid((state.width + block.x - 1) / block.x,
                    (state.height + block.y - 1) / block.y);
    overlay_kernel<<<grid, block>>>(
        ImageView<rgb8>{state.d_current.buffer, state.width, state.height, state.d_current.stride},
        ImageView<uint8_t>{state.d_marker_mask, state.width, state.height, state.width},
        state.width,
        state.height);
}

// ====================
// PIPELINE
// ====================
static void process_frame(ProcessingState& state, uint8_t* buffer, int width, int height, int stride)
{
    if (!state.initialized || state.width != width || state.height != height)
    {
        init_state(state, buffer, width, height, stride);
        return;
    }

    const auto& cfg = state.config;
    const bool run_gpu_diff = cfg.use_gpu && cfg.gpu_diff;
    const bool run_gpu_hyst = cfg.use_gpu && cfg.gpu_hysteresis;
    const bool run_gpu_morph = cfg.use_gpu && cfg.gpu_morphology;
    const bool run_gpu_bg = cfg.use_gpu && cfg.gpu_background_update;
    const bool run_gpu_overlay = cfg.use_gpu && cfg.gpu_overlay;

    auto start = Clock::now();
    if (run_gpu_diff)
    {
        compute_diff_gpu(state, buffer, stride);
    }
    else
    {
        compute_diff_cpu(state, buffer, stride);
    }
    state.timing.diff_ms = DurationMs(Clock::now() - start).count();

    if (run_gpu_diff && !run_gpu_hyst)
        download_diff_masks_from_device(state);
    if (!run_gpu_diff && run_gpu_hyst)
        upload_diff_masks_to_device(state);

    start = Clock::now();
    if (run_gpu_hyst)
    {
        if (!run_gpu_diff)
            upload_diff_masks_to_device(state);
        hysteresis_gpu(state);
    }
    else
    {
        hysteresis_cpu(state);
    }
    state.timing.hyst_ms = DurationMs(Clock::now() - start).count();

    if (run_gpu_hyst && !run_gpu_morph)
        download_motion_mask_from_device(state);
    if (!run_gpu_hyst && run_gpu_morph)
        upload_motion_mask_to_device(state);

    start = Clock::now();
    if (run_gpu_morph)
    {
        if (!run_gpu_hyst)
            upload_motion_mask_to_device(state);
        morphology_gpu(state);
    }
    else
    {
        morphology_cpu(state);
    }
    state.timing.morph_ms = DurationMs(Clock::now() - start).count();

    if (!run_gpu_morph)
        sync_motion_to_marker(state);

    const bool do_bg_update = should_update_background(state);
    start = Clock::now();
    if (do_bg_update)
    {
        if (run_gpu_bg)
        {
            upload_current_frame(state, buffer, stride);
            if (!run_gpu_morph)
                upload_motion_mask_to_device(state);
            update_background_gpu(state);
        }
        else
        {
            update_background_cpu(state, buffer, stride);
        }
    }
    state.timing.update_ms = DurationMs(Clock::now() - start).count();

    if (run_gpu_overlay)
    {
        upload_current_frame(state, buffer, stride);
        if (!run_gpu_morph)
            upload_motion_mask_to_device(state);
        overlay_gpu(state);
        cudaMemcpy2D(buffer,
                     stride,
                     state.d_current.buffer,
                     state.d_current.stride,
                     width * sizeof(rgb8),
                     height,
                     cudaMemcpyDeviceToHost);
    }
    else
    {
        overlay_cpu(state, buffer, stride);
    }
    state.frame_counter++;
}

extern "C" void cpt_init(Parameters* params)
{
    g_params = *params;
    reset_device_buffers(g_state);
    g_state = ProcessingState();
    g_state.config.use_gpu = (params->device == e_device_t::GPU);
    g_state.config.gpu_diff = params->opt_gpu_diff && g_state.config.use_gpu;
    g_state.config.gpu_hysteresis = params->opt_gpu_hysteresis && g_state.config.use_gpu;
    g_state.config.gpu_morphology = params->opt_gpu_morphology && g_state.config.use_gpu;
    g_state.config.gpu_background_update = params->opt_gpu_background && g_state.config.use_gpu;
    g_state.config.gpu_overlay = params->opt_gpu_overlay && g_state.config.use_gpu;
    g_state.config.kernel_fusion = params->opt_kernel_fusion && g_state.config.use_gpu;
    const uint8_t low = static_cast<uint8_t>(std::clamp(g_params.th_low, 0, 255));
    const uint8_t high = static_cast<uint8_t>(std::clamp(g_params.th_high, 0, 255));
    cudaMemcpyToSymbol(c_low_threshold, &low, sizeof(low));
    cudaMemcpyToSymbol(c_high_threshold, &high, sizeof(high));
}

extern "C" void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
{
    process_frame(g_state, buffer, width, height, stride);
}
