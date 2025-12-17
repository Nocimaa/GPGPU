#include "Compute.hpp"
#include "Image.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
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
    bool kernel_fusion = false; // kept for compatibility (used to fuse diff+overlay)
    bool opt_cpu_simd = false;
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
    std::vector<uint8_t> background_r;
    std::vector<uint8_t> background_g;
    std::vector<uint8_t> background_b;
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

static void sync_background_planes(ProcessingState& state)
{
    const size_t total = total_pixels(state);
    state.background_r.resize(total);
    state.background_g.resize(total);
    state.background_b.resize(total);

    for (size_t i = 0; i < total; ++i)
    {
        const rgb8& bg = state.background[i];
        state.background_r[i] = bg.r;
        state.background_g[i] = bg.g;
        state.background_b[i] = bg.b;
    }
}

static inline __m128i abs_diff_epi16(const __m128i& a, const __m128i& b)
{
    const __m128i diff = _mm_sub_epi16(a, b);
    const __m128i neg = _mm_sub_epi16(b, a);
    const __m128i mask = _mm_cmpgt_epi16(diff, _mm_setzero_si128());
    return _mm_or_si128(_mm_and_si128(mask, diff),
                        _mm_andnot_si128(mask, neg));
}

// ====================
// SHARED (CPU/GPU) PRIMITIVES
// ====================

__host__ __device__ inline int iabs_int(int v) { return v < 0 ? -v : v; }

__host__ __device__ inline void compute_diff_pixel(const rgb8& curr, const rgb8& bg, uint8_t& low_mask, uint8_t& high_mask, uint8_t low_th, uint8_t high_th)
{
    const int dr = iabs_int(int(curr.r) - int(bg.r));
    const int dg = iabs_int(int(curr.g) - int(bg.g));
    const int db = iabs_int(int(curr.b) - int(bg.b));
    const int total = dr + dg + db;

    low_mask  = (total > int(low_th))  ? 255 : 0;
    high_mask = (total > int(high_th)) ? 255 : 0;
}

__host__ __device__
inline void overlay_pixel(rgb8& p, uint8_t mask)
{
    if (mask == 255)
    {
        p.r = 255;
        p.g = 0;
        p.b = 0;
    }
}

__host__ __device__ inline bool hysteresis_activate(const uint8_t* motion, const uint8_t* low, int x, int y, int w, int h)
{
    const int idx = y * w + x;
    if (low[idx] == 0) return false;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            if (motion[ny * w + nx] == 255) return true;
        }
    }
    return false;
}

__host__ __device__ inline bool erosion_test(const uint8_t* src, int x, int y, int w, int h, int radius)
{
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) return false;
            if (src[ny * w + nx] != 255) return false;
        }
    }
    return true;
}

__host__ __device__ inline bool dilation_test(const uint8_t* src, int x, int y, int w, int h, int radius)
{
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            if (src[ny * w + nx] == 255) return true;
        }
    }
    return false;
}

// ====================
// DEVICE BUFFER MGMT
// ====================
static void reset_device_buffers(ProcessingState& state)
{
    if (!state.config.use_gpu)
        return;

    if (state.d_input_mask)   { cudaFree(state.d_input_mask);   state.d_input_mask = nullptr; }
    if (state.d_marker_mask)  { cudaFree(state.d_marker_mask);  state.d_marker_mask = nullptr; }
    if (state.d_temp_mask)    { cudaFree(state.d_temp_mask);    state.d_temp_mask = nullptr; }
    if (state.d_changed)      { cudaFree(state.d_changed);      state.d_changed = nullptr; }
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

    // init background from first frame
    for (int y = 0; y < height; ++y)
    {
        auto* dst = state.background.data() + y * width;
        auto* src = reinterpret_cast<rgb8*>((std::byte*)buffer + y * stride);
        std::memcpy(dst, src, width * sizeof(rgb8));
    }

    sync_background_planes(state);

    if (state.config.use_gpu)
    {
        state.d_background = Image<rgb8>(width, height, true);
        state.d_current    = Image<rgb8>(width, height, true);

        reset_device_buffers(state);
        cudaMalloc(&state.d_input_mask,  width * height);
        cudaMalloc(&state.d_marker_mask, width * height);
        cudaMalloc(&state.d_temp_mask,   width * height);
        cudaMalloc(&state.d_changed,     sizeof(uint32_t));

        cudaMemcpy2D(state.d_background.buffer,
                     state.d_background.stride,
                     state.background.data(),
                     width * sizeof(rgb8),
                     width * sizeof(rgb8),
                     height,
                     cudaMemcpyHostToDevice);
    }

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

// Return true only when the current build actually supports SSE2, so we can
// safely fall back on platforms (e.g. ARM) where SIMD intrinsics are missing.
static bool cpu_simd_available()
{
#if defined(__SSE2__)
    return true;
#else
    return false;
#endif
}

static void compute_diff_cpu_simd(ProcessingState& state, uint8_t* buffer, int stride)
{
    const int w = state.width;
    const int h = state.height;

    const uint8_t low_th  = static_cast<uint8_t>(std::clamp(g_params.th_low,  0, 255));
    const uint8_t high_th = static_cast<uint8_t>(std::clamp(g_params.th_high, 0, 255));

    const auto* bg_r = state.background_r.data();
    const auto* bg_g = state.background_g.data();
    const auto* bg_b = state.background_b.data();
    auto* low_mask   = state.input_mask.data();
    auto* high_mask  = state.marker_mask.data();

    constexpr int kSimdChunk = 8;
    alignas(16) uint8_t curr_r[kSimdChunk];
    alignas(16) uint8_t curr_g[kSimdChunk];
    alignas(16) uint8_t curr_b[kSimdChunk];

    const __m128i low_th_vec  = _mm_set1_epi16(static_cast<int16_t>(low_th));
    const __m128i high_th_vec = _mm_set1_epi16(static_cast<int16_t>(high_th));
    const __m128i zero        = _mm_setzero_si128();

    for (int y = 0; y < h; ++y)
    {
        const auto* curr_row =
            reinterpret_cast<const rgb8*>((std::byte*)buffer + y * stride);

        int x = 0;
        const int row_base = y * w;

        for (; x + kSimdChunk <= w; x += kSimdChunk)
        {
            for (int i = 0; i < kSimdChunk; ++i)
            {
                const rgb8& p = curr_row[x + i];
                curr_r[i] = p.r;
                curr_g[i] = p.g;
                curr_b[i] = p.b;
            }

            const int index = row_base + x;

            const __m128i curr_r_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(curr_r));
            const __m128i curr_g_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(curr_g));
            const __m128i curr_b_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(curr_b));

            const __m128i bg_r_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bg_r + index));
            const __m128i bg_g_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bg_g + index));
            const __m128i bg_b_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bg_b + index));

            const __m128i curr_r_lo = _mm_unpacklo_epi8(curr_r_vec, zero);
            const __m128i curr_g_lo = _mm_unpacklo_epi8(curr_g_vec, zero);
            const __m128i curr_b_lo = _mm_unpacklo_epi8(curr_b_vec, zero);

            const __m128i bg_r_lo = _mm_unpacklo_epi8(bg_r_vec, zero);
            const __m128i bg_g_lo = _mm_unpacklo_epi8(bg_g_vec, zero);
            const __m128i bg_b_lo = _mm_unpacklo_epi8(bg_b_vec, zero);

            const __m128i dr = abs_diff_epi16(curr_r_lo, bg_r_lo);
            const __m128i dg = abs_diff_epi16(curr_g_lo, bg_g_lo);
            const __m128i db = abs_diff_epi16(curr_b_lo, bg_b_lo);

            const __m128i sum_rg  = _mm_add_epi16(dr, dg);
            const __m128i sum_rgb = _mm_add_epi16(sum_rg, db);

            const __m128i low_cmp  = _mm_cmpgt_epi16(sum_rgb, low_th_vec);
            const __m128i high_cmp = _mm_cmpgt_epi16(sum_rgb, high_th_vec);

            // FIX: use signed packing so 0xFFFF -> 255
            const __m128i low_packed  = _mm_packs_epi16(low_cmp,  zero);
            const __m128i high_packed = _mm_packs_epi16(high_cmp, zero);

            _mm_storel_epi64(reinterpret_cast<__m128i*>(low_mask  + index), low_packed);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(high_mask + index), high_packed);
        }

        for (; x < w; ++x)
        {
            const int i = row_base + x;
            compute_diff_pixel(curr_row[x],
                               state.background[i],
                               low_mask[i],
                               high_mask[i],
                               low_th,
                               high_th);
        }
    }
}


static void compute_diff_cpu(ProcessingState& state, uint8_t* buffer, int stride)
{
    const int w = state.width;
    const int h = state.height;

    const uint8_t low_th  = static_cast<uint8_t>(std::clamp(g_params.th_low, 0, 255));
    const uint8_t high_th = static_cast<uint8_t>(std::clamp(g_params.th_high, 0, 255));

    for (int y = 0; y < h; ++y)
    {
        auto* curr_row = reinterpret_cast<rgb8*>((std::byte*)buffer + y * stride);
        for (int x = 0; x < w; ++x)
        {
            const int i = y * w + x;
            compute_diff_pixel(curr_row[x],
                               state.background[i],
                               state.input_mask[i],
                               state.marker_mask[i],
                               low_th,
                               high_th);
        }
    }
}

static void hysteresis_cpu(ProcessingState& state)
{
    // start from marker_mask
    std::copy(state.marker_mask.begin(), state.marker_mask.end(), state.motion_mask.begin());

    const int w = state.width;
    const int h = state.height;

    bool changed;
    do
    {
        changed = false;
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const int i = y * w + x;
                if (state.motion_mask[i] != 0) continue;   // already active
                if (state.input_mask[i] == 0) continue;    // not even in low mask

                if (hysteresis_activate(state.motion_mask.data(), state.input_mask.data(), x, y, w, h))
                {
                    state.motion_mask[i] = 255;
                    changed = true;
                }
            }
        }
    } while (changed);

    sync_motion_to_marker(state);
}

static void morphology_cpu(ProcessingState& state)
{
    const int w = state.width;
    const int h = state.height;

    const int opening = std::max(1, g_params.opening_size);
    const int radius  = opening / 2;

    const uint8_t* src = state.motion_mask.data();
    uint8_t* tmp = state.temp_mask.data();

    // Erosion
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            tmp[y * w + x] = erosion_test(src, x, y, w, h, radius) ? 255 : 0;

    // Dilation
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            state.motion_mask[y * w + x] = dilation_test(tmp, x, y, w, h, radius) ? 255 : 0;

    sync_motion_to_marker(state);
}

static void update_background_cpu(ProcessingState& state, uint8_t* buffer, int stride)
{
    const int w = state.width;
    const int h = state.height;
    const int den = std::max(1, g_params.bg_number_frame);

    for (int y = 0; y < h; ++y)
    {
        auto* curr_row = reinterpret_cast<rgb8*>((std::byte*)buffer + y * stride);
        for (int x = 0; x < w; ++x)
        {
            const int i = y * w + x;
            if (state.motion_mask[i] != 0) continue;

            rgb8& bg = state.background[i];
            const rgb8 curr = curr_row[x];

            bg.r = (bg.r * (den - 1) + curr.r) / den;
            bg.g = (bg.g * (den - 1) + curr.g) / den;
            bg.b = (bg.b * (den - 1) + curr.b) / den;

            state.background_r[i] = bg.r;
            state.background_g[i] = bg.g;
            state.background_b[i] = bg.b;
        }
    }
}

static void overlay_cpu(ProcessingState& state, uint8_t* buffer, int stride)
{
    const int w = state.width;
    const int h = state.height;

    for (int y = 0; y < h; ++y)
    {
        auto* curr_row = reinterpret_cast<rgb8*>((std::byte*)buffer + y * stride);
        for (int x = 0; x < w; ++x)
        {
            overlay_pixel(curr_row[x], state.motion_mask[y * w + x]);
        }
    }
}

// ====================
// CUDA KERNELS
// ====================

__global__ void diff_kernel(ImageView<rgb8> curr, ImageView<rgb8> bg, ImageView<uint8_t> low_mask, ImageView<uint8_t> high_mask, bool do_overlay, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    auto* curr_row  = (rgb8*)((std::byte*)curr.buffer + y * curr.stride);
    auto* bg_row    = (rgb8*)((std::byte*)bg.buffer + y * bg.stride);
    auto* low_row   = (uint8_t*)((std::byte*)low_mask.buffer + y * low_mask.stride);
    auto* high_row  = (uint8_t*)((std::byte*)high_mask.buffer + y * high_mask.stride);

    uint8_t l = 0, hmask = 0;
    compute_diff_pixel(curr_row[x], bg_row[x], l, hmask, c_low_threshold, c_high_threshold);

    low_row[x]  = l;
    high_row[x] = hmask;

    if (do_overlay && hmask == 255)
    {
        overlay_pixel(curr_row[x], 255);
    }
}

__global__ void hysteresis_propagate_kernel(const uint8_t* motion_in, uint8_t* motion_out, const uint8_t* low_mask, int width, int height, uint32_t* changed)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    if (motion_in[idx] == 255)
    {
        motion_out[idx] = 255;
        return;
    }
    if (low_mask[idx] == 0)
    {
        motion_out[idx] = 0;
        return;
    }

    if (hysteresis_activate(motion_in, low_mask, x, y, width, height))
    {
        motion_out[idx] = 255;
        atomicExch(changed, 1u);
    }
    else
    {
        motion_out[idx] = 0;
    }
}

__global__ void erosion_kernel(const uint8_t* src, uint8_t* dst, int w, int h, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dst[y * w + x] = erosion_test(src, x, y, w, h, radius) ? 255 : 0;
}

__global__ void dilation_kernel(const uint8_t* src, uint8_t* dst, int w, int h, int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dst[y * w + x] = dilation_test(src, x, y, w, h, radius) ? 255 : 0;
}

__global__ void update_background_kernel(ImageView<rgb8> current, ImageView<rgb8> background, ImageView<uint8_t> motion_mask, int width, int height, int blend_num, int blend_den)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    auto* curr_row = (rgb8*)((std::byte*)current.buffer + y * current.stride);
    auto* bg_row   = (rgb8*)((std::byte*)background.buffer + y * background.stride);
    auto* m_row    = (uint8_t*)((std::byte*)motion_mask.buffer + y * motion_mask.stride);

    if (m_row[x] != 0) return;

    rgb8& bgp = bg_row[x];
    const rgb8 cp = curr_row[x];

    bgp.r = (bgp.r * (blend_den - blend_num) + cp.r * blend_num) / blend_den;
    bgp.g = (bgp.g * (blend_den - blend_num) + cp.g * blend_num) / blend_den;
    bgp.b = (bgp.b * (blend_den - blend_num) + cp.b * blend_num) / blend_den;
}

__global__ void overlay_kernel(ImageView<rgb8> current, ImageView<uint8_t> motion_mask, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    auto* curr_row = (rgb8*)((std::byte*)current.buffer + y * current.stride);
    auto* m_row    = (uint8_t*)((std::byte*)motion_mask.buffer + y * motion_mask.stride);

    overlay_pixel(curr_row[x], m_row[x]);
}

// ====================
// GPU HELPERS
// ====================
static dim3 default_block() { return dim3(16, 16); }

static dim3 default_grid(int w, int h, dim3 block)
{
    return dim3((w + int(block.x) - 1) / int(block.x),
                (h + int(block.y) - 1) / int(block.y));
}

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
    cudaMemcpy(state.input_mask.data(),  state.d_input_mask,  total, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.marker_mask.data(), state.d_marker_mask, total, cudaMemcpyDeviceToHost);
}

static void upload_diff_masks_to_device(ProcessingState& state)
{
    const size_t total = total_pixels(state);
    cudaMemcpy(state.d_input_mask,  state.input_mask.data(),  total, cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_marker_mask, state.marker_mask.data(), total, cudaMemcpyHostToDevice);
}

static void download_motion_mask_from_device(ProcessingState& state)
{
    const size_t total = total_pixels(state);
    cudaMemcpy(state.motion_mask.data(), state.d_marker_mask, total, cudaMemcpyDeviceToHost);
    sync_motion_to_marker(state);
}

static void upload_motion_mask_to_device(ProcessingState& state)
{
    const size_t total = total_pixels(state);
    cudaMemcpy(state.d_marker_mask, state.motion_mask.data(), total, cudaMemcpyHostToDevice);
}

// ====================
// GPU WRAPPERS
// ====================
static void compute_diff_gpu(ProcessingState& state, uint8_t* buffer, int stride, bool fuse_overlay)
{
    upload_current_frame(state, buffer, stride);

    const dim3 block = default_block();
    const dim3 grid  = default_grid(state.width, state.height, block);

    diff_kernel<<<grid, block>>>(
        ImageView<rgb8>{state.d_current.buffer, state.width, state.height, state.d_current.stride},
        ImageView<rgb8>{state.d_background.buffer, state.width, state.height, state.d_background.stride},
        ImageView<uint8_t>{state.d_input_mask, state.width, state.height, state.width},
        ImageView<uint8_t>{state.d_marker_mask, state.width, state.height, state.width},
        fuse_overlay,
        state.width,
        state.height);
}

static void hysteresis_gpu(ProcessingState& state)
{
    const int w = state.width;
    const int h = state.height;

    const dim3 block = default_block();
    const dim3 grid  = default_grid(w, h, block);

    const int max_iter = 10;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        cudaMemset(state.d_changed, 0, sizeof(uint32_t));

        hysteresis_propagate_kernel<<<grid, block>>>(
            state.d_marker_mask,
            state.d_temp_mask,
            state.d_input_mask,
            w, h,
            state.d_changed);

        uint32_t host_changed = 0;
        cudaMemcpy(&host_changed, state.d_changed, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(state.d_marker_mask, state.d_temp_mask, w * h, cudaMemcpyDeviceToDevice);

        if (host_changed == 0)
            break;
    }
}

static void morphology_gpu(ProcessingState& state)
{
    const int w = state.width;
    const int h = state.height;

    const dim3 block = default_block();
    const dim3 grid  = default_grid(w, h, block);

    const int opening = std::max(1, g_params.opening_size);
    const int radius  = opening / 2;

    erosion_kernel<<<grid, block>>>(state.d_marker_mask, state.d_temp_mask, w, h, radius);
    dilation_kernel<<<grid, block>>>(state.d_temp_mask, state.d_marker_mask, w, h, radius);

}

static void update_background_gpu(ProcessingState& state)
{
    const dim3 block = default_block();
    const dim3 grid  = default_grid(state.width, state.height, block);

    const int den = std::max(1, g_params.bg_number_frame);

    update_background_kernel<<<grid, block>>>(
        ImageView<rgb8>{state.d_current.buffer, state.width, state.height, state.d_current.stride},
        ImageView<rgb8>{state.d_background.buffer, state.width, state.height, state.d_background.stride},
        ImageView<uint8_t>{state.d_marker_mask, state.width, state.height, state.width},
        state.width,
        state.height,
        1,
        den);

    // keep host background in sync (like your original code)
    cudaMemcpy2D(state.background.data(),
                 state.width * sizeof(rgb8),
                 state.d_background.buffer,
                 state.d_background.stride,
                 state.width * sizeof(rgb8),
                 state.height,
                 cudaMemcpyDeviceToHost);

    sync_background_planes(state);
}

static void overlay_gpu(ProcessingState& state)
{
    const dim3 block = default_block();
    const dim3 grid  = default_grid(state.width, state.height, block);

    overlay_kernel<<<grid, block>>>(
        ImageView<rgb8>{state.d_current.buffer, state.width, state.height, state.d_current.stride},
        ImageView<uint8_t>{state.d_marker_mask, state.width, state.height, state.width},
        state.width,
        state.height);
}

template <typename CpuFn, typename GpuFn>
static void run_processing_stage(bool use_gpu, CpuFn&& cpu_fn, GpuFn&& gpu_fn, float& timing)
{
    const auto start = Clock::now();
    if (use_gpu)
        gpu_fn();
    else
        cpu_fn();
    timing = DurationMs(Clock::now() - start).count();
}

static void sync_diff_and_hysteresis_masks(ProcessingState& state, bool diff_gpu, bool hyst_gpu)
{
    if (diff_gpu && !hyst_gpu)
        download_diff_masks_from_device(state);
    else if (!diff_gpu && hyst_gpu)
        upload_diff_masks_to_device(state);
}

static void sync_hysteresis_and_morph_masks(ProcessingState& state, bool hyst_gpu, bool morph_gpu)
{
    if (hyst_gpu && !morph_gpu)
        download_motion_mask_from_device(state);
    else if (!hyst_gpu && morph_gpu)
        upload_motion_mask_to_device(state);
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

    const bool run_gpu_diff    = cfg.use_gpu && cfg.gpu_diff;
    const bool run_gpu_hyst    = cfg.use_gpu && cfg.gpu_hysteresis;
    const bool run_gpu_morph   = cfg.use_gpu && cfg.gpu_morphology;
    const bool run_gpu_bg      = cfg.use_gpu && cfg.gpu_background_update;
    const bool run_gpu_overlay = cfg.use_gpu && cfg.gpu_overlay;

    const bool fuse_overlay_in_diff = (run_gpu_diff && run_gpu_overlay && cfg.kernel_fusion);

    auto run_diff_cpu = [&]() {
        const bool do_simd = cfg.opt_cpu_simd && cpu_simd_available();
        if (do_simd)
            compute_diff_cpu_simd(state, buffer, stride);
        else
            compute_diff_cpu(state, buffer, stride);
    };
    auto run_diff_gpu = [&]() { compute_diff_gpu(state, buffer, stride, fuse_overlay_in_diff); };
    run_processing_stage(run_gpu_diff, run_diff_cpu, run_diff_gpu, state.timing.diff_ms);

    sync_diff_and_hysteresis_masks(state, run_gpu_diff, run_gpu_hyst);

    auto run_hyst_cpu = [&]() { hysteresis_cpu(state); };
    auto run_hyst_gpu = [&]() { hysteresis_gpu(state); };
    run_processing_stage(run_gpu_hyst, run_hyst_cpu, run_hyst_gpu, state.timing.hyst_ms);

    sync_hysteresis_and_morph_masks(state, run_gpu_hyst, run_gpu_morph);

    auto run_morph_cpu = [&]() { morphology_cpu(state); };
    auto run_morph_gpu = [&]() { morphology_gpu(state); };
    run_processing_stage(run_gpu_morph, run_morph_cpu, run_morph_gpu, state.timing.morph_ms);

    const bool do_bg_update = should_update_background(state);
    if (run_gpu_morph)
    {
        const bool needs_host_for_bg = do_bg_update && !run_gpu_bg;
        const bool needs_host_for_overlay = !run_gpu_overlay;
        if (needs_host_for_bg || needs_host_for_overlay)
            download_motion_mask_from_device(state);
    }

    if (do_bg_update)
    {
        auto run_bg_cpu = [&]() { update_background_cpu(state, buffer, stride); };
        auto run_bg_gpu = [&]() {
            if (!run_gpu_diff) upload_current_frame(state, buffer, stride);
            if (!run_gpu_morph) upload_motion_mask_to_device(state);
            update_background_gpu(state);
        };
        run_processing_stage(run_gpu_bg, run_bg_cpu, run_bg_gpu, state.timing.update_ms);
    }
    else
    {
        state.timing.update_ms = 0.f;
    }

    if (run_gpu_overlay)
    {
        if (!run_gpu_diff) upload_current_frame(state, buffer, stride);
        if (!run_gpu_morph) upload_motion_mask_to_device(state);

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

// ====================
// C API
// ====================
extern "C" void cpt_init(Parameters* params)
{
    g_params = *params;

    g_state = ProcessingState();

    g_state.config.use_gpu = (params->device == e_device_t::GPU);

    g_state.config.gpu_diff              = params->opt_gpu_diff       && g_state.config.use_gpu;
    g_state.config.gpu_hysteresis        = params->opt_gpu_hysteresis && g_state.config.use_gpu;
    g_state.config.gpu_morphology        = params->opt_gpu_morphology && g_state.config.use_gpu;
    g_state.config.gpu_background_update = params->opt_gpu_background && g_state.config.use_gpu;
    g_state.config.gpu_overlay           = params->opt_gpu_overlay    && g_state.config.use_gpu;
    g_state.config.kernel_fusion         = params->opt_kernel_fusion  && g_state.config.use_gpu;
    
    g_state.config.opt_cpu_simd          = params->opt_cpu_simd;

    if (g_state.config.use_gpu)
    {
        const uint8_t low  = static_cast<uint8_t>(std::clamp(g_params.th_low,  0, 255));
        const uint8_t high = static_cast<uint8_t>(std::clamp(g_params.th_high, 0, 255));

        cudaError_t err_l = cudaMemcpyToSymbol(c_low_threshold,  &low,  sizeof(low));
        cudaError_t err_h = cudaMemcpyToSymbol(c_high_threshold, &high, sizeof(high));

        if (err_l != cudaSuccess || err_h != cudaSuccess) {
            g_state.config.use_gpu = false;
        }
    }
}

extern "C" void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
{
    process_frame(g_state, buffer, width, height, stride);
}
