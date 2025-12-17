#include "StreamEngine.h"

#include <gst/gst.h>
#include <gst/video/video.h>
#include "gstfilter.h"

#include <string>

namespace
{
static gboolean plugin_init(GstPlugin* plugin)
{
    return gst_element_register(plugin, "myfilter", GST_RANK_NONE, GST_TYPE_MYFILTER);
}

static void register_myfilter()
{
    static bool registered = false;
    if (registered)
        return;
    registered = true;

    gst_plugin_register_static(
        GST_VERSION_MAJOR,
        GST_VERSION_MINOR,
        "myfilter",
        "Private elements of my application",
        plugin_init,
        "1.0",
        "LGPL",
        "",
        "",
        "");
}

static std::string current_pipeline_string(bool has_output)
{
    if (has_output)
    {
        return "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink name=fdst";
    }
    return "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter ! videoconvert ! fpsdisplaysink sync=false";
}

static int run_pipeline(const std::string& pipeline_desc,
                        const std::string& filename,
                        const std::string& output)
{
    GError* error = nullptr;
    auto* pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
    if (!pipeline)
    {
        if (error)
        {
            g_printerr("Failed to create pipeline: %s\n", error->message);
            g_error_free(error);
        }
        return 1;
    }

    auto* filesrc = gst_bin_get_by_name(GST_BIN(pipeline), "fsrc");
    g_object_set(filesrc, "location", filename.c_str(), nullptr);
    g_object_unref(filesrc);

    if (!output.empty())
    {
        auto* filesink = gst_bin_get_by_name(GST_BIN(pipeline), "fdst");
        g_object_set(filesink, "location", output.c_str(), nullptr);
        g_object_unref(filesink);
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    auto* bus = gst_element_get_bus(pipeline);
    auto* msg = gst_bus_timed_pop_filtered(bus,
                                           GST_CLOCK_TIME_NONE,
                                           static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    int result = 0;
    if (msg == nullptr)
        result = 1;
    else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR)
        result = 1;

    if (msg != nullptr)
        gst_message_unref(msg);
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return result;
}
}  // namespace

int run_stream(const Parameters& params,
               const std::string& mode,
               const std::string& filename,
               const std::string& output,
               const std::string& background)
{
    if (filename.empty())
        return 1;

    // Avoid the GStreamer GL/KMS/VAAPI plugins probing /dev/dri endlessly on headless servers.
    // We only need CPU ↔ CUDA, so force the registry to ignore those sinks.
    g_setenv("GST_GL_API", "none", TRUE);
    g_setenv("GST_PLUGIN_FEATURE_RANK", "gl*:0,kms*:0,vaapi*:0", FALSE);

    if (!gst_is_initialized())
        gst_init(nullptr, nullptr);

    register_myfilter();

    Parameters working_params = params;
    cpt_init(&working_params);

    (void)background;

    g_printerr("Using method: %s\n", mode.c_str());
    g_print("Output: %s\n", output.c_str());

    const bool has_output = !output.empty();
    const auto pipeline_desc = current_pipeline_string(has_output);
    return run_pipeline(pipeline_desc, filename, output);
}

extern "C" int run_stream_c(const char* mode,
                            const char* filename,
                            const char* output,
                            const char* background,
                            int opening_size,
                            int th_low,
                            int th_high,
                            int bg_sampling_rate,
                            int bg_number_frame,
                            int cpu_simd)
{
    if (mode == nullptr || filename == nullptr)
        return 1;

    const std::string mode_str(mode);
    Parameters params = {};
    if (mode_str == "gpu")
        params.device = e_device_t::GPU;
    else if (mode_str == "cpu")
        params.device = e_device_t::CPU;
    else
    {
        g_printerr("Invalid mode: %s\n", mode);
        return 1;
    }

    const bool use_gpu = params.device == e_device_t::GPU;
    params.opt_gpu_diff = use_gpu;
    params.opt_gpu_hysteresis = use_gpu;
    params.opt_gpu_morphology = use_gpu;
    params.opt_gpu_background = use_gpu;
    params.opt_gpu_overlay = use_gpu;
    params.opt_kernel_fusion = false;
    params.opt_cpu_simd = cpu_simd != 0;
    params.opening_size = opening_size;
    params.th_low = th_low;
    params.th_high = th_high;
    params.bg_sampling_rate = bg_sampling_rate;
    params.bg_number_frame = bg_number_frame;

    return run_stream(params,
                      mode_str,
                      std::string(filename),
                      output ? std::string(output) : std::string(),
                      background ? std::string(background) : std::string());
}
