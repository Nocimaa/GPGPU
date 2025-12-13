#include "StreamEngine.hpp"

#include "gstfilter.h"

#include <gst/gst.h>

namespace
{

constexpr auto kPipelineDisplay = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter ! videoconvert ! fpsdisplaysink sync=false";
constexpr auto kPipelineOutput = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink name=fdst";

static gboolean plugin_init(GstPlugin* plugin)
{
  return gst_element_register(plugin, "myfilter", GST_RANK_NONE, GST_TYPE_MYFILTER);
}

static void my_code_init()
{
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

void ensure_gst_initialized()
{
  static bool initialized = false;
  if (!initialized)
  {
    gst_init(nullptr, nullptr);
    initialized = true;
  }
}

void ensure_plugin_registered()
{
  static bool registered = false;
  if (!registered)
  {
    my_code_init();
    registered = true;
  }
}

bool parse_mode(const std::string& mode, e_device_t* device)
{
  if (mode == "cpu")
  {
    *device = e_device_t::CPU;
    return true;
  }
  if (mode == "gpu")
  {
    *device = e_device_t::GPU;
    return true;
  }
  return false;
}

} // namespace

int run_stream(const StreamOptions& options)
{
  if (options.video_path.empty())
  {
    g_printerr("Video path is required\n");
    return 1;
  }

  ensure_gst_initialized();
  ensure_plugin_registered();

  Parameters params = options.params;
  cpt_init(&params);

  g_print("Output: %s\n", options.output_path.c_str());
  const char* pipeline_description = options.output_path.empty() ? kPipelineDisplay : kPipelineOutput;

  GError* error = nullptr;
  auto* pipeline = gst_parse_launch(pipeline_description, &error);
  if (!pipeline)
  {
    g_printerr("Failed to create pipeline: %s\n", (error && error->message) ? error->message : "unknown error");
    if (error)
      g_error_free(error);
    return 1;
  }

  auto* filesrc = gst_bin_get_by_name(GST_BIN(pipeline), "fsrc");
  g_object_set(filesrc, "location", options.video_path.c_str(), nullptr);
  g_object_unref(filesrc);

  if (!options.output_path.empty())
  {
    auto* filesink = gst_bin_get_by_name(GST_BIN(pipeline), "fdst");
    g_object_set(filesink, "location", options.output_path.c_str(), nullptr);
    g_object_unref(filesink);
  }

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  auto* bus = gst_element_get_bus(pipeline);
  auto* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
      static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  int result = 0;
  if (msg != nullptr)
  {
    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR)
    {
      GError* err = nullptr;
      gchar* debug = nullptr;
      gst_message_parse_error(msg, &err, &debug);
      g_printerr("Pipeline reported error: %s\n", err ? err->message : "unknown");
      if (err)
        g_error_free(err);
      if (debug)
        g_free(debug);
      result = 1;
    }
    gst_message_unref(msg);
  }
  else
  {
    result = 1;
  }

  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  return result;
}

int run_stream_c(
    const char* mode,
    const char* filename,
    const char* output,
    const char* background,
    int opening_size,
    int th_low,
    int th_high,
    int bg_sampling_rate,
    int bg_number_frame)
{
  StreamOptions opts;
  const std::string mode_value = mode ? mode : "";
  if (!parse_mode(mode_value, &opts.params.device))
  {
    g_printerr("Invalid method: %s\n", mode_value.c_str());
    return 1;
  }

  if (!filename || filename[0] == '\0')
  {
    g_printerr("Video filename is required\n");
    return 1;
  }

  opts.video_path = filename;
  if (output)
    opts.output_path = output;
  if (background)
    opts.background_path = background;

  opts.opening_size = opening_size;
  opts.th_low = th_low;
  opts.th_high = th_high;
  opts.bg_sampling_rate = bg_sampling_rate;
  opts.bg_number_frame = bg_number_frame;

  return run_stream(opts);
}
