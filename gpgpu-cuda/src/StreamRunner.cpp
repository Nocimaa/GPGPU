#include "StreamRunner.hpp"
#include "Compute.hpp"
#include "gstfilter.h"

#include <gst/gst.h>
#include <mutex>
#include <string>

static gboolean plugin_init(GstPlugin* plugin)
{
  return gst_element_register(plugin, "myfilter", GST_RANK_NONE, GST_TYPE_MYFILTER);
}

namespace
{

const char* const kPipelineCpu = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter ! videoconvert ! fpsdisplaysink sync=false";
const char* const kPipelineWithOutput = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink name=fdst";

void ensure_filter_registered()
{
  static std::once_flag once;
  std::call_once(once, []() {
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
  });
}

int set_pipeline_locations(GstElement* pipeline, const StreamOptions& options)
{
  auto filesrc = gst_bin_get_by_name(GST_BIN(pipeline), "fsrc");
  if (!filesrc)
    return 1;

  g_object_set(filesrc, "location", options.filename.c_str(), nullptr);
  g_object_unref(filesrc);

  if (options.output.empty())
    return 0;

  auto filesink = gst_bin_get_by_name(GST_BIN(pipeline), "fdst");
  if (!filesink)
    return 1;
  g_object_set(filesink, "location", options.output.c_str(), nullptr);
  g_object_unref(filesink);
  return 0;
}

} // namespace

int run_stream(const StreamOptions& options)
{
  if (options.filename.empty())
  {
    g_printerr("Missing filename\n");
    return 1;
  }

  const std::string mode = options.mode;
  Parameters params;
  params.device = (mode == "gpu") ? e_device_t::GPU : e_device_t::CPU;
  params.bg_uri = options.background.empty() ? nullptr : options.background.c_str();
  params.opening_size = options.opening_size;
  params.th_low = options.th_low;
  params.th_high = options.th_high;
  params.bg_sampling_rate_ms = options.bg_sampling_rate;
  params.bg_number_frame = options.bg_number_frame;

  gst_init(nullptr, nullptr);
  cpt_init(&params);
  ensure_filter_registered();

  const char* pipeline_desc = options.output.empty() ? kPipelineCpu : kPipelineWithOutput;
  GError* error = nullptr;
  auto pipeline = gst_parse_launch(pipeline_desc, &error);
  if (!pipeline)
  {
    if (error)
    {
      g_printerr("Failed to create pipeline: %s\n", error->message);
      g_error_free(error);
    }
    return 1;
  }

  if (set_pipeline_locations(pipeline, options) != 0)
  {
    gst_object_unref(pipeline);
    return 1;
  }

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  auto bus = gst_element_get_bus(pipeline);
  auto msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  if (msg)
    gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  return 0;
}

extern "C" int run_stream_c(const char* mode,
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
  opts.mode = mode ? std::string(mode) : "cpu";
  opts.filename = filename ? std::string(filename) : "";
  opts.output = output ? std::string(output) : "";
  opts.background = background ? std::string(background) : "";
  opts.opening_size = opening_size;
  opts.th_low = th_low;
  opts.th_high = th_high;
  opts.bg_sampling_rate = bg_sampling_rate;
  opts.bg_number_frame = bg_number_frame;
  return run_stream(opts);
}
