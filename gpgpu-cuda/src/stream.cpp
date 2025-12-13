// Create a GStreamer pipeline to stream a video through our plugin

#include <gst/gst.h>
#include <format>
#include <string>
#include "gstfilter.h"
#include "argh.h"


static gboolean
plugin_init (GstPlugin * plugin)
{

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register (plugin, "myfilter", GST_RANK_NONE,
      GST_TYPE_MYFILTER);
}

static void my_code_init ()
{
  gst_plugin_register_static (
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

  Parameters params;
  auto method = cmdl("mode", "cpu").str();
  if (method == "cpu")
    params.device = e_device_t::CPU;
  else if (method == "gpu")
    params.device = e_device_t::GPU;
  else
  {
    g_printerr("Invalid method: %s\n", method.c_str());
    return 1;
  }

  const std::string bg_uri = cmdl("bg", "").str();
  params.bg_uri = bg_uri.empty() ? nullptr : bg_uri.c_str();
  params.opening_size = parse_int("opening_size", 3);
  params.th_low = parse_int("th_low", 3);
  params.th_high = parse_int("th_high", 30);
  params.bg_sampling_rate_ms = parse_int("bg_sampling_rate", 500);
  params.bg_number_frame = parse_int("bg_number_frame", 10);

  const std::string output = cmdl({"-o", "--output"}, "").str();

  g_debug("Using method: %s", method.c_str());
  gst_init(&argc, &argv);
  my_code_init();
  cpt_init(&params);

  // Create a GStreamer pipeline to stream a video through our plugin
  const char* pipe_str;
  g_print("Output: %s\n", output.c_str());
  if (output.empty())
    pipe_str = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter ! videoconvert ! fpsdisplaysink sync=false";
  else
    pipe_str = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink name=fdst";



  GError *error = NULL;
  auto pipeline = gst_parse_launch(pipe_str, &error);
  if (!pipeline)
  {
    g_printerr("Failed to create pipeline: %s\n", error->message);
    return 1;
  }

  auto filesrc = gst_bin_get_by_name (GST_BIN (pipeline), "fsrc");
  g_object_set (filesrc, "location", filename.c_str(), NULL);
  g_object_unref (filesrc);

  if (!output.empty())
  {
    auto filesink = gst_bin_get_by_name (GST_BIN (pipeline), "fdst");
    g_object_set (filesink, "location", output.c_str(), NULL);
    g_object_unref (filesink);
  }

  // Start the pipeline
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  // Wait until error or EOS
  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  // Free resources
  if (msg != nullptr)
    gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  return 0;
}
