#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"

#include <chrono>
#include <thread>


/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);


/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
  for (int y = 0; y < in.height; ++y)
  {
    rgb8* lineptr = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    for (int x = 0; x < in.width; ++x)
    {
      lineptr[x].r = 0; // Back out red component

      if (x < logo_width && y < logo_height)
      {
        float alpha  = logo_data[y * logo_width + x] / 255.f;
        lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1 - alpha) * 255);
        lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1 - alpha) * 255);
      }
    }
  }

}
