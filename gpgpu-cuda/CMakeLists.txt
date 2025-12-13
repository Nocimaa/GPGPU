cmake_minimum_required (VERSION 3.18)
project (PRPA)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)


pkg_check_modules(GStreamer REQUIRED gstreamer-1.0 IMPORTED_TARGET)
pkg_check_modules(GStreamerVideo REQUIRED gstreamer-video-1.0 IMPORTED_TARGET)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")


add_library(stb_image src/stb_image.h src/stb_image.c)
add_library(Compute src/Compute.cpp src/Compute.cu src/logo.c)
target_link_libraries(Compute PUBLIC CUDA::cudart)
set_target_properties(Compute PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(stream src/stream.cpp src/StreamEngine.cpp src/gstfilter.c src/gstfilter.h)


target_link_libraries(stream PUBLIC PkgConfig::GStreamerVideo Compute)

add_library(stream_engine SHARED src/StreamEngine.cpp src/gstfilter.c)
target_link_libraries(stream_engine PUBLIC PkgConfig::GStreamerVideo Compute)
target_link_directories(stream_engine PUBLIC "${CUDAToolkit_LIBRARY_DIR}")
set_target_properties(stream_engine PROPERTIES
  OUTPUT_NAME "_stream_engine"
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)

## La beauté de NIX....
target_link_directories(stream PUBLIC "${CUDAToolkit_LIBRARY_DIR}")
set_target_properties(stream PROPERTIES
  INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)
