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
set_target_properties(stb_image PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(Compute src/Compute.cpp src/Compute.cu src/logo.c)
set_target_properties(Compute PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Compute PRIVATE CUDA::cudart stb_image)

add_library(stream_engine SHARED
  src/StreamRunner.cpp
  src/gstfilter.c
  src/gstfilter.h)
target_link_libraries(stream_engine PRIVATE Compute PkgConfig::GStreamerVideo CUDA::cudart)
set_target_properties(stream_engine PROPERTIES
  OUTPUT_NAME "_stream_engine"
  PREFIX ""
  SUFFIX ".so")
add_custom_command(TARGET stream_engine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:stream_engine>
    "${CMAKE_SOURCE_DIR}/../python-backend/_stream_engine.so")

# Create lightweight executable that calls into stream_engine
add_executable(stream src/stream.cpp)
target_link_libraries(stream PUBLIC stream_engine)
target_link_libraries(stream PUBLIC PkgConfig::GStreamerVideo)

## La beauté de NIX....
target_link_libraries(stream PUBLIC CUDA::cudart)
target_link_directories(stream PUBLIC "${CUDAToolkit_LIBRARY_DIR}")
set_target_properties(stream PROPERTIES
  INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)
