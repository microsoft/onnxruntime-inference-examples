# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
string(APPEND CMAKE_CXX_FLAGS " /W4")

#onnxruntime-openvino provider
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(OPENCV_ROOTDIR "OpenCV root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
option(OPENCL_LIB "OpenCL lib dir")
option(OPENCL_INCLUDE "OpenCL header dir")

if(NOT ONNXRUNTIME_ROOTDIR)
  set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
endif()
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if(OPENCV_ROOTDIR)
  set(OPENCV_FOUND true)
  set(OPENCV_INCLUDE_DIRS "${OPENCV_ROOTDIR}/include")
  set(OPENCV_LIBDIR "${OPENCV_ROOTDIR}/x64/vc16/lib")
  file(GLOB OPENCV_DEBUG_LIBRARIES "${OPENCV_LIBDIR}/opencv_world*d.lib")
  file(GLOB OPENCV_RELEASE_LIBRARIES "${OPENCV_LIBDIR}/opencv_world*.lib")
  list(FILTER OPENCV_RELEASE_LIBRARIES EXCLUDE REGEX ".*d\\.lib")
endif()

if(OPENCL_LIB AND OPENCL_INCLUDE)
  set(OPENCL_FOUND true)
endif()

if(onnxruntime_USE_OPENVINO)
  add_definitions(-DUSE_OPENVINO)
endif()

if(OPENCV_FOUND)
  add_subdirectory(squeezenet_classification)
endif()

add_subdirectory(model-explorer)
