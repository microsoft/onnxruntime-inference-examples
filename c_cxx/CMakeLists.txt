# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files (x86)/onnxruntime")
  else()
    include_directories("/usr/local/include/onnxruntime")
  endif()
endif()

if(WIN32)
  add_library(wil INTERFACE)


  FetchContent_Declare(
      microsoft_wil
      URL https://github.com/microsoft/wil/archive/refs/tags/v1.0.220914.1.zip
    )
  FetchContent_Populate(microsoft_wil)
  target_include_directories(wil INTERFACE ${microsoft_wil_SOURCE_DIR}/include)
  set(WIL_LIB wil)
endif()

#TODO: we should only need one of them.
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# On Linux the samples use libjpeg and libpng for decoding images.
# On Windows they use Windows Image Component(WIC)
if(NOT WIN32)
    find_package(JPEG)
    if(LIBPNG_ROOTDIR)
      set(PNG_FOUND true)
      set(PNG_LIBRARIES png16)
      set(PNG_INCLUDE_DIRS "${LIBPNG_ROOTDIR}/include")
      set(PNG_LIBDIR "${LIBPNG_ROOTDIR}/lib")
    else()
      find_package(PNG)
    endif()
endif()

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_OPENVINO)
  add_definitions(-DUSE_OPENVINO)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  add_definitions(-DUSE_NNAPI)
endif()
if(onnxruntime_USE_DNNL)
  add_definitions(-DUSE_DNNL)
endif()
if(onnxruntime_USE_NUPHAR)
  add_definitions(-DUSE_NUPHAR)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

# some examples require a Windows build environment
if(WIN32)
  add_subdirectory(imagenet)
  add_subdirectory(MNIST)
endif()
add_subdirectory(squeezenet)
if(WIN32 OR PNG_FOUND)
  add_subdirectory(fns_candy_style_transfer)
endif()
#missing experimental_onnxruntime_cxx_api.h
#add_subdirectory(model-explorer)
