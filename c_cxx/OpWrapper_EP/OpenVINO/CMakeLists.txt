# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

project(opwrapper_ov_test)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root directory")

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files (x86)/onnxruntime")
  else()
    include_directories("/usr/local/include/onnxruntime")
  endif()
endif()

# opwrapper_provider_factory.h contains EP-specific APIs.
include(CheckIncludeFileCXX)
set(CMAKE_REQUIRED_INCLUDES "${ONNXRUNTIME_ROOTDIR}/include")
CHECK_INCLUDE_FILE_CXX(opwrapper_provider_factory.h HAVE_OPWRAPPER_PROVIDER_FACTORY_H)

include(CMakePrintHelpers)
cmake_print_variables(ONNXRUNTIME_ROOTDIR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories( 
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/providers/opwrapper/
)
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

add_executable(opwrapper_ov_test main.cpp utils.cpp)

if(HAVE_OPWRAPPER_PROVIDER_FACTORY_H)
    target_compile_definitions(opwrapper_ov_test PRIVATE -DHAVE_OPWRAPPER_PROVIDER_FACTORY_H)
endif()

target_link_libraries(opwrapper_ov_test onnxruntime)

add_subdirectory(custom_op)

if(WIN32)
  # Copy onnxruntime.dll into the executable's directory.
  add_custom_command(TARGET opwrapper_ov_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_ROOTDIR}/bin/onnxruntime.dll"
    $<TARGET_FILE_DIR:opwrapper_ov_test>)
endif()

# TODO: Run a script that 1) downloads OpenVINO model and
# 2) runs a python program to create the onnx model wrapper.
# Copy ONNX model to executable's directory.
add_custom_command(TARGET opwrapper_ov_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${CMAKE_SOURCE_DIR}/custom_op_ov_ep_wrapper.onnx"
  "${CMAKE_BINARY_DIR}/data/custom_op_ov_ep_wrapper.onnx")
