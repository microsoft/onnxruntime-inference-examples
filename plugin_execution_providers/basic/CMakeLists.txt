# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.31)

project(BasicPluginEp)

set(CMAKE_CXX_STANDARD 20)

set(plugin_ep_common_dir ${CMAKE_SOURCE_DIR}/../common)

include(${plugin_ep_common_dir}/cmake/onnxruntime_library_utils.cmake)

# Set ORT_HOME (e.g., with cmake -DORT_HOME=/path/to/ort_home) to specify the directory with the ONNX Runtime header
# and library files to use.
# This is optional. If unspecified, the ONNX Runtime files will be downloaded.

set_onnxruntime_paths(
  ORT_HOME ${ORT_HOME}
  DEFAULT_ORT_VERSION "1.23.2"
  ORT_INCLUDE_DIR_VAR ORT_INCLUDE_DIR
  ORT_LIBRARY_DIR_VAR ORT_LIBRARY_DIR)

message(STATUS "ORT_LIBRARY_DIR: ${ORT_LIBRARY_DIR}")
message(STATUS "ORT_INCLUDE_DIR: ${ORT_INCLUDE_DIR}")

#
# basic_plugin_ep
#
block()

add_library(basic_plugin_ep MODULE)

target_sources(basic_plugin_ep PRIVATE
  ${CMAKE_SOURCE_DIR}/src/ep_factory.cc
  ${CMAKE_SOURCE_DIR}/src/ep_factory.h
  ${CMAKE_SOURCE_DIR}/src/ep_lib_entry.cc
  ${CMAKE_SOURCE_DIR}/src/ep.cc
  ${CMAKE_SOURCE_DIR}/src/ep.h
  ${plugin_ep_common_dir}/src/plugin_ep_utils.h
)

target_include_directories(basic_plugin_ep PRIVATE
  ${ORT_INCLUDE_DIR}
  ${plugin_ep_common_dir}/src
)

target_link_directories(basic_plugin_ep PRIVATE ${ORT_LIBRARY_DIR})
target_link_libraries(basic_plugin_ep PRIVATE onnxruntime)

set(basic_plugin_ep_link_options)
if(MSVC)
  list(APPEND basic_plugin_ep_link_options
       "-DEF:${CMAKE_SOURCE_DIR}/src/ep_lib.def")
else()
  if(UNIX)
    if(APPLE)
      list(APPEND basic_plugin_ep_link_options
           "LINKER:-dead_strip")
    else()
      list(APPEND basic_plugin_ep_link_options
           "LINKER:--version-script=${CMAKE_SOURCE_DIR}/src/ep_lib.lds"
           "LINKER:--no-undefined"
           "LINKER:--gc-sections"
           "-z" "noexecstack")
    endif()
  endif()
endif()

target_link_options(basic_plugin_ep PRIVATE ${basic_plugin_ep_link_options})

endblock()
