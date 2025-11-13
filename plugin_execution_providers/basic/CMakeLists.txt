cmake_minimum_required(VERSION 3.31)

project(BasicPluginEp)

set(CMAKE_CXX_STANDARD 20)

if(NOT DEFINED ORT_HOME)
  message(FATAL_ERROR "ORT_HOME should be set to the directory containing the onnxruntime library and headers.")
endif()

set(ORT_LIBRARY_DIR ${ORT_HOME}/lib)
set(ORT_INCLUDE_DIR ${ORT_HOME}/include)

find_library(ORT_LIBRARY
  NAMES onnxruntime
  PATHS ${ORT_LIBRARY_DIR}
  REQUIRED)

message(VERBOSE "ORT_LIBRARY: ${ORT_LIBRARY}")

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
  ${CMAKE_SOURCE_DIR}/src/plugin_ep_utils.h
)

target_link_libraries(basic_plugin_ep PRIVATE ${ORT_LIBRARY})
target_include_directories(basic_plugin_ep PRIVATE ${ORT_INCLUDE_DIR})

set(basic_plugin_ep_link_options)
if(MSVC)
  list(APPEND basic_plugin_ep_link_options
       "-DEF:${CMAKE_SOURCE_DIR}/src/ep_lib.def"
  )
else()
  if(UNIX)
    if(APPLE)
      list(APPEND basic_plugin_ep_link_options
           "-Xlinker" "-dead_strip")
    else()
      list(APPEND basic_plugin_ep_link_options
           "-Xlinker" "--version-script=${CMAKE_SOURCE_DIR}/src/ep_lib.lds"
           "-Xlinker" "--no-undefined"
           "-Xlinker" "--gc-sections"
           "-z" "noexecstack")
    endif()
  endif()
endif()

target_link_options(basic_plugin_ep PRIVATE ${basic_plugin_ep_link_options})

endblock()
