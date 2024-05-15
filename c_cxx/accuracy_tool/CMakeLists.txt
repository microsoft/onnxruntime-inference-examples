# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)
project(onnxruntime_accuracy_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ONNXRUNTIME_ROOTDIR "ONNX Runtime root directory")
option(QNN_SDK_ROOTDIR "QNN SDK root directory")
option(QNN_HEXAGON_ARCH_VERSION "QNN Hexagon architecture version (e.g., 68 or 73)")

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
  else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()

  if(NOT EXISTS "${ONNXRUNTIME_ROOTDIR}")
    message(FATAL_ERROR "Must set a valid ONNX Runtime root directory using -DONNXRUNTIME_ROOTDIR")
  else()
    message(WARNING "Did not set ONNXRUNTIME_ROOTDIR. Using: ${ONNXRUNTIME_ROOTDIR}")
  endif()
endif()

# QNN_SDK_ROOTDIR is optinal, but must exist if provided.
if(QNN_SDK_ROOTDIR)
  if(NOT EXISTS "${QNN_SDK_ROOTDIR}")
    message(FATAL_ERROR "The provided QNN SDK directory does not exist: ${QNN_SDK_ROOTDIR}")
  endif()

  if(NOT QNN_HEXAGON_ARCH_VERSION)
    set(QNN_HEXAGON_ARCH_VERSION "73")
    message(WARNING "Assuming QNN Hexagon version ${QNN_HEXAGON_ARCH_VERSION}. Set with -DQNN_HEXAGON_ARCH_VERSION")
  endif()

  set(QNN_SDK_HEX_DIR "${QNN_SDK_ROOTDIR}/lib/hexagon-v${QNN_HEXAGON_ARCH_VERSION}/unsigned")
  if(NOT EXISTS "${QNN_SDK_HEX_DIR}")
    message(FATAL_ERROR "The QNN SDK Hexagon directory does not exist: ${QNN_SDK_HEX_DIR}")
  endif()
endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

add_executable(accuracy_test src/main.cc
                             src/accuracy_tester.h
                             src/accuracy_tester.cc
                             src/cmd_args.h
                             src/cmd_args.cc
                             src/ep_cmd_args/qnn_cmd_args.h
                             src/ep_cmd_args/qnn_cmd_args.cc
                             src/basic_utils.h
                             src/basic_utils.cc
                             src/model_io_utils.h
                             src/model_io_utils.cc
                             src/data_loader.h
                             src/data_loader.cc
                             src/acc_task.h
                             src/acc_task.cc
                             src/task_thread_pool.h
                             src/task_thread_pool.cc)
target_include_directories(accuracy_test PUBLIC "${PROJECT_SOURCE_DIR}/src")

find_package(Threads REQUIRED)
target_link_libraries(accuracy_test onnxruntime Threads::Threads)

function(target_copy_artifacts target_name artifacts_glob)
  if (MSVC)
    file(GLOB ARTIFACTS ${artifacts_glob})
    foreach(ARTIFACT ${ARTIFACTS})
        add_custom_command(TARGET ${target_name} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${ARTIFACT}  $<TARGET_FILE_DIR:${target_name}>)
    endforeach()
  endif()
endfunction()

# Windows might have an onnxruntime.dll in the system directory so it's more robust to manually copy the dlls to
# the output dir.
if(MSVC)
  target_copy_artifacts(accuracy_test "${ONNXRUNTIME_ROOTDIR}/bin/*")
endif()

if(EXISTS "${QNN_SDK_ROOTDIR}")
  set(QNN_SDK_HEX_DIR "${QNN_SDK_ROOTDIR}/lib/hexagon-v${QNN_HEXAGON_ARCH_VERSION}/unsigned")
  target_copy_artifacts(accuracy_test "${QNN_SDK_HEX_DIR}/*.so")
  target_copy_artifacts(accuracy_test "${QNN_SDK_HEX_DIR}/*.cat")

  if(WIN32)
    set(QNN_WIN_ARCH_DIR "x86_64-windows-msvc")
    if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ARM64")
      set(QNN_WIN_ARCH_DIR "aarch64-windows-msvc")
    endif()

    set(QNN_SDK_WIN_LIBS_DIR "${QNN_SDK_ROOTDIR}/lib/${QNN_WIN_ARCH_DIR}")
    target_copy_artifacts(accuracy_test "${QNN_SDK_WIN_LIBS_DIR}/*.dll")
    target_copy_artifacts(accuracy_test "${QNN_SDK_WIN_LIBS_DIR}/*.lib")
  else()
    message(FATAL_ERROR "TODO: Copy QNN artifacts on Linux")
  endif()
endif()
