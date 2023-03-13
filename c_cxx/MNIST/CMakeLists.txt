# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

add_executable(mnist MNIST.cpp)
target_link_options(mnist PRIVATE "/SUBSYSTEM:WINDOWS")

copy_ort_dlls(mnist)

add_custom_command(TARGET mnist POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/MNIST/mnist.onnx ${CMAKE_BINARY_DIR}/MNIST)
