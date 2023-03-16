# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

add_executable(mnist MNIST.cpp)
target_link_options(mnist PRIVATE "/SUBSYSTEM:WINDOWS")

copy_ort_dlls(mnist)

# change working directory for VS debugger to same directory the binary is created in, and copy the model there.
set_property(TARGET mnist PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(OutDir)")
add_custom_command(TARGET mnist POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/MNIST/mnist.onnx $<TARGET_FILE_DIR:mnist>)
