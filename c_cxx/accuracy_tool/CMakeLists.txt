# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(accuracy_tool main.cc
                             cmd_args.h
                             cmd_args.cc
                             basic_utils.h
                             basic_utils.cc
                             model_io_utils.h
                             model_io_utils.cc
                             data_loader.h
                             data_loader.cc
                             acc_task.h
                             acc_task.cc
                             task_thread_pool.h
                             task_thread_pool.cc)

find_package(Threads REQUIRED)
target_link_libraries(accuracy_tool onnxruntime Threads::Threads)

# Copy onnxruntime.dll into the executable's directory
copy_ort_dlls(accuracy_tool)
