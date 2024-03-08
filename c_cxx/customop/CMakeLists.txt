cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(customop_example C CXX)

set(CMAKE_CXX_STANDARD 17)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

add_executable(example example.cc)

target_include_directories(example PRIVATE "${ONNXRUNTIME_ROOTDIR}/include"
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")

target_link_directories(example PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib")

target_link_libraries(example onnxruntime)
