cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(customop_example C CXX)

set(CMAKE_CXX_STANDARD 17)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

include_directories("${ONNXRUNTIME_ROOTDIR}/include"
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

add_executable(example example.cc)

target_link_libraries(example onnxruntime)
