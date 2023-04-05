project(qnn_ep_sample C CXX)

set(CMAKE_BUILD_TYPE Release)

cmake_minimum_required(VERSION 3.13)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${ONNXRUNTIME_ROOTDIR}/include)
        
ADD_EXECUTABLE(qnn_ep_sample  main.cpp)
target_link_libraries(qnn_ep_sample onnxruntime)
