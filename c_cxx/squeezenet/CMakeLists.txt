project(capi_test)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.13)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories( 
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/providers/tensorrt/
)
        
ADD_EXECUTABLE(capi_test  main.cpp)
target_link_libraries(capi_test onnxruntime)
