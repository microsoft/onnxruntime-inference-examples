project(qnn_ep_sample C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

cmake_minimum_required(VERSION 3.13)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
option(ONNXRUNTIME_BUILDDIR "onnxruntime build dir")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${ONNXRUNTIME_ROOTDIR}/include/)
include_directories(${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/)
  
ADD_EXECUTABLE(qnn_ep_sample  main.cpp)

set(ORT_LIBS onnxruntime)
if(ANDROID)
list(APPEND ORT_LIBS libc++_shared.so)
endif()
target_link_libraries(qnn_ep_sample ${ORT_LIBS})

if(MSVC)
  target_link_directories(qnn_ep_sample PRIVATE ${ONNXRUNTIME_BUILDDIR})
else()
  target_link_directories(qnn_ep_sample PRIVATE ${ONNXRUNTIME_ROOTDIR}/build/Android/${CMAKE_BUILD_TYPE})
endif()
