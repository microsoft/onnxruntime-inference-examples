project(snpe_ep_sample C CXX)

set(CMAKE_BUILD_TYPE Release)

cmake_minimum_required(VERSION 3.13)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/)

  
ADD_EXECUTABLE(snpe_ep_sample  main.cpp)
if(HAVE_TENSORRT_PROVIDER_FACTORY_H)
  target_compile_definitions(snpe_ep_sample PRIVATE -DHAVE_TENSORRT_PROVIDER_FACTORY_H)
endif()

set(ORT_LIBS onnxruntime)
if(ANDROID)
list(APPEND ORT_LIBS libc++_shared.so)
endif()
target_link_libraries(snpe_ep_sample ${ORT_LIBS})

if(MSVC)
  target_link_directories(snpe_ep_sample PRIVATE ${ONNXRUNTIME_ROOTDIR}/build/Windows/Release/Release)
else()
  target_link_directories(snpe_ep_sample PRIVATE ${ONNXRUNTIME_ROOTDIR}/build/Android/Release)
endif()
