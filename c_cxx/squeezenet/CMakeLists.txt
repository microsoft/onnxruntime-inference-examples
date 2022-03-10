project(capi_test)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required (VERSION 2.6)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories( 
    # The ${PROJECT_SOURCE_DIR} is used in Azure DevOps Pipelines only.
    # You can also specify ONNXRuntime include files from your local machine.
    ${PROJECT_SOURCE_DIR}/../../include/onnxruntime/core/session/
    ${PROJECT_SOURCE_DIR}/../../include/onnxruntime/core/providers/tensorrt/
)
        
ADD_EXECUTABLE(capi_test  main.cpp)
target_link_libraries(capi_test onnxruntime)
