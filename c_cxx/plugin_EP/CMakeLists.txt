# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug -DORT_HOME=/path/to/ort_package/onnxruntime-win-x64-gpu-1.23.0
cmake_minimum_required(VERSION 3.26)
project(plugin_ep_app VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)

file(GLOB app_src "./*.cc")
add_executable(app ${app_src})

# Add dependencies
include(FetchContent)

# Add GSL
FetchContent_Declare(
  gsl
  GIT_REPOSITORY https://github.com/microsoft/GSL.git
  GIT_TAG        v4.0.0  # Use a specific tag or commit
)

FetchContent_MakeAvailable(gsl)


set(DEPS_PATH "${CMAKE_BINARY_DIR}/_deps")

target_include_directories(app PUBLIC "${ORT_HOME}/include"
                                      "${DEPS_PATH}/gsl-src/include" # GSL is header-only
)

target_link_libraries(app PUBLIC "onnxruntime.dll")
