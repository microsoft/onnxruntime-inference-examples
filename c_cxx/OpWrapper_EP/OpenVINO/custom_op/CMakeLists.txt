find_package(OpenVINO REQUIRED)

add_library(openvino_wrapper MODULE custom_op.cpp openvino_wrapper.cpp)
target_link_libraries(openvino_wrapper PRIVATE openvino::runtime)

if(UNIX)
  if (APPLE)
    set(OPENVINO_WRAPPER_LIB_LINK_FLAG "-Xlinker -dead_strip")
  else()
    set(OPENVINO_WRAPPER_LIB_LINK_FLAG "-Xlinker --version-script=${CMAKE_CURRENT_SOURCE_DIR}/custom_op.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
  endif()
else()
  set(OPENVINO_WRAPPER_LIB_LINK_FLAG "-DEF:${CMAKE_CURRENT_SOURCE_DIR}/custom_op.def")
endif()

set_property(TARGET openvino_wrapper APPEND_STRING PROPERTY LINK_FLAGS ${OPENVINO_WRAPPER_LIB_LINK_FLAG})
