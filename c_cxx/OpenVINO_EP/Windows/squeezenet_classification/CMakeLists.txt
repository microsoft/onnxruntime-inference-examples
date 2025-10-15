# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(run_squeezenet "squeezenet_cpp_app.cpp")
target_include_directories(run_squeezenet PRIVATE ${OPENCV_INCLUDE_DIRS})
target_link_libraries(run_squeezenet PRIVATE onnxruntime)

if(OPENCV_LIBDIR)
  target_link_directories(run_squeezenet PRIVATE ${OPENCV_LIBDIR})
  foreach(RelLib DebLib IN ZIP_LISTS OPENCV_RELEASE_LIBRARIES OPENCV_DEBUG_LIBRARIES)
    target_link_libraries(run_squeezenet PRIVATE optimized ${RelLib} debug ${DebLib})
  endforeach()
endif()

#In onnxruntime deafault install path, the required dlls are in lib and bin folders
set(DLL_DIRS "${ONNXRUNTIME_ROOTDIR}/lib;${ONNXRUNTIME_ROOTDIR}/bin;${OPENCV_ROOTDIR}/x64/vc16/bin")

foreach(DLL_DIR IN LISTS DLL_DIRS)
  file(GLOB ALL_DLLS ${DLL_DIR}/*.dll)
  foreach(DLLFile IN LISTS ALL_DLLS)
    add_custom_command(TARGET run_squeezenet POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLLFile}"
        $<TARGET_FILE_DIR:run_squeezenet>)
  endforeach()
endforeach()