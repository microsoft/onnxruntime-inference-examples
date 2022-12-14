# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(run_squeezenet_io_buffer "squeezenet_cpp_app_io_buffer.cpp")
target_include_directories(run_squeezenet_io_buffer PRIVATE ${OPENCV_INCLUDE_DIRS} ${OPENCL_INCLUDE})
target_link_libraries(run_squeezenet_io_buffer PRIVATE onnxruntime)

if(OPENCV_LIBDIR)
  target_link_directories(run_squeezenet_io_buffer PRIVATE ${OPENCV_LIBDIR})
  foreach(RelLib DebLib IN ZIP_LISTS OPENCV_RELEASE_LIBRARIES OPENCV_DEBUG_LIBRARIES)
    target_link_libraries(run_squeezenet_io_buffer PRIVATE optimized ${RelLib} debug ${DebLib})
  endforeach()
endif()

if(OPENCL_LIB)
  target_link_directories(run_squeezenet_io_buffer PRIVATE ${OPENCL_LIB})
  target_link_libraries(run_squeezenet_io_buffer PRIVATE OpenCL.lib)
endif()

#In onnxruntime deafault install path, the required dlls are in lib and bin folders
set(DLL_DIRS "${ONNXRUNTIME_ROOTDIR}/lib;${ONNXRUNTIME_ROOTDIR}/bin")
foreach(DLL_DIR IN LISTS DLL_DIRS)
  file(GLOB ALL_DLLS ${DLL_DIR}/*.dll)
  foreach(ORTDll IN LISTS ALL_DLLS)
    add_custom_command(TARGET run_squeezenet_io_buffer POST_BUILD 
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${ORTDll}"              
				$<TARGET_FILE_DIR:run_squeezenet_io_buffer>)
  endforeach()
endforeach()