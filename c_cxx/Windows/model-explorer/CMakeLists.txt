# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(model-explorer model-explorer.cpp)
target_link_libraries(model-explorer PRIVATE onnxruntime)

if(WIN32)
	if(LIB_PACKAGE_DIR)
		set(DLL_DIRS "${LIB_PACKAGE_DIR}/ort_libs")
		file(GLOB OV_DLLS ${LIB_PACKAGE_DIR}/ov_libs/*)
		foreach(OvDll IN LISTS OV_DLLS)
		  add_custom_command(TARGET model-explorer POST_BUILD 
						COMMAND ${CMAKE_COMMAND} -E copy_if_different
						"${OvDll}"              
						$<TARGET_FILE_DIR:model-explorer>)
		endforeach()
	else()
		#In onnxruntime deafault install path, the required dlls are in lib and bin folders
		set(DLL_DIRS "${ONNXRUNTIME_ROOTDIR}/lib;${ONNXRUNTIME_ROOTDIR}/bin")
	endif()
	foreach(DLL_DIR IN LISTS DLL_DIRS)
		file(GLOB ALL_DLLS ${DLL_DIR}/*.dll)
		foreach(CurrentDll IN LISTS ALL_DLLS)
		  add_custom_command(TARGET model-explorer POST_BUILD 
						COMMAND ${CMAKE_COMMAND} -E copy_if_different
						"${CurrentDll}"              
						$<TARGET_FILE_DIR:model-explorer>)
		endforeach()
	endforeach()
endif()