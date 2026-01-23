# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

# Sets ONNX Runtime header and library paths.
# If an ONNX Runtime directory is specified by `ORT_HOME`, this function will use that.
# Otherwise, this function will download ONNX Runtime.
function(set_onnxruntime_paths)
  set(options)
  set(one_value_keywords
      # Specifies directory containing ONNX Runtime header and library files.
      # Generally, `ORT_HOME`/include should contain the headers and `ORT_HOME`/lib should contain the libraries.
      # For Android, `ORT_HOME` should match the directory structure of the ORT AAR.
      # Optional. If unset or empty, the ONNX Runtime files will be downloaded to the build directory.
      ORT_HOME
      # Specifies the ONNX Runtime version to use if downloading ONNX Runtime.
      DEFAULT_ORT_VERSION
      # Specifies the name of the output variable that will contain the ONNX Runtime include directory.
      ORT_INCLUDE_DIR_VAR
      # Specifies the name of the output variable that will contain the ONNX Runtime library directory.
      ORT_LIBRARY_DIR_VAR)
  set(multi_value_keywords)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${one_value_keywords}" "${multi_value_keywords}")

  set(required_args ORT_INCLUDE_DIR_VAR ORT_LIBRARY_DIR_VAR)
  foreach(required_arg IN ITEMS ${required_args})
    if(NOT DEFINED arg_${required_arg})
      message(FATAL_ERROR "${required_arg} must be provided.")
    endif()
  endforeach()

  if(DEFINED arg_ORT_HOME AND (NOT arg_ORT_HOME STREQUAL ""))
    use_onnxruntime_home_and_set_paths(${arg_ORT_HOME} ort_include_dir ort_lib_dir)
  else()
    if(NOT DEFINED arg_DEFAULT_ORT_VERSION)
      message(FATAL_ERROR "DEFAULT_ORT_VERSION must be provided if ORT_HOME is not provided.")
    endif()
    download_onnxruntime_and_set_paths(${arg_DEFAULT_ORT_VERSION} ort_include_dir ort_lib_dir)
  endif()

  set(${arg_ORT_INCLUDE_DIR_VAR} ${ort_include_dir} PARENT_SCOPE)
  set(${arg_ORT_LIBRARY_DIR_VAR} ${ort_lib_dir} PARENT_SCOPE)
endfunction()

function(download_onnxruntime_and_set_paths ORT_VERSION ORT_INCLUDE_DIR_VAR ORT_LIBRARY_DIR_VAR)
  set(ORT_FEED_ORG_NAME "aiinfra")
  set(ORT_FEED_PROJECT "2692857e-05ef-43b4-ba9c-ccf1c22c437c")
  set(ORT_NIGHTLY_FEED_ID "7982ae20-ed19-4a35-a362-a96ac99897b7")
  set(ORT_PACKAGE_NAME "Microsoft.ML.OnnxRuntime")

  set(ORT_FETCH_URL "https://pkgs.dev.azure.com/${ORT_FEED_ORG_NAME}/${ORT_FEED_PROJECT}/_apis/packaging/feeds/${ORT_NIGHTLY_FEED_ID}/nuget/packages/${ORT_PACKAGE_NAME}/versions/${ORT_VERSION}/content?api-version=6.0-preview.1")

  message(STATUS "Using ONNX Runtime package ${ORT_PACKAGE_NAME} version ${ORT_VERSION}")

  FetchContent_Declare(
    ortlib
    URL ${ORT_FETCH_URL}
  )
  FetchContent_makeAvailable(ortlib)

  set(ORT_HEADER_DIR ${ortlib_SOURCE_DIR}/build/native/include)

  if(ANDROID)
    file(ARCHIVE_EXTRACT INPUT ${ortlib_SOURCE_DIR}/runtimes/android/native/onnxruntime.aar DESTINATION ${ortlib_SOURCE_DIR}/runtimes/android/native/)
    set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/android/native/jni/${ANDROID_ABI})
  elseif(IOS OR MAC_CATALYST)
    file(ARCHIVE_EXTRACT INPUT ${ortlib_SOURCE_DIR}/runtimes/ios/native/onnxruntime.xcframework.zip DESTINATION ${ortlib_SOURCE_DIR}/runtimes/ios/native/)
    set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/ios/native/)
  else()
    set(ORT_BINARY_PLATFORM "x64")
    if (APPLE)
      if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(ORT_BINARY_PLATFORM "arm64")
      endif()
      set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/osx-${ORT_BINARY_PLATFORM}/native)
    elseif(WIN32)
      if (CMAKE_GENERATOR_PLATFORM)
        if (CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64EC" OR CMAKE_GENERATOR_PLATFORM STREQUAL "arm64")
          set(ORT_BINARY_PLATFORM "arm64")
        endif()
      elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
        set(ORT_BINARY_PLATFORM "arm64")
      endif()
      set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/win-${ORT_BINARY_PLATFORM}/native)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(ORT_BINARY_PLATFORM "arm64")
      endif()
      set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/linux-${ORT_BINARY_PLATFORM}/native)
    else()
      message(FATAL_ERROR "Auto download ONNX Runtime for this platform is not supported.")
    endif()
  endif()

  set(${ORT_INCLUDE_DIR_VAR} ${ORT_HEADER_DIR} PARENT_SCOPE)
  set(${ORT_LIBRARY_DIR_VAR} ${ORT_LIB_DIR} PARENT_SCOPE)
endfunction()

function(use_onnxruntime_home_and_set_paths ORT_HOME ORT_INCLUDE_DIR_VAR ORT_LIBRARY_DIR_VAR)
  file(REAL_PATH ${ORT_HOME} ORT_HOME)

  if(ANDROID)
    # Paths are based on the directory structure of the ORT AAR.
    set(ORT_HEADER_DIR ${ORT_HOME}/headers)
    set(ORT_LIB_DIR ${ORT_HOME}/jni/${ANDROID_ABI})
  else()
    set(ORT_HEADER_DIR ${ORT_HOME}/include)
    set(ORT_LIB_DIR ${ORT_HOME}/lib)
  endif()

  if(NOT IS_DIRECTORY ${ORT_HEADER_DIR})
    message(FATAL_ERROR "ORT_HEADER_DIR (${ORT_HEADER_DIR}) is not a directory.")
  endif()

  if(NOT IS_DIRECTORY ${ORT_LIB_DIR})
    message(FATAL_ERROR "ORT_LIB_DIR (${ORT_LIB_DIR}) is not a directory.")
  endif()

  set(${ORT_INCLUDE_DIR_VAR} ${ORT_HEADER_DIR} PARENT_SCOPE)
  set(${ORT_LIBRARY_DIR_VAR} ${ORT_LIB_DIR} PARENT_SCOPE)
endfunction()
