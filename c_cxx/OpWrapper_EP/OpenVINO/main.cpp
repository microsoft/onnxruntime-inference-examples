// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opwrapper_cxx_api.h>

int main(int argc, char** argv) {
  std::cout << "ORT version: " << OrtGetApiBase()->GetVersionString() << std::endl;
  
#ifdef _WIN32
  const wchar_t* model_path = L"custom_op_ov_ep_wrapper.onnx";
  const char* custom_op_dll_path = "openvino_wrapper.dll";
#else
  const char* model_path = "custom_op_ov_ep_wrapper.onnx";
  const char* custom_op_dll_path = "libopenvino_wrapper.so";
#endif
  
  try {
    Ort::Env env;
    Ort::SessionOptions session_opts;

    // TODO: Add RegisterCustomOpsLibrary as a method of Ort::SessionOptions.
    void* lib_handle = nullptr;
    Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(static_cast<OrtSessionOptions*>(session_opts),
                                                             custom_op_dll_path,
                                                             &lib_handle));
    Ort::OpWrapper::ProviderOptions op_options(
      std::unordered_map<std::string, std::string>({{"device_type", "CPU"}})
    );
    Ort::OpWrapper::AppendExecutionProvider(session_opts, "OpenVINO_EP_Wrapper", op_options);

    Ort::Session session(env, model_path, session_opts);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
 

  return 0;
}
