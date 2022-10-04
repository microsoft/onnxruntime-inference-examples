// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <onnxruntime_cxx_api.h>
#include <opwrapper_cxx_api.h>

template <size_t N>
static int64_t GetShapeSize(const std::array<int64_t, N>& shape) {
  int64_t size = 1;

  for (auto dim : shape) {
    size *= dim;
  }

  return size;
}

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
    Ort::OpWrapper::ProviderOptions op_options;
    op_options.UpdateOptions({{"device_type", "CPU"}});

    Ort::OpWrapper::AppendExecutionProvider(session_opts, "OpenVINO_EP_Wrapper", op_options);
    Ort::Session session(env, model_path, session_opts);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Setup input
    std::array<int64_t, 4> input_shape{1, 3, 224, 224};
    std::vector<float> input_vals(GetShapeSize(input_shape), 1.0f);
    std::array<const char*, 1> input_names{"data"};
    std::array<Ort::Value, 1> ort_inputs{Ort::Value::CreateTensor<float>(memory_info, input_vals.data(),
                                                                         input_vals.size(), input_shape.data(),
                                                                         input_shape.size())};

    // Run session and get output
    std::array<const char*, 1> output_names{"prob"};
    std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr},
                                                      input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                                      output_names.data(), output_names.size());

    // Print probabilities.
    Ort::Value& ort_output = ort_outputs[0];

    auto typeshape = ort_output.GetTensorTypeAndShapeInfo();
    const size_t num_probs = typeshape.GetElementCount();
    const float* probs = ort_output.GetTensorData<float>();

    std::ostringstream probs_sstream;
    for (size_t i = 0; i < num_probs; ++i) {
      probs_sstream << i << ": " << probs[i] << std::endl;
    }

    std::cout << "Probabilities:" << std::endl << probs_sstream.str() << std::endl;

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
 

  return 0;
}
