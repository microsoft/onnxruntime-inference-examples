// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <onnxruntime_cxx_api.h>

bool CheckStatus(const OrtApi* g_ort, OrtStatus* status) {
  if (status != nullptr) {
    const char* msg = g_ort->GetErrorMessage(status);
    std::cerr << msg << std::endl;
    g_ort->ReleaseStatus(status);
    throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
  }
  return true;
}

void run_ort_snpe_ep(std::string backend, std::string input_path) {
#ifdef _WIN32
  const wchar_t* model_path = L"snpe_inception_v3.onnx";
#else
  const char* model_path = "snpe_inception_v3.onnx";
#endif

  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env;
  CheckStatus(g_ort, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  OrtSessionOptions* session_options;
  CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
  CheckStatus(g_ort, g_ort->SetIntraOpNumThreads(session_options, 1));
  CheckStatus(g_ort, g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC));

  std::vector<const char*> options_keys = {"runtime", "buffer_type"};
  std::vector<const char*> options_values = {backend.c_str(), "FLOAT"};  // set to TF8 if use quantized data

  CheckStatus(g_ort, g_ort->SessionOptionsAppendExecutionProvider(session_options, "SNPE", options_keys.data(),
                                                                  options_values.data(), options_keys.size()));
  OrtSession* session;
  CheckStatus(g_ort, g_ort->CreateSession(env, model_path, session_options, &session));

  OrtAllocator* allocator;
  CheckStatus(g_ort, g_ort->GetAllocatorWithDefaultOptions(&allocator));
  size_t num_input_nodes;
  CheckStatus(g_ort, g_ort->SessionGetInputCount(session, &num_input_nodes));

  std::vector<const char*> input_node_names;
  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<OrtValue*> input_tensors;

  input_node_names.resize(num_input_nodes);
  input_node_dims.resize(num_input_nodes);
  input_types.resize(num_input_nodes);
  input_tensors.resize(num_input_nodes);

  for (size_t i = 0; i < num_input_nodes; i++) {
    // Get input node names
    char* input_name;
    CheckStatus(g_ort, g_ort->SessionGetInputName(session, i, allocator, &input_name));
    input_node_names[i] = input_name;

    // Get input node types
    OrtTypeInfo* typeinfo;
    CheckStatus(g_ort, g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CheckStatus(g_ort, g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
    CheckStatus(g_ort, g_ort->GetTensorElementType(tensor_info, &type));
    input_types[i] = type;

    // Get input shapes/dims
    size_t num_dims;
    CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
    input_node_dims[i].resize(num_dims);
    CheckStatus(g_ort, g_ort->GetDimensions(tensor_info, input_node_dims[i].data(), num_dims));

    size_t tensor_size;
    CheckStatus(g_ort, g_ort->GetTensorShapeElementCount(tensor_info, &tensor_size));

    if (typeinfo) g_ort->ReleaseTypeInfo(typeinfo);
  }

  size_t num_output_nodes;
  std::vector<const char*> output_node_names;
  std::vector<std::vector<int64_t>> output_node_dims;
  std::vector<OrtValue*> output_tensors;
  CheckStatus(g_ort, g_ort->SessionGetOutputCount(session, &num_output_nodes));
  output_node_names.resize(num_output_nodes);
  output_node_dims.resize(num_output_nodes);
  output_tensors.resize(num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; i++) {
    // Get output node names
    char* output_name;
    CheckStatus(g_ort, g_ort->SessionGetOutputName(session, i, allocator, &output_name));
    output_node_names[i] = output_name;

    OrtTypeInfo* typeinfo;
    CheckStatus(g_ort, g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CheckStatus(g_ort, g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    // Get output shapes/dims
    size_t num_dims;
    CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
    output_node_dims[i].resize(num_dims);
    CheckStatus(g_ort, g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims[i].data(), num_dims));

    size_t tensor_size;
    CheckStatus(g_ort, g_ort->GetTensorShapeElementCount(tensor_info, &tensor_size));

    if (typeinfo) g_ort->ReleaseTypeInfo(typeinfo);
  }

  OrtMemoryInfo* memory_info;
  size_t input_data_size = 1 * 299 * 299 * 3;
  size_t input_data_length = input_data_size * sizeof(float);
  std::vector<float> input_data(input_data_size, 1.0);

  std::ifstream input_raw_file(input_path, std::ios::binary);
  input_raw_file.seekg(0, std::ios::end);
  const size_t num_elements = input_raw_file.tellg() / sizeof(float);
  input_raw_file.seekg(0, std::ios::beg);
  input_raw_file.read(reinterpret_cast<char*>(&input_data[0]), num_elements * sizeof(float));

  CheckStatus(g_ort, g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  CheckStatus(g_ort, g_ort->CreateTensorWithDataAsOrtValue(
                         memory_info, reinterpret_cast<void*>(input_data.data()), input_data_length,
                         input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));
  g_ort->ReleaseMemoryInfo(memory_info);

  CheckStatus(g_ort, g_ort->Run(session, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
                                input_tensors.size(), output_node_names.data(), output_node_names.size(),
                                output_tensors.data()));

  size_t output_data_size = 1 * 1001;
  size_t output_data_length = output_data_size * sizeof(float);
  std::vector<float> output_data(output_data_length);
  void* output_buffer;
  CheckStatus(g_ort, g_ort->GetTensorMutableData(output_tensors[0], &output_buffer));
  float* float_buffer = reinterpret_cast<float*>(output_buffer);

  auto max = std::max_element(float_buffer, float_buffer + output_data_size);
  int max_index = static_cast<int>(std::distance(float_buffer, max));

  std::fstream label_file("imagenet_slim_labels.txt", std::ios::in);
  std::unordered_map<int, std::string> label_table;
  label_table.reserve(output_data_size);
  int i = 0;
  std::string line;
  while (std::getline(label_file, line)) {
    label_table.emplace(i++, line);
  }

  printf("%d, %f, %s \n", max_index, *max, label_table[max_index].c_str());
}

void PrintHelp() {
  std::cout << "To run the sample, use the following command:" << std::endl;
  std::cout << "Example: ./snpe_ep_sample --cpu <path_to_raw_input>" << std::endl;
  std::cout << "To Run with SNPE CPU backend. Example: ./snpe_ep_sample --cpu chairs.raw" << std::endl;
  std::cout << "To Run with SNPE DSP backend. Example: ./snpe_ep_sample --dsp chairs.raw" << std::endl;
}

constexpr const char* CPUBACKEDN = "--cpu";
constexpr const char* DSPBACKEDN = "--dsp";

int main(int argc, char* argv[]) {
  std::string backend = "CPU";

  if (argc != 3) {
    PrintHelp();
    return 1;
  }

  if (strcmp(argv[1], CPUBACKEDN) == 0) {
    backend = "CPU";
  } else if (strcmp(argv[1], DSPBACKEDN) == 0) {
    backend = "DSP";
  } else {
    std::cout << "This sample only support CPU, DSP." << std::endl;
    PrintHelp();
    return 1;
  }
  std::string input_path(argv[2]);

  run_ort_snpe_ep(backend, input_path);
  return 0;
}
