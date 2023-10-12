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

template <typename T_QuantType>
void QuantizedData(T_QuantType* out, const float* in, int32_t offset, float scale, size_t num_elements) {
  static_assert(std::is_unsigned<T_QuantType>::value, "QuantizedData supports unsigned only!");

  if (nullptr == out || nullptr == in) {
    throw Ort::Exception("Received a nullptr", OrtErrorCode::ORT_EP_FAIL);
  }

  size_t data_type_size_in_bytes = sizeof(T_QuantType);
  size_t bit_width = data_type_size_in_bytes * 8;
  double true_bit_width_max = pow(2, bit_width) - 1;
  double encoding_min = offset * scale;
  double encoding_max = (true_bit_width_max + offset) * scale;
  double encoding_range = encoding_max - encoding_min;

  for (size_t i = 0; i < num_elements; ++i) {
    int quantized_value = static_cast<int>(round(true_bit_width_max * (in[i] - encoding_min) / encoding_range));
    if (quantized_value < 0) {
      quantized_value = 0;
    } else if (quantized_value > (int)true_bit_width_max) {
      quantized_value = (int)true_bit_width_max;
    }
    out[i] = static_cast<T_QuantType>(quantized_value);
  }
}


template <typename T_QuantType>
void DequantizedData(float* out, const T_QuantType* in, int32_t offset, float scale, size_t num_elements) {
  static_assert(std::is_unsigned<T_QuantType>::value, "DequantizedData supports unsigned only!");

  if (nullptr == out || nullptr == in) {
    throw Ort::Exception("Received a nullptr", OrtErrorCode::ORT_EP_FAIL);
  }

  for (size_t i = 0; i < num_elements; i++) {
    double quantized_value = static_cast<double>(in[i]);
    double offset_double = static_cast<double>(offset);
    out[i] = static_cast<float>((quantized_value + offset_double) * scale);
  }
}

void run_ort_qnn_ep(const std::string& backend, const std::string& model_path, const std::string& input_path,
                    bool generated_from_native_qnn, bool generate_ctx) {
  std::wstring model_path_wstr = std::wstring(model_path.begin(), model_path.end());

  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env;
  // Can set to ORT_LOGGING_LEVEL_INFO or ORT_LOGGING_LEVEL_VERBOSE for more info
  CheckStatus(g_ort, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  OrtSessionOptions* session_options;
  CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
  CheckStatus(g_ort, g_ort->SetIntraOpNumThreads(session_options, 1));
  CheckStatus(g_ort, g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC));

  // More option details refers to https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
  std::vector<const char*> options_keys = {"backend_path"};
  std::vector<const char*> options_values = {backend.c_str()};

  // If it runs from a QDQ model on HTP backend
  // It will generate an Onnx model with Qnn context binary.
  // The context binary can be embedded inside the model in EPContext->ep_cache_context (by default),
  // or the context binary can be a separate .bin file, with relative path set in EPContext->ep_cache_context (qnn_context_embed_mode = 0)
  if (generate_ctx) {
    options_keys.push_back("qnn_context_cache_enable");
    options_values.push_back("1");
  }
  // qnn_context_cache_path -- you can specify the path and file name as you want
  // If not specified, OnnxRuntime QNN EP will generate it at [model_path]_qnn_ctx.onnx

  CheckStatus(g_ort, g_ort->SessionOptionsAppendExecutionProvider(session_options, "QNN", options_keys.data(),
                                                                  options_values.data(), options_keys.size()));
  OrtSession* session;
  CheckStatus(g_ort, g_ort->CreateSession(env, model_path_wstr.c_str(), session_options, &session));
  if (generate_ctx) {
    printf("\nOnnx model with QNN context binary is generated.\n");
    return;
  }

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
  size_t input_data_size = 1 * 3 * 224 * 224;
  std::vector<float> input_data(input_data_size, 1.0);
  std::vector<uint8_t> quantized_input_data(input_data_size * sizeof(uint8_t), 1);

  std::ifstream input_raw_file(input_path, std::ios::binary);
  input_raw_file.seekg(0, std::ios::end);
  const size_t num_elements = input_raw_file.tellg() / sizeof(float);
  input_raw_file.seekg(0, std::ios::beg);
  input_raw_file.read(reinterpret_cast<char*>(&input_data[0]), num_elements * sizeof(float));

  CheckStatus(g_ort, g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  // QNN native tool chain generated quantized model use quantized data as inputs & outputs
  if (generated_from_native_qnn) {
    size_t input_data_length = input_data_size * sizeof(uint8_t);
    QuantizedData(quantized_input_data.data(), input_data.data(), -116, 0.015875209f, input_data_size);
    CheckStatus(g_ort, g_ort->CreateTensorWithDataAsOrtValue(
                           memory_info, reinterpret_cast<void*>(quantized_input_data.data()), input_data_length,
                           input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));
  } else { // Ort generate QDQ model still use float32 data as inputs & outputs
    size_t input_data_length = input_data_size * sizeof(float);
    CheckStatus(g_ort, g_ort->CreateTensorWithDataAsOrtValue(
                           memory_info, reinterpret_cast<void*>(input_data.data()), input_data_length,
                           input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));
  }
  g_ort->ReleaseMemoryInfo(memory_info);

  CheckStatus(g_ort, g_ort->Run(session, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
                                input_tensors.size(), output_node_names.data(), output_node_names.size(),
                                output_tensors.data()));

  size_t output_data_size = 1 * 1000;
  std::vector<float> output_data(output_data_size);
  void* output_buffer;
  CheckStatus(g_ort, g_ort->GetTensorMutableData(output_tensors[0], &output_buffer));
  float* float_buffer = nullptr;
  if (generated_from_native_qnn) {
    uint8_t* buffer = reinterpret_cast<uint8_t*>(output_buffer);
    DequantizedData(output_data.data(), buffer, -86, 0.08069417f, output_data_size);
    float_buffer = output_data.data();
  } else {
    float_buffer = reinterpret_cast<float*>(output_buffer);
  }

  auto max = std::max_element(float_buffer, float_buffer + output_data_size);
  int max_index = static_cast<int>(std::distance(float_buffer, max));

  std::fstream label_file("synset.txt", std::ios::in);
  std::unordered_map<int, std::string> label_table;
  label_table.reserve(output_data_size);
  int i = 0;
  std::string line;
  while (std::getline(label_file, line)) {
    label_table.emplace(i++, line);
  }

  printf("\nResult: \n");
  printf("position=%d, classification=%s, probability=%f \n", max_index, label_table[max_index].c_str(), *max);
}

void PrintHelp() {
  std::cout << "To run the sample, use the following command:" << std::endl;
  std::cout << "Example: ./qnn_ep_sample --cpu <model_path> <path_to_raw_input>" << std::endl;
  std::cout << "To Run with QNN CPU backend. Example: ./qnn_ep_sample --cpu mobilenetv2-12_shape.onnx kitten_input.raw" << std::endl;
  std::cout << "To Run with QNN HTP backend. Example: ./qnn_ep_sample --htp mobilenetv2-12_quant_shape.onnx kitten_input.raw" << std::endl;
  std::cout << "To Run with QNN HTP backend and generate Qnn context binary model. Example: ./qnn_ep_sample --htp mobilenetv2-12_quant_shape.onnx kitten_input.raw --gen_ctx" << std::endl;
  std::cout << "To Run with QNN native context binary on QNN HTP backend . Example: ./qnn_ep_sample --qnn qnn_native_ctx_binary.onnx kitten_input_nhwc.raw" << std::endl;
}

constexpr const char* CPUBACKEDN = "--cpu";
constexpr const char* HTPBACKEDN = "--htp";
constexpr const char* QNNCTXBINARY = "--qnn";
constexpr const char* GENERATE_CTX = "--gen_ctx";

int main(int argc, char* argv[]) {

  if (argc != 4 && argc != 5) {
    PrintHelp();
    return 1;
  }

  bool generate_ctx = false;
  if (argc == 5) {
    if (strcmp(argv[4], GENERATE_CTX) == 0) {
      generate_ctx = true;
    } else {
      std::cout << "The expected last parameter is --gen_ctx." << std::endl;
      PrintHelp();
      return 1;
    }
  }

  std::string backend = "";
  bool generated_from_native_qnn = false;
  if (strcmp(argv[1], CPUBACKEDN) == 0) {
    backend = "QnnCpu.dll";
    if (generate_ctx) {
      std::cout << "--gen_ctx won't work with CPU backend." << std::endl;
      return 1;
    }
  } else if (strcmp(argv[1], HTPBACKEDN) == 0) {
    backend = "QnnHtp.dll";
  } else if (strcmp(argv[1], QNNCTXBINARY) == 0) {
    backend = "QnnHtp.dll";
    generated_from_native_qnn = true;
    if (generate_ctx) {
      std::cout << "--gen_ctx won't work with --qnn." << std::endl;
      return 1;
    }
  } else {
    std::cout << "This sample only support option cpu, htp, qnn." << std::endl;
    PrintHelp();
    return 1;
  }

  std::string model_path(argv[2]);
  std::string input_path(argv[3]);

  run_ort_qnn_ep(backend, model_path, input_path, generated_from_native_qnn, generate_ctx);
  return 0;
}
