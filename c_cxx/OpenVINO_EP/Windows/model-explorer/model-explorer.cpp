// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * This sample application demonstrates how to use components of the experimental C++ API
 * to query for model inputs/outputs and how to run inferrence on a model.
 *
 * This example is best run with one of the ResNet models (i.e. ResNet18) from the onnx model zoo at
 *   https://github.com/onnx/models
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *
 * 
 * In this example, we do the following:
 *  1) read in an onnx model
 *  2) print out some metadata information about inputs and outputs that the model expects
 *  3) generate random data for an input tensor
 *  4) pass tensor through the model and check the resulting tensor
 *
 */

#include <algorithm>  // std::generate
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <onnxruntime_cxx_api.h>

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= i;
  return total;
}

using namespace std;

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << "Usage: model-explorer.exe <onnx_model.onnx>" << endl;
    return -1;
  }

#ifdef _WIN32
  std::string str = argv[1];
  std::wstring wide_string = std::wstring(str.begin(), str.end());
  std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
  std::string model_file = argv[1];
#endif

  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
  Ort::SessionOptions session_options;
  //Appending OpenVINO Execution Provider API
  #ifdef USE_OPENVINO
    // Using OPENVINO backend
   std::unordered_map<std::string, std::string> options;
   options["device_type"] = "CPU";
   std::cout << "OpenVINO device type is set to: " << options["device_type"] << std::endl;
   session_options.AppendExecutionProvider_OpenVINO_V2(options);
  #endif
  Ort::Session session(env, model_file.c_str(), session_options);
  Ort::AllocatorWithDefaultOptions allocator;

  size_t num_input_nodes = session.GetInputCount();
  std::vector<std::string> input_names;
  std::vector<std::vector<int64_t>> input_shapes;

  cout << "Input Node Name/Shape (" << num_input_nodes << "):" << endl;
  for (size_t i = 0; i < num_input_nodes; i++) {
    // Get input name
    auto input_name = session.GetInputNameAllocated(i, allocator);
    input_names.push_back(std::string(input_name.get()));
    
    // Get input shape
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    input_shapes.push_back(input_dims);
    
    cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << endl;
    
  }

  size_t num_output_nodes = session.GetOutputCount();
  std::vector<std::string> output_names;
  std::vector<std::vector<int64_t>> output_shapes;
  
  cout << "Output Node Name/Shape (" << num_output_nodes << "):" << endl;
  for (size_t i = 0; i < num_output_nodes; i++) {
    // Get output name
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    output_names.push_back(std::string(output_name.get()));
    
    // Get output shape
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();
    output_shapes.push_back(output_dims);
    
    cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << endl;
    
  }
  
  assert(input_names.size() == 1 && output_names.size() == 1);

  // Create a single Ort tensor of random numbers
  auto input_shape = input_shapes[0];
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values(total_number_elements);
  std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 255; });
  
  // Create input tensor
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), 
                                                         input_tensor_values.size(), input_shape.data(), 
                                                         input_shape.size()));

  // double-check the dimensions of the input tensor
  assert(input_tensors[0].IsTensor() &&
         input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
  cout << "\ninput_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << endl;

  // Create input/output name arrays for Run()
  std::vector<const char*> input_names_char(input_names.size(), nullptr);
  std::vector<const char*> output_names_char(output_names.size(), nullptr);
  
  for (size_t i = 0; i < input_names.size(); i++) {
    input_names_char[i] = input_names[i].c_str();
  }
  for (size_t i = 0; i < output_names.size(); i++) {
    output_names_char[i] = output_names[i].c_str();
  }

  // pass data through model
  cout << "Running model...";
  try {
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), 
                                     input_tensors.data(), input_names_char.size(), 
                                     output_names_char.data(), output_names_char.size());
    cout << "done" << endl;

    // double-check the dimensions of the output tensors
    assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
    cout << "output_tensor_shape: " << print_shape(output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << endl;

  } catch (const Ort::Exception& exception) {
    cout << "ERROR running model inference: " << exception.what() << endl;
    exit(-1);
  }

  return 0;

}
