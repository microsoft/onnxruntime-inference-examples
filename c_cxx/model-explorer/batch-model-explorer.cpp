// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * This example demonstrates how to batch process data using the C++ API.
 *
 * This example is based on the model-explorer.cpp example except it demonstrates how to
 * batch process data. Please start by checking out model-explorer.cpp first.
 *
 * This example is best run with one of the ResNet models (i.e. ResNet18) from the onnx model zoo at
 *   https://github.com/onnx/models
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *  2) The onnx model has a symbolic first dimension (i.e. -1x3x224x224)
 *
 *
 * In this example, we do the following:
 *  1) read in an onnx model
 *  2) print out some metadata information about inputs and outputs that the model expects
 *  3) create tensors by generating 3 random batches of data (with batch_size = 5) for input to the model
 *  4) pass each batch through the model and check the resulting output
 *
 *
 * NOTE: Some onnx models may not have a symbolic first dimension. To prepare the onnx model, see the python code snippet below.
 * =============  Python Example  ======================
 * import onnx
 * model = onnx.load_model('model.onnx')
 * model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
 * onnx.save_model(model, 'model-symbolic.onnx')
 *
 */

#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t>& v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

// XXX: this function does not handle integer overflow
int calculate_product(const std::vector<std::int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= static_cast<int>(i);
  return total;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

#ifdef _WIN32
int wmain(int argc, ORTCHAR_T* argv[]) {
#else
int main(int argc, ORTCHAR_T* argv[]) {
#endif
  if (argc != 2) {
    std::cout << "Usage: ./batch-model-explorer <onnx_model.onnx>" << std::endl;
    return -1;
  }

  std::basic_string<ORTCHAR_T> model_file = argv[1];

  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "batch-model-explorer");
  Ort::SessionOptions session_options;
  Ort::Session session = Ort::Session(env, model_file.c_str(), session_options);

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_shapes;
  std::cout << "Input Node Name/Shape (" << session.GetInputCount() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetInputCount(); i++) {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
  }

  // print name/shape of outputs
  std::vector<std::string> output_names;
  std::cout << "Output Node Name/Shape (" << session.GetOutputCount() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
    output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
  }

  // Assume model has 1 input node and 1 output node.
  assert(input_names.size() == 1 && output_names.size() == 1);

  int batch_size = 5;
  int num_batches = 3;
  auto input_shape = input_shapes;
  assert(input_shape[0] < 0);  // symbolic dimensions are represented by a negative value (typically -1)
  input_shape[0] = batch_size;
  int num_elements_per_batch = calculate_product(input_shape);

  // Prepare input and output name char arrays for session.Run()
  std::vector<const char*> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  std::vector<const char*> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  // process multiple batches
  for (int i = 0; i < num_batches; i++) {
    std::cout << "\nProcessing batch #" << i << std::endl;

    // Create an Ort tensor containing random numbers
    std::vector<float> batch_input_tensor_values(num_elements_per_batch);
    std::generate(batch_input_tensor_values.begin(), batch_input_tensor_values.end(),
                  [&] { return static_cast<float>(rand() % 255); });
    std::vector<Ort::Value> batch_input_tensors;
    batch_input_tensors.emplace_back(vec_to_tensor<float>(batch_input_tensor_values, input_shape));

    // double-check the dimensions of the input tensor
    assert(batch_input_tensors[0].IsTensor() &&
           batch_input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
    std::cout << "batch_input_tensor shape: "
              << print_shape(batch_input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;

    // pass data through model
    try {
      auto batch_output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), batch_input_tensors.data(),
                                              input_names_char.size(), output_names_char.data(), output_names_char.size());
      // double-check the dimensions of the output tensors
      // NOTE: the number of output tensors is equal to the number of output nodes specified in the Run() call
      assert(batch_output_tensors.size() == output_names.size() &&
             batch_output_tensors[0].IsTensor() &&
             batch_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[0] == batch_size);
      std::cout << "batch_output_tensor_shape: "
                << print_shape(batch_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
    } catch (const Ort::Exception& exception) {
      std::cout << "ERROR running model inference: " << exception.what() << std::endl;
      exit(-1);
    }
  }
  std::cout << "\nDone" << std::endl;
  return 0;
}
