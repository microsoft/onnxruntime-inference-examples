#include <onnxruntime_lite_custom_op.h>

#include <ctime>
#include <iostream>
#include <vector>

using namespace std;
using namespace Ort::Custom;

void KernelOne(const Ort::Custom::Tensor<float>& X, const Ort::Custom::Tensor<float>& Y,
               Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();
  auto x_raw = X.Data();
  auto y_raw = Y.Data();
  auto z_raw = Z.Allocate(input_shape);
  for (int64_t i = 0; i < Z.NumberOfElement(); ++i) {
    z_raw[i] = x_raw[i] + y_raw[i];
  }
}

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::CustomOpDomain v1_domain{"v1"};
  // please make sure that custom_op_one has the same lifetime as the consuming session
  std::unique_ptr<OrtLiteCustomOp> custom_op_one{
      Ort::Custom::CreateLiteCustomOp("CustomOpOne", "CPUExecutionProvider", KernelOne)};
  v1_domain.Add(custom_op_one.get());
  Ort::SessionOptions session_options;
  session_options.Add(v1_domain);

#ifdef _WIN32
  const wchar_t* model_path = L"custom_kernel_one_model.onnx";
#else
  const char* model_path = "custom_kernel_one_model.onnx";
#endif

  Ort::Session session(env, model_path, session_options);

  // Get input/output node names
  using AllocatedStringPtr = std::unique_ptr<char, Ort::detail::AllocatedFree>;
  std::vector<const char*> input_names;
  std::vector<AllocatedStringPtr> inputNodeNameAllocatedStrings;
  std::vector<const char*> output_names;
  std::vector<AllocatedStringPtr> outputNodeNameAllocatedStrings;
  Ort::AllocatorWithDefaultOptions allocator;
  size_t numInputNodes = session.GetInputCount();
  for (int i = 0; i < numInputNodes; i++) {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    input_names.emplace_back(inputNodeNameAllocatedStrings.back().get());
  }
  size_t numOutputNodes = session.GetOutputCount();
  for (int i = 0; i < numOutputNodes; i++) {
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    output_names.emplace_back(outputNodeNameAllocatedStrings.back().get());
  }

  std::vector<int64_t> input_shape = {3};
  std::vector<float> input_values_1 = {1.0f, 2.0f, 3.0f};
  std::vector<float> input_values_2 = {4.0f, 5.0f, 6.0f};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor_1 = Ort::Value::CreateTensor<float>(memory_info, input_values_1.data(), input_values_1.size(),
                                                              input_shape.data(), input_shape.size());
  Ort::Value input_tensor_2 = Ort::Value::CreateTensor<float>(memory_info, input_values_2.data(), input_values_2.size(),
                                                              input_shape.data(), input_shape.size());
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(std::move(input_tensor_1));
  input_tensors.emplace_back(std::move(input_tensor_2));

  std::vector<Ort::Value> output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), 2, output_names.data(), 1);

  std::cout << std::fixed;
  for (auto j = 0; j < output_tensors.size(); j++) {
    const float* floatarr = output_tensors[j].GetTensorMutableData<float>();
    for (int i = 0; i < 3; i++) {
      std::cout << floatarr[i] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}