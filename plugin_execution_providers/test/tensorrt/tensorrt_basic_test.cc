#include <gtest/gtest.h>
#include <string>
#include <gsl/gsl>

#include "onnxruntime_cxx_api.h"
#include "test_trt_ep_utils.h"
#include "path_string.h"

namespace test {
namespace trt_ep {

// char type for filesystem paths
using PathChar = ORTCHAR_T;
// string type for filesystem paths
using PathString = std::basic_string<PathChar>;

template <typename T>   
void VerifyOutptus(const std::vector<Ort::Value>& fetches,
                   const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  const Ort::Value& actual_output = fetches[0];
  Ort::TensorTypeAndShapeInfo type_shape_info = actual_output.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType element_type = type_shape_info.GetElementType();
  auto shape = type_shape_info.GetShape();

  ASSERT_EQ(element_type, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_EQ(shape, expected_dims);

  size_t element_cnt = type_shape_info.GetElementCount();
  const T* actual_values = actual_output.GetTensorData<T>();

  ASSERT_EQ(element_cnt, expected_values.size());

  for (size_t i = 0; i != element_cnt; ++i) {
    ASSERT_EQ(actual_values[i], expected_values[i]);
  }
}

static OrtStatus* CreateOrtSession(Ort::Env& env,
                                   PathString model_name,
                                   std::string ep_name,
                                   OrtSession** session) {
  const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  {
    std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();

    // Find the Ort::EpDevice for "TensorRTEp".
    std::vector<Ort::ConstEpDevice> selected_ep_devices = {};
    for (Ort::ConstEpDevice ep_device : ep_devices) {
      // EP name should match the name assigned by the EP factory when creating the EP (i.e., in the implementation of
      // OrtEP::CreateEp())
      if (std::string(ep_device.EpName()) == ep_name) {
        selected_ep_devices.push_back(ep_device);
        break;
      }
    }

    if (selected_ep_devices[0] == nullptr) {
      // Did not find EP. Report application error ...
      std::string message = "Did not find EP: " + ep_name;
      return ort_api->CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    std::unordered_map<std::string, std::string> ep_options;  // Optional EP options.
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(env, selected_ep_devices, ep_options);

    Ort::Session ort_session(env, model_name.c_str(), session_options);
    *session = ort_session.release();
  }

  return nullptr;
}

static OrtStatus* RunInference(Ort::Session& session,
                               std::vector<Ort::Value>& outputs) {
  // Get default ORT allocator
  Ort::AllocatorWithDefaultOptions allocator;

  RETURN_IF_NOT(session.GetInputCount() == 3);

  // Get input names
  Ort::AllocatedStringPtr input_name_ptr =
      session.GetInputNameAllocated(0, allocator);  // Keep the smart pointer alive to avoid dangling pointer
  const char* input_name = input_name_ptr.get();

  Ort::AllocatedStringPtr input_name2_ptr = session.GetInputNameAllocated(1, allocator);
  const char* input_name2 = input_name2_ptr.get();

  Ort::AllocatedStringPtr input_name3_ptr = session.GetInputNameAllocated(2, allocator);
  const char* input_name3 = input_name3_ptr.get();

  // Input data.
  std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Input shape: (1, 3, 2)
  std::vector<int64_t> input_shape{1, 3, 2};

  // Create tensor
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create input data as an OrtValue.
  // Make input2 data and input3 data same as input1 data.
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_values.data(), input_values.size(),
                                                            input_shape.data(), input_shape.size());
  Ort::Value input2_tensor = Ort::Value::CreateTensor<float>(memory_info, input_values.data(), input_values.size(),
                                                             input_shape.data(), input_shape.size());
  Ort::Value input3_tensor = Ort::Value::CreateTensor<float>(memory_info, input_values.data(), input_values.size(),
                                                             input_shape.data(), input_shape.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.reserve(3);
  input_tensors.push_back(std::move(input_tensor));
  input_tensors.push_back(std::move(input2_tensor));
  input_tensors.push_back(std::move(input3_tensor));

  // Get output name
  Ort::AllocatedStringPtr output_name_ptr =
      session.GetOutputNameAllocated(0, allocator);  // Keep the smart pointer alive to avoid dangling pointer
  const char* output_name = output_name_ptr.get();

  // Run session
  std::vector<const char*> input_names{input_name, input_name2, input_name3};
  std::vector<const char*> output_names{output_name};

  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                                    input_tensors.size(), output_names.data(), 1);
  outputs = std::move(output_tensors);

  return nullptr;
}



TEST(TensorrtExecutionProviderTest, CreateSessionAndRunInference) {
  Ort::Env env;
  std::string lib_registration_name = "TensorRTEp";
  std::string& ep_name = lib_registration_name;
  PathString lib_path = ORT_TSTR("TensorRTEp.dll");

  // Register plugin TRT EP library with ONNX Runtime.
  env.RegisterExecutionProviderLibrary(
      lib_registration_name.c_str(),  // Registration name can be anything the application chooses.
      lib_path                        // Path to the plugin TRT EP library.
  );

  // Unregister the library using the application-specified registration name.
  // Must only unregister a library after all sessions that use the library have been released.
  auto unregister_plugin_eps_at_scope_exit =
      gsl::finally([&]() { env.UnregisterExecutionProviderLibrary(lib_registration_name.c_str()); });


  std::string model_name = "basic_model_for_test.onnx";
  std::string graph_name = "basic_model";
  std::vector<int64_t> dims = {1, 3, 2};
  CreateBaseModel(model_name, graph_name, dims);

  OrtSession* session = nullptr;
  ASSERT_EQ(CreateOrtSession(env, ToPathString(model_name), ep_name, &session), nullptr);
  ASSERT_NE(session, nullptr);
  Ort::Session ort_session{session};

  std::vector<Ort::Value> output_tensors;
  ASSERT_EQ(RunInference(ort_session, output_tensors), nullptr);

  // Extract output data
  float* output_data = output_tensors.front().GetTensorMutableData<float>();

  std::cout << "Output:" << std::endl;
  for (int i = 0; i < 6; i++) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;

  std::vector<float> expected_values = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
  std::vector<int64_t> expected_shape{1, 3, 2};
  VerifyOutptus(output_tensors, expected_shape, expected_values);


}

}  // namespace trt_ep
}  // namespace test