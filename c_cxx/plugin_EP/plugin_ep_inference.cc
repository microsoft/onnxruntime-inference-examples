#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <gsl/gsl>

int RunInference() { 
  const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  Ort::Env env;

  // Registration name can be anything the application chooses
  const char* lib_registration_name = "TensorRTEp";

  // Register plugin EP library with ONNX Runtime.
  env.RegisterExecutionProviderLibrary(
      lib_registration_name,      // Registration name can be anything the application chooses.
      ORT_TSTR("TensorRTEp.dll")  // Path to the plugin EP library.
  );

  // Unregister the library using the application-specified registration name.
  // Must only unregister a library after all sessions that use the library have been released.
  auto unregister_plugin_eps_at_scope_exit = gsl::finally([&]() { 
    env.UnregisterExecutionProviderLibrary(lib_registration_name);
  });

  {
    std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();
    // EP name should match the name assigned by the EP factory when creating the EP (i.e., in the implementation of OrtEP::CreateEp())
    std::string ep_name = lib_registration_name;

    // Find the Ort::EpDevice for "TensorRTEp".
    std::vector<Ort::ConstEpDevice> selected_ep_devices = {};
    for (Ort::ConstEpDevice ep_device : ep_devices) {
      if (std::string(ep_device.EpName()) == ep_name) {
        selected_ep_devices.push_back(ep_device);
        break;
      }
    }

    if (selected_ep_devices[0] == nullptr) {
      // Did not find EP. Report application error ...
      std::cerr << "Did not find EP: " << ep_name << std::endl;
      return -1;
    }

    std::unordered_map<std::string, std::string> ep_options;  // Optional EP options.
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(env, selected_ep_devices, ep_options);

    Ort::Session session(env, ORT_TSTR("mul_1.onnx"), session_options);

    // Get default ORT allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input name
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator); // Keep the smart pointer alive to avoid dangling pointer
    const char* input_name = input_name_ptr.get();

    // Input data
    std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Input shape: (3, 2)
    std::vector<int64_t> input_shape{3, 2};

    // Create tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_values.data(), input_values.size(),
                                                              input_shape.data(), input_shape.size());

    // Get output name
    Ort::AllocatedStringPtr output_name_ptr =
        session.GetOutputNameAllocated(0, allocator);  // Keep the smart pointer alive to avoid dangling pointer
    const char* output_name = output_name_ptr.get();

    // Run session
    std::vector<const char*> input_names{input_name};
    std::vector<const char*> output_names{output_name};

    auto output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Extract output
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < 6; i++) {
      std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;

    // Expected output: [[1,4],[9,16],[25,36]]
  }

  return 0;
}

int main(int argc, char* argv[]) {
  return RunInference();
}

// Note:
// The mul_1.onnx can be found here:
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/mul_1.onnx
