# Running Inference with a Plugin EP using C++ API
## Prerequisites
- ONNX Runtime version >= 1.23.0
- A dynamic/shared EP library that exports the functions `CreateEpFactories()` and `ReleaseEpFactory()`.
- ONNX Runtime built as a shared library (e.g., `onnxruntime.dll` on Windows or `libonnxruntime.so` on Linux), since the EP library relies on the public ORT C API (which is ABI-stable) to interact with ONNX Runtime.
- The `onnxruntime_providers_shared.dll` (Windows) or `libonnxruntime_providers_shared.so` (Linux) is also required. When a plugin EP is registered, ONNX Runtime internally calls `LoadPluginOrProviderBridge`, which depends on this shared library to determine whether the EP DLL is a plugin or a provider-bridge.
- If you are using a pre-built ONNX Runtime package, all required libraries (e.g., `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, etc.) are already included.

## Run Inference with explicit OrtEpDevice(s)

Please see `plugin_ep_inference.cc` for a full example.
1. Register plugin EP library with ONNX Runtime
   ````c++
   env.RegisterExecutionProviderLibrary(
       "plugin_ep",              // Registration name can be anything the application chooses.
       ORT_TSTR("plugin_ep.so")  // Path to the plugin EP library.
   );
   ````
2. Find the OrtEpDevice for that plugin EP
   ````c++
   // Find the Ort::EpDevice for ep_name
    std::vector<Ort::ConstEpDevice> selected_ep_devices = {};
    for (Ort::ConstEpDevice ep_device : ep_devices) {
      if (std::string(ep_device.EpName()) == ep_name) {
        selected_ep_devices.push_back(ep_device);
        break;
      }
    }
    ````
3. Append the EP to ORT session option
    ````c++
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(env, selected_ep_devices, ep_options);
    ````
5. Create ORT session with the EP
    ````c++
    Ort::Session session(env, ORT_TSTR("path\to\model"), session_options);
    ````
6. Run ORT session
   ````c++
    auto output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
   ````
7. Unregister plugin EP library
   ````c++
   env.UnregisterExecutionProviderLibrary(lib_registration_name);
   ````


 ## Run Inference with automatic EP selection
 The workflow is the same as above except for step 2 and 3.
 Instead, set the selection policy directly 
 ````Python
 session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);
 ````
 Available "policy":
 - `OrtExecutionProviderDevicePolicy_DEFAULT`
 - `OrtExecutionProviderDevicePolicy_PREFER_CPU`
 - `OrtExecutionProviderDevicePolicy_PREFER_NPU`
 - `OrtExecutionProviderDevicePolicy_PREFER_GPU`
 - `OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE`
 - `OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY`
 - `OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER`

 ## Note
 For additional APIs and details on plugin EP usage, see the official documentation:
 https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html#using-a-plugin-ep-library


