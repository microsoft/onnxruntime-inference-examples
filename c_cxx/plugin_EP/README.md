# Running Inference with a Plugin EP using C++ API
## Prerequisites
- ONNX Runtime version >= 1.23.0
- A dynamic/shared EP library that exports the functions `CreateEpFactories()` and `ReleaseEpFactory()`.
- ONNX Runtime built as a shared library (e.g., `onnxruntime.dll` on Windows or `libonnxruntime.so` on Linux), since the EP library relies on the public ORT C API (which is ABI-stable) to interact with ONNX Runtime.
- The `onnxruntime_providers_shared.dll` (Windows) or `libonnxruntime_providers_shared.so` (Linux) is also required. When a plugin EP is registered, ONNX Runtime internally calls `LoadPluginOrProviderBridge`, which depends on this shared library to determine whether the EP DLL is a plugin or a provider-bridge.
- If you are using a pre-built ONNX Runtime package, all required libraries (e.g., `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, etc.) are already included.

## Run Inference with explicit OrtEpDevice(s)

Please see `plugin_ep_inference.py` for a full example.
1. Register plugin EP library with ONNX Runtime
   ````python
   onnxruntime.register_execution_provider_library("plugin_ep.so")
   ````
2. Find the OrtEpDevice for that EP
   ````Python
   ep_device = onnxruntime.get_ep_devices()
   for ep_device in ep_devices:
       if ep_device.ep_name == ep_name:
           target_ep_device = ep_device
    ````
3. Append the EP to ORT session option
    ````Python
    sess_options.add_provider_for_devices([target_ep_device], {})
    ````
5. Create ORT session with the EP
    ```Python
    sess = onnxrt.InferenceSession("/path/to/model", sess_options=sess_options)
    ````
6. Run ORT session
   ````Python
   res = sess.run([], {input_name: x})
   ````
7. Unregister plugin EP library
    ```Python
   onnxruntime.unregister_execution_provider_library(ep_registration_name)
   ````


 ## Run Inference with automatic EP selection
 The workflow is the same as above except for step 2 and 3.
 Instead, set the selection policy directly 
 ````Python
 sess_options.set_provider_selection_policy(policy)
 ````
 Available "policy":
 - `onnxruntime.OrtExecutionProviderDevicePolicy_DEFAULT`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_PREFER_CPU`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_PREFER_NPU`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_PREFER_GPU`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER`

 ## Note
 For additional APIs and details on plugin EP usage, see the official documentation:
 https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html#using-a-plugin-ep-library


