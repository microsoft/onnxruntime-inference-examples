# Running Inference with a Plugin EP
## Prerequisites
- A dynamic/shared EP library that exports the functions `CreateEpFactories()` and `ReleaseEpFactory()`.
- ONNX Runtime built as a shared library (e.g., `onnxruntime.dll` on Windows or `libonnxruntime.so` on Linux), since the EP library relies on the public ORT C API (which is ABI-stable) to interact with ONNX Runtime. 

## Run Inference with explicit OrtEpDevice(s)

Please see `plugin_ep_inference.py` for details
1. Register plugin EP library with ONNX Runtime via `onnxruntime.register_execution_provider_library()`
2. Find the OrtEpDevice for that ep name via `onnxruntime.get_ep_devices()`
3. Append the ep to ORT session option via `sess_options.add_provider_for_devices`
4. Create ORT session with the ep
5. Run ORT session
6. Unregister plugin EP library via `onnxruntime.unregister_execution_provider_library()`


 ## Run Inference with automatic EP selection
 The workflow is the same as above except #2 and #3 step and should be replaced with `sess_options.set_provider_selection_policy(policy)`,
 "policy" could be:
 - `onnxruntime.OrtExecutionProviderDevicePolicy_DEFAULT`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_PREFER_CPU`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_PREFER_NPU`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_PREFER_GPU`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY`
 - `onnxruntime.OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER`

 
