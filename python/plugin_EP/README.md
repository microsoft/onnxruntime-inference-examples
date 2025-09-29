# Running Inference with a Plugin EP
## Prerequisites
- A dynamic/shared EP library that exports the functions `CreateEpFactories()` and `ReleaseEpFactory()`.
- ONNX Runtime built as a shared library (e.g., `onnxruntime.dll` on Windows or `libonnxruntime.so` on Linux), since the EP library relies on the public ORT C API (which is ABI-stable) to interact with ONNX Runtime. 
