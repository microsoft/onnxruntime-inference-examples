# Python API Samples

This directory contains sample scripts demonstrating various ONNX Runtime Python API features:

- `getting_started.py`  
  Introduces the basics of exporting a simple PyTorch model to ONNX, running inference with ONNX Runtime, and handling inputs/outputs as NumPy arrays.

- `compile_api.py`  
  Shows how to programmatically compile an ONNX model for a specific execution provider (e.g., TensorRT RTX) to an [EP context](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) ONNX. The sample measures model load and compile times to demonstrate performance improvements and has the option to specify an input model.
  - For `NvTensorRTRTXExecutionProvider` try adding the provider option for a runtime cache (`-p NvTensorRTRTXExecutionProvider -popt "nv_runtime_cache_path=./cache"`) which will further increase the load speed of a compiled model.

- `device_bindings.py`  
  Demonstrates advanced device bindings, including running ONNX models on CPU or GPU, using ONNX Runtime's `OrtValue` for device memory, and direct inference with PyTorch tensors on the selected device. It also demonstrates how to interact with ORT using dlpack.

Each sample is self-contained and includes comments explaining the main concepts.

### Setup 

Besides installing the ONNX Runtime package there are some other dependencies for the samples to work correctly. 
Please pick your selected [onnxruntime package](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime) manually.
```
pip install -r requirements.txt
# to install ORT GPU with required cuda dependencies
pip install onnxruntime-gpu[cuda,cudnn]
```