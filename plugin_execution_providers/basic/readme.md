# Basic Plugin Execution Provider
This directory contains a basic example of a custom ONNX Runtime Execution Provider (EP) implemented as a plugin.

## Contents
- `CMakeLists.txt`: Build configuration for the basic plugin EP.
- `src`: Contains source code for the basic plugin EP.
- `android`: Contains example code for setting up and using an Android package.
- `csharp`: Contains example code for setting up and using a C# NuGet package.
- `python`: Contains example code for setting up and using a Python package.
- `gen_mul_model.py`: Reference script used to generate `mul.onnx` models used in usage examples. The model files are checked in.

## Build Instructions
Use CMake to configure and build the project:

```bash
cmake -B ./build -S .
cmake --build ./build
```

The resulting plugin EP library can be registered with ONNX Runtime for inference.

## Usage
Refer to the ONNX Runtime documentation for details on loading and using plugin EPs. This example is intended for plugin EP developers.

## References
- [ONNX Runtime Plugin EP Documentation](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/)
