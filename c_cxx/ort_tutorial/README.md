# C/C++ ORT Tutorial

These C/C++ samples are intended to be run built and run independently.
Please see the README's of the respective samples to get started with the API.

## Building the samples

To build the samples, you will need to have CMake installed and a functional C++ compiler.
To compile the samples, with support for ONNX Runtime and TensorRT RTX EP, you can use the following commands:
```
cmake -B build -S . -DONNX_RUNTIME_PATH=path/to/onnxruntime> -DTRTRTX_RUNTIME_PATH=<path/to/TRTRTX/libs> 
cmake --build build --config Release
```

- `ONNX_RUNTIME_PATH` should be set to the directory containing the ONNX Runtime headers inside `include/` and libraries inside `lib/`.
- (optional)`TRTRTX_RUNTIME_PATH` will make it easier to run TensorRT RTX EP since it ensures all libraries are copied to the executable directory. If this is not used they are required to be on the system path. In addition, TensorRT RTX EP required the CUDA Runtime to be on the system path as well.
