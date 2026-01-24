# Plugin TensorRT EP

This repo contains:
- The plugin TRT EP implementation
- How to build plugin TRT EP
- How to build python wheel for plugin TRT EP
- How to run inference with plugin TRT EP using python API

Plugin TRT EP is migrated from the original TRT EP and provides the implementations of `OrtEpFactory`, `OrtEp`, `OrtNodeComputeInfo`, `OrtDataTransferImpl` ... that are required for a plugin EP to be able to interact with ONNX Runtime via the EP ABI (introduced in ORT 1.23.0).

## How to build (on Windows) ##
````bash
mkdir build;cd build
````
````bash
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug -DTENSORRT_HOME=C:/folder/to/trt -DORT_HOME=C:/folder/to/ort
````
````bash
cmake --build ./ --config Debug
`````

If the build succeeds, you will see the TRT EP DLL being generated at:
```
C:\repos\onnxruntime-inference-examples\plugin_execution_providers\tensorrt\build> ls .\Debug

TensorRTEp.dll
```


Note: The `ORT_HOME` should contain the `include` and `lib` folder as below
```
C:\folder\to\ort
      |
      | ----- lib
      |          | ----- onnxruntime.dll
      |          | ----- onnxruntime.lib
      |          | ----- onnxruntime.pdb
      |          ...
      |
      | ---- include
      |          | ----- onnxruntime_c_api.h
      |          | ----- onnxruntime_ep_c_api.h
      |          | ----- onnxruntime_cxx_api.h
      |          | ----- onnxruntime_cxx_inline_api.h
      |          ...
```
## How to build python wheel (on Windows) ##
```
setup.py bdist_wheel
```
Once it's done, you will see the wheel file at:
```
C:\repos\onnxruntime-inference-examples\plugin_execution_providers\tensorrt> ls .\dist

plugin_trt_ep-0.1.0-cp312-cp312-win_amd64.whl
```