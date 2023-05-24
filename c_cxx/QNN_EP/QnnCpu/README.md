## About
- Builds the sample compiled against the ONNX Runtime built with support for Qualcomm AI Engine Direct SDK (Qualcomm Neural Network (QNN) SDK)
- The sample uses the QNN EP but runs on QnnCPU.dll (QC)
- The sample can be compiled and run on an Intel/AMD AMD64 device or QC or other ARM64 device. Does not require a NPU (Neural Processing Unit)
- More info on QNN EP - https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

## Prerequisites
- Windows 11
- Visual Studio 2022
- OnnxRuntime ARM Build with QNN support 
    - Either pre-compiled download such as ORT 1.15+ OR 
    - Compiled from onnxruntime source - https://onnxruntime.ai/docs/build/eps.html#QNN

## How to run the application
(Windows11) Run ```run_qnn_ep_sample.bat``` with path to onnxruntime root directory (for includes) and path to bin directory
```
(release MUST include QNN support)
.\run_qnn_ep_sample.bat %USERPROFILE%\Downloads\onnxruntime-win-ARCH-x %USERPROFILE%\Downloads\onnxruntime-win-ARCH-x\lib 

For Local Dev Build
.\run_qnn_ep_sample.bat C:\src\onnxruntime\build\Windows\RelWithDebInfo\RelWithDebInfo
```