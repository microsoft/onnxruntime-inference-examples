## About
- Builds the sample compiled against the ONNX Runtime built with support for Qualcomm AI Engine Direct SDK (Qualcomm Neural Network (QNN) SDK)
- The sample uses the QNN EP, run with Qnn CPU banckend and HTP backend
- The sample downloads the mobilenetv2 model from Onnx model zoo, and use mobilenetv2_helper.py to quantize the float32 model to QDQ model which is required for HTP backend
- The sample is targeted to run on QC ARM64 device.
- More info on QNN EP - https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

## Prerequisites
- Windows 11
- Visual Studio 2022
- OnnxRuntime ARM Build with initial QNN support such as ONNX Runtime (ORT) 1.16+ (onnxruntime-win-arm64-1.16.0.zip)
  - ORT Drop DOES NOT INCLUDE QNN so QNN binaries must be copied from QC SDK. E.g
    - robocopy C:\Qualcomm\AIStack\QNN\2.10.40.4\lib\aarch64-windows-msvc %USERPROFILE%\Downloads\onnxruntime-win-arm64-1.16.0\lib
    - copy C:\Qualcomm\AIStack\QNN\2.10.40.4\lib\hexagon-v68\unsigned\libQnnHtpV68Skel.so %USERPROFILE%\Downloads\onnxruntime-win-arm64-1.16.0\lib
- (OR) Compiled from onnxruntime source with QNN support - https://onnxruntime.ai/docs/build/eps.html#qnn

## How to run the application
(Windows11) Run ```run_qnn_ep_sample.bat``` with path to onnxruntime root directory (for includes) and path to bin directory
```
run_qnn_ep_sample.bat PATH_TO_ORT_ROOT_WITH_INCLUDE_FOLDER PATH_TO_ORT_BINARIES_WITH_QNN
Example (Drop): run_qnn_ep_sample.bat %USERPROFILE%\Downloads\onnxruntime-win-arm64-1.16.0 %USERPROFILE%\Downloads\onnxruntime-win-arm64-1.16.0\lib
Example (Src): run_qnn_ep_sample.bat C:\src\onnxruntime C:\src\onnxruntime\build\Windows\RelWithDebInfo\RelWithDebInfo
```

## Example run result
```
...
REM run with QNN CPU backend
qnn_ep_sample.exe --cpu kitten_input.raw

Result:
position=281, classification=n02123045 tabby, tabby cat, probability=13.663173

REM run with QNN HTP backend
qnn_ep_sample.exe --htp kitten_input.raw

Result:
position=281, classification=n02123045 tabby, tabby cat, probability=13.637316
...
```