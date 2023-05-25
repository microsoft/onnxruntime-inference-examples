## About
- Builds the sample compiled against the ONNX Runtime built with support for Qualcomm AI Engine Direct SDK (Qualcomm Neural Network (QNN) SDK)
- The sample uses the QNN EP, run with Qnn CPU banckend and HTP backend
- The sample downloads the mobilenetv2 model from Onnx model zoo, and use mobilenetv2_helper.py to quantize the float32 model to QDQ model which is required for HTP backend
- The sample is targeted to run on QC ARM64 device.
- More info on QNN EP - https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

## Prerequisites
- Windows 11
- Visual Studio 2022
- OnnxRuntime ARM Build with initial QNN support such as ONNX Runtime (ORT) Microsoft.ML.OnnxRuntime.QNN 1.15+ 
  - Download from https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.QNN and unzip
  - ORT Drop DOES NOT INCLUDE QNN so QNN binaries must be copied from QC SDK. E.g
    - robocopy C:\Qualcomm\AIStack\QNN\2.10.40.4\lib\aarch64-windows-msvc %USERPROFILE%\Downloads\microsoft.ml.onnxruntime.qnn.1.15.0\runtimes\win-arm64\native
    - copy C:\Qualcomm\AIStack\QNN\2.10.40.4\lib\hexagon-v68\unsigned\libQnnHtpV68Skel.so %USERPROFILE%\Downloads\microsoft.ml.onnxruntime.qnn.1.15.0\runtimes\win-arm64\native
- (OR) Compiled from onnxruntime source with QNN support - https://onnxruntime.ai/docs/build/eps.html#qnn

## How to run the application
(Windows11) Run ```run_qnn_ep_sample.bat``` with path to onnxruntime root directory (for includes) and path to bin directory
```
run_qnn_ep_sample.bat PATH_TO_ORT_ROOT_WITH_INCLUDE_FOLDER PATH_TO_ORT_BINARIES_WITH_QNN
Example (Drop): run_qnn_ep_sample.bat %USERPROFILE%\Downloads\microsoft.ml.onnxruntime.qnn.1.15.0\build\native %USERPROFILE%\Downloads\microsoft.ml.onnxruntime.qnn.1.15.0\runtimes\win-arm64\native
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