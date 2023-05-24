## About
- Builds the sample compiled against the ONNX Runtime built with support for Qualcomm AI Engine Direct SDK (Qualcomm Neural Network (QNN) SDK)
- The sample uses the QNN EP, run with Qnn CPU banckend and HTP backend
- The sample downloads the mobilenetv2 model from Onnx model zoo, and use mobilenetv2_helper.py to quantize the float32 model to QDQ model which is required for HTP backend
- The sample is targeted to run on QC ARM64 device.
- More info on QNN EP - https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

## Prerequisites
- Windows 11
- Visual Studio 2022
- OnnxRuntime ARM Build with QNN support
    - Compiled from onnxruntime source - https://onnxruntime.ai/docs/build/eps.html#QNN

## How to run the application
(Windows11) Run ```run_qnn_ep_sample.bat``` with path to onnxruntime root directory (for includes) and path to bin directory
```
.\run_qnn_ep_sample.bat C:\src\onnxruntime\build\Windows\Release\Release
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