## About
- Builds the sample compiled against the ONNX Runtime and Qualcomm® Neural Network (NN) Software Development Kit (SDK)
- The sample uses the QNN EP but runs on QnnCPU.dll
- The sample can be compiled and run on an Intel/AMD amd64 device or QC or other arm64 device. Does not require NPU (Neural Processing Unit)

## Prerequisites
- Windows 11
- Visual Studio 2022
- OnnxRuntime ARM Build with QNN support
    - Either pre-compiled download OR 
    - Compiled from onnxruntime source with --use_qnn and --qnn_home $(QNN_SDK_ROOT) parameters
        - If running on amd64 default params are fine. For arm64 us the --arm64 option as well
        - Python
        - QNN SDK (Qualcomm® Neural Network (NN) Software Development Kit (SDK)) from Qualcomm CreatePoint 
            - Example: Qualcomm_NN_SDK.WIN.2.0 Installer Preview from https://createpoint.qti.qualcomm.com/tools/ - Filter results to 'NN'

## How to run the application
(Windows11) Run ```run_qnn_ep_sample.bat``` with path to onnxruntime root directory (for includes) and path to bin directory
```
(release MUST include QNN support)
.\run_qnn_ep_sample.bat %USERPROFILE%\Downloads\onnxruntime-win-arm-x %USERPROFILE%\Downloads\onnxruntime-win-arm-x\lib 

For Local Dev Build
.\run_qnn_ep_sample.bat C:\src\onnxruntime\build\Windows\RelWithDebInfo\RelWithDebInfo
```