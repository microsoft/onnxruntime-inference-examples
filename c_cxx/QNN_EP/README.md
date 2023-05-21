## Prerequisites
- Qualcomm ARM Device - e.g. Surface Pro9 5G, Windows Dev Kit 2023
- Visual Studio 2022
- OnnxRuntime ARM Build with QNN support
    - Either pre-compiled download OR 
    - Compiled from onnxruntime source with --arm64 --use_qnn and --qnn_home $(QNN_SDK_ROOT) parameters
        - Python
        - QNN SDK (Qualcomm® Neural Network (NN) Software Development Kit (SDK)) from Qualcomm CreatePoint 
            - Example: Qualcomm_NN_SDK.WIN.2.0 Installer Preview from https://createpoint.qti.qualcomm.com/tools/ - Filter results to 'NN'

## How to run the application
(Windows11/ARM64) Run ```run_qnn_ep_sample.bat``` with path to onnxruntime root directory (for includes) and path to bin directory
```
(release MUST include QNN support)
.\run_qnn_ep_sample.bat %USERPROFILE%\Downloads\onnxruntime-win-arm-x %USERPROFILE%\Downloads\onnxruntime-win-arm-x\lib 

For Local Dev Build
.\run_qnn_ep_sample.bat C:\src\onnxruntime\build\Windows\RelWithDebInfo\RelWithDebInfo
```