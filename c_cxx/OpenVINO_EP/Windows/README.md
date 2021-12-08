# Windows C++ sample with OVEP:

1. model-explorer
2. Squeezenet classification

## How to build

#### Build ONNX Runtime
Open x64 Native Tools Command Prompt for VS 2019.
```
build.bat --config RelWithDebInfo --use_openvino CPU_FP32 --build_shared_lib --parallel --cmake_extra_defines CMAKE_INSTALL_PREFIX=c:\dev\ort_install
```

By default products of the build on Windows go to build\Windows\config folder. In the case above it would be build\Windows\RelWithDebInfo.
Run the following commands.

```
cd build\Windows\RelWithDebInfo
msbuild INSTALL.vcxproj /p:Configuration=RelWithDebInfo
```

#### Build the samples

Open x64 Native Tools Command Prompt for VS 2019, Git clone the sample repo.
```
git clone https://github.com/microsoft/onnxruntime-inference-examples.git
```
Change your current directory to c_cxx\OpenVINO_EP\Windows, then run
```bat
mkdir build && cd build
cmake .. -A x64 -T host=x64 -Donnxruntime_USE_OPENVINO=ON -DONNXRUNTIME_ROOTDIR=c:\dev\ort_install -DOPENCV_ROOTDIR="C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv"
```
Choose required opencv path. Skip the opencv flag if you don't want to build squeezenet sample.
Build samples using msbuild either for Debug or Release configuration.

```bat
msbuild onnxruntime_samples.sln /p:Configuration=Debug|Release
```

To run the samples make sure you source openvino variables using setupvars.bat.
