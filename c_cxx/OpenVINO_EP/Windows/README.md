# Windows C++ sample with OVEP:

1. model-explorer
2. Squeezenet classification

## How to build

#### Build ONNX Runtime
Open x64 Native Tools Command Prompt for VS 2019.
For running the sample with IO Buffer optimization feature, make sure you set the OpenCL paths. For example if you are setting the path from openvino source build folder, the paths will be like:

```
set OPENCL_LIBS=\path\to\openvino\folder\bin\intel64\Release\OpenCL.lib
set OPENCL_INCS=\path\to\openvino\folder\thirdparty\ocl\clhpp_headers\include
```

```
build.bat --config RelWithDebInfo --use_openvino CPU_FP32 --build_shared_lib --parallel --cmake_extra_defines CMAKE_INSTALL_PREFIX=c:\dev\ort_install --skip_tests
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
cmake .. -A x64 -T host=x64 -Donnxruntime_USE_OPENVINO=ON -DONNXRUNTIME_ROOTDIR=c:\dev\ort_install -DOPENCV_ROOTDIR="path\to\opencv"
```
Choose required opencv path. Skip the opencv flag if you don't want to build squeezenet sample.

To get the squeezenet sample with IO buffer feature enabled, pass opencl paths as well:
```bat
mkdir build && cd build
cmake .. -A x64 -T host=x64 -Donnxruntime_USE_OPENVINO=ON -DONNXRUNTIME_ROOTDIR=c:\dev\ort_install -DOPENCV_ROOTDIR="path\to\opencv -DOPENCL_LIB=path\to\openvino\folder\bin\intel64\Release\ -DOPENCL_INCLUDE=path\to\openvino\folder\thirdparty\ocl\clhpp_headers\include"
```

**Note:**
If you are using the opencv from openvino package, below are the paths:
* For openvino version 2022.1.0, run download_opencv.ps1 in \path\to\openvino\extras\script and the opencv folder will be downloaded at \path\to\openvino\extras.
* For older openvino version, opencv folder is available at openvino directory itself.
* The current cmake files are adjusted with the opencv folders coming along with openvino packages. Plase make sure you are updating the opencv paths according to your custom builds.
For the squeezenet IO buffer sample:
Make sure you are creating the opencl context for the right GPU device in a multi-GPU environment.

Build samples using msbuild either for Debug or Release configuration.

```bat
msbuild onnxruntime_samples.sln /p:Configuration=Debug|Release
```

To run the samples make sure you source openvino variables using setupvars.bat. Also add opencv dll paths to $PATH.
