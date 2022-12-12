# Windows C++ sample with OVEP:

1. model-explorer

    This sample application demonstrates how to use components of the experimental C++ API to query for model inputs/outputs and how to run inferrence using OpenVINO Execution Provider for ONNXRT on a model. The source code for this sample is available [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP/Windows/model-explorer).

2. Squeezenet classification sample

    The sample involves presenting an image to the ONNX Runtime (RT), which uses the OpenVINO Execution Provider for ONNXRT to run inference on various Intel hardware devices like Intel CPU, GPU, VPU and more. The sample uses OpenCV for image processing and ONNX Runtime OpenVINO EP for inference. After the sample image is inferred, the terminal will output the predicted label classes in order of their confidence. The source code for this sample is available [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP/Windows/squeezenet_classification).

3. Squeezenet classification sample with IO Buffer feature

    This sample is also doing the same process but with IO Buffer optimization enabled. With IO Buffer interfaces we can avoid any memory copy overhead when plugging OpenVINO™ inference into an existing GPU pipeline. It also enables OpenCL kernels to participate in the pipeline to become native buffer consumers or producers of the OpenVINO™ inference. Refer [here](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API.html) for more details. This sample is for GPUs only. The source code for this sample is available [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP/Windows/squeezenet_classification_io_buffer).

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
