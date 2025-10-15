# Windows C++ sample with OVEP:

1. model-explorer

   This sample application demonstrates how to use the **ONNX Runtime C++ API** with the OpenVINO Execution Provider (OVEP).  
   It loads an ONNX model, inspects the input/output node names and shapes, creates random input data, and runs inference.  
   The sample is useful for exploring model structure and verifying end-to-end execution with OVEP.   [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP/Windows/model-explorer).

2. Squeezenet classification sample

    The sample involves presenting an image to the ONNX Runtime (RT), which uses the OpenVINO Execution Provider for ONNXRT to run inference on various Intel hardware devices like Intel CPU, GPU and NPU. The sample uses OpenCV for image processing and ONNX Runtime OpenVINO EP for inference. After the sample image is inferred, the terminal will output the predicted label classes in order of their confidence. The source code for this sample is available [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP/Windows/squeezenet_classification).

## How to build

## Prerequisites
1. [The Intel<sup>®</sup> Distribution of OpenVINO toolkit](https://docs.openvino.ai/2025/get-started/install-openvino.html)
2. Use opencv [OpenCV](https://opencv.org/releases/)
3. Use any sample image as input to the sample.
4. Download the latest Squeezenet model from the ONNX Model Zoo.
   This example was adapted from [ONNX Model Zoo](https://github.com/onnx/models). Download the latest version of the [Squeezenet](https://github.com/onnx/models/tree/master/vision/classification/squeezenet) model from here.

## Build ONNX Runtime with OpenVINO on Windows

Make sure you open **x64 Native Tools Command Prompt for VS 2019** before running the following steps.

### 1. Download OpenVINO package
Download the OpenVINO archive package from the official repository:  
[OpenVINO Archive Packages](https://storage.openvinotoolkit.org/repositories/openvino/packages)

Extract the downloaded archive to a directory (e.g., `C:\openvino`).

---

### 2. Set up OpenVINO environment
After extracting, run the following command to set up environment variables:

```cmd
"C:\openvino\setupvars.bat"
```

### 3. Build ONNX Runtime

```
build.bat --config RelWithDebInfo --use_openvino CPU --build_shared_lib --parallel --cmake_extra_defines CMAKE_INSTALL_PREFIX=c:\dev\ort_install --skip_tests
```

By default products of the build on Windows go to build\Windows\config folder. In the case above it would be build\Windows\RelWithDebInfo.
Run the following commands.

```
cd build\Windows\RelWithDebInfo
msbuild INSTALL.vcxproj /p:Configuration=RelWithDebInfo
```

### Build the samples

Open x64 Native Tools Command Prompt for VS 2022, Git clone the sample repo.
```
git clone https://github.com/microsoft/onnxruntime-inference-examples.git
```
Change your current directory to c_cxx\OpenVINO_EP\Windows, then run

```bat
mkdir build && cd build
cmake .. -A x64 -T host=x64 -Donnxruntime_USE_OPENVINO=ON -DONNXRUNTIME_ROOTDIR=c:\dev\ort_install -DOPENCV_ROOTDIR="path\to\opencv"
```
Choose required opencv path. Skip the opencv flag if you don't want to build squeezenet sample.

### Note
To run the samples make sure you source openvino variables using setupvars.bat. 

#### Run the sample

   - To Run the general sample
      (using Intel OpenVINO-EP)
      ```
      run_squeezenet.exe --use_openvino <path_to_onnx_model> <path_to_sample_image> <path_to_labels_file>
      ```
      Example:
      ```
      run_squeezenet.exe --use_openvino squeezenet1.1-7.onnx demo.jpeg synset.txt  (using Intel OpenVINO-EP)
      ```
      (using Default CPU)
      ```
      run_squeezenet.exe --use_cpu <path_to_onnx_model> <path_to_sample_image> <path_to_labels_file>
      ```

## References:

[OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

[Other ONNXRT Reference Samples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx)