# **Local LLM sample application running Phi3-mini on iOS**

## **Steps**

### General prerequisites

See the general prerequisites [here](../../../../../README.md#General-Prerequisites).

**Note**: 
  The current Xcode project contains a built .dylib for ORT and ORT GenAI. The following steps `A, B, C` under `step 1.` for building from source for the libraries are optional.
  However if you want to build from source to include the latest updates, please use the `step 1.` as a reference.

### 1. Steps to build from source for ONNX Runtime and Generative AI libraries [Optional]

#### **A. Preparation**

1. macOS 14+

2. Xcode 15+

3. iOS SDK 16.x + (iPhone 14 or iPhone 15 powered by a A16 or A17 preferred)

4. Install Python 3.10+

5. Install flatbuffers
  ```
    pip3 install flatbuffers
  ```

6. Install [CMake](https://cmake.org/download/)

#### **B. Compiling ONNX Runtime for iOS**

```bash

git clone https://github.com/microsoft/onnxruntime.git

cd onnxruntime

./build.sh --build_shared_lib --ios --skip_tests --parallel --build_dir ./build_ios --ios --apple_sysroot iphoneos --osx_arch arm64 --apple_deploy_target 16.6 --cmake_generator Xcode --config Release

```

***Notice***

  1. Before compiling, you must ensure that Xcode is configured correctly and set it on the terminal

```bash

sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer 

```

  1. ONNX Runtime needs to be compiled based on different platforms. For iOS, you can compile for arm64 or x86_64 based on needs. If you running on ios simulator on an Intel mac, compile for x86_64. And arm64 for an ARM based mac to run the simulator or actual iphone device.
   
  2. It is recommended to directly use the latest iOS SDK for compilation. Of course, you can also lower the version to be compatible with past SDKs.

#### **C. Compiling Generative AI with ONNX Runtime for iOS**

```bash

git clone https://github.com/microsoft/onnxruntime-genai

cd onnxruntime-genai

python3 build.py --parallel --build_dir ./build_iphoneos --ios --ios_sysroot iphoneos --ios_arch arm64 --ios_deployment_target 16.6 --cmake_generator Xcode

```


### 2. Create/Open the iOS application in Xcode

The app uses Objective-C/C++ since using Generative AI with ONNX Runtime C++ API, Objective-C has better compatiblility.

### 3. Copy over latest header files and required .dylibs built from source [Optional]

If you built from source and get the latest .dylibs for ORT and ORT GenAI, please copy the dylibs over to `<PROJECT_ROOT>/lib` and copy the latest header source files over to `<PROJECT_ROOT>/header` .

Source header files required including:
`<ORT_MAIN_SOURCE_REPO>/onnxruntime/core/session/onnxruntime_c_api.h`
`<ORT_GENAI_MAIN_SOURCE_REPO>/src/ort_genai.h`
`<ORT_GENAI_MAIN_SOURCE_REPO>/src/ort_genai_c.h`

### 4. Copy the ONNX quantized INT4 model to the App application project

Download from hf repo: <https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4>

After downloading completes, you need to copy files over to the `Resources` directory in the `Destination` column of `Target-LocalLLM`->`Build Phases`-> `New Copy File Phases` -> `Copy Files`.

### 5. Run the app and checkout the streaming output token results

**Note**: The current app only sets up with a simple initial prompt question, you can adjust/try your own or refine the UI based on requirements.

***Notice:*** The current Xcode project runs on iOS 16.6, feel free to adjust latest iOS/build for lates iOS versions accordingly.