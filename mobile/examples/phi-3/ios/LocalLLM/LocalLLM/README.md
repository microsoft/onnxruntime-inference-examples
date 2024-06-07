# **Local LLM sample application running Phi3-mini on iOS**

## **Steps**

### General prerequisites

See the general prerequisites [here](../../../../../README.md#General-Prerequisites).

For this application, the following prerequisites are preferred:

1. macOS 14+

2. Xcode 15+ (latest Xcode version perferred.)

3. iOS SDK 16.x + (iPhone 14 or iPhone 15 powered by a A16 or A17 preferred)

**Note**: 
  The current Xcode project contains a built .dylib for ORT and ORT GenAI. The following steps `A, B, C` under `step 1.` for building from source for the libraries are optional.
  However if you want to build from source to include the latest updates, please use the `step 1.` as a reference.

### 1. Steps to build from source for ONNX Runtime and Generative AI libraries [Optional]

#### **A. Preparation**

   - Install Python 3.10+

   - Install flatbuffers
     ```
       pip3 install flatbuffers
     ```

   - Install [CMake](https://cmake.org/download/)

#### **B. Compiling ONNX Runtime for iOS**

```bash

git clone https://github.com/microsoft/onnxruntime.git

cd onnxruntime

./build.sh --build_shared_lib --skip_tests --parallel --build_dir ./build_ios --ios --apple_sysroot iphoneos --osx_arch arm64 --apple_deploy_target 16.6 --cmake_generator Xcode --config Release

```

***Notice***

  1. Before compiling, you must ensure that Xcode is configured correctly and set it on the terminal

```bash

sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer 

```

  2. ONNX Runtime needs to be compiled based on different platforms. For iOS, you can compile for arm64 or x86_64 based on needs. If you are running an iOS simulator on an Intel mac, compile for x86_64. Use arm64 for an ARM based mac to run the simulator, and to run on an iPhone.
   
  3. It is recommended to directly use the latest iOS SDK for compilation. Of course, you can also lower the version to be compatible with past SDKs.

#### **C. Compiling Generative AI with ONNX Runtime for iOS**

```bash

git clone https://github.com/microsoft/onnxruntime-genai

cd onnxruntime-genai

python3 build.py --parallel --build_dir ./build_iphoneos --ios --ios_sysroot iphoneos --ios_arch arm64 --ios_deployment_target 16.6 --cmake_generator Xcode

```

#### **D. Copy over latest header files and required .dylibs built from source**

If you build from source and get the latest .dylibs for ORT and ORT GenAI, please copy the .dylibs over to `mobile\examples\phi-3\ios\LocalLLM\LocalLLM\lib` and copy the latest header files over to `mobile\examples\phi-3\ios\LocalLLM\LocalLLM\header` 

Usually the build output path for libonnxruntime.dylib is under `<ORT_PROJECT_ROOT>/build/intermediates/<platform>_<arch>/<build_config>/<build_config-platform>/libonnxruntime.dylib` and the build output path for libonnxruntime-genai.dylib is under `<ORT_GENAI_PROJECT_ROOT>/build/<build_config-platform>/libonnxruntime-genai.dylib`. For example: 
it may look like: `onnxruntime/build/intermediates/iphoneos_arm64/Release/Release-iphoneos/libonnxruntime.1.19.0.dylib`
and similarly `onnxruntime-genai/build/Release/Release-iphoneos/libonnxruntime-genai.dylib`.
The resulting header directory should correctly contains:
`mobile\examples\phi-3\ios\LocalLLM\LocalLLM\lib\libonnxruntime-genai.dylib`
`mobile\examples\phi-3\ios\LocalLLM\LocalLLM\lib\libonnxruntime.1.19.0.dylib`

And the source header files required including:
`<ORT_MAIN_SOURCE_REPO>/onnxruntime/core/session/onnxruntime_c_api.h`,
`<ORT_GENAI_MAIN_SOURCE_REPO>/src/ort_genai.h`,
`<ORT_GENAI_MAIN_SOURCE_REPO>/src/ort_genai_c.h`.

### 2. Create/Open the iOS application in Xcode

The app uses Objective-C/C++ since using Generative AI with ONNX Runtime C++ API, Objective-C has better compatiblility.

### 3. Copy the ONNX quantized INT4 model to the App application project

Download from hf repo: <https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4>

After downloading completes, you need to copy files over to the `Resources` directory in the `Destination` column of `Target-LocalLLM`->`Build Phases`-> `New Copy File Phases` -> `Copy Files`.

Upon app launching, Xcode will automatically copy and install the model files from Resources folder and directly download to the iOS device.

### 4. Run the app and checkout the streaming output token results

**Note**: The current app only sets up with a simple initial prompt question, you can adjust/try your own or refine the UI based on requirements.

***Notice:*** The current Xcode project runs on iOS 16.6, feel free to adjust latest iOS/build for lates iOS versions accordingly.