# **Local LLM sample application running Phi3-mini on iOS**

## **Steps**

### **A. Preparation**

1. macOS 14+

2. Xcode 15+

3. iOS SDK 17.x (iPhone 14 or iPhone 15 powered by a A16 or A17)

4. Install Python 3.10+ (Conda is recommended)

5. Install the Python library - python-flatbuffers

6. Install CMake

### **B. Compiling ONNX Runtime for iOS**

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

  2. ONNX Runtime needs to be compiled based on different platforms. For iOS, you can compile based on arm64 / x86_64

  3. It is recommended to directly use the latest iOS SDK for compilation. Of course, you can also lower the version to be compatible with past SDKs.

### **C. Compiling Generative AI with ONNX Runtime for iOS**

```bash

git clone https://github.com/microsoft/onnxruntime-genai

cd onnxruntime-genai

python3 build.py --parallel --build_dir ./build_iphoneos --ios --ios_sysroot iphoneos --ios_arch arm64 --ios_deployment_target 17.4 --cmake_generator Xcode

```

### **D. Create/Open the iOS application in Xcode**

The app uses Objective-C/C++ since using Generative AI with ONNX Runtime C++ API, Objective-C has better compatiblility.

### **E. Copy the ONNX quantized INT4 model to the App application project**

Download from hf repo: <https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4>

After downloading completes, you need to copy files over to the `Resources` directory in the `Destination` column of `Target-LocalLLM`->`Build Phases`-> `New Copy File Phases` -> `Copy Files`.

### **F. Run the app and checkout the streaming output token results**

**Note**: The current app only sets up with a simple initial prompt question, you can adjust/try your own or refine the UI based on requirements.

***Notice:*** The current Xcode project runs on iOS 16.6, feel free to adjust latest iOS/build for lates iOS versions accordingly.