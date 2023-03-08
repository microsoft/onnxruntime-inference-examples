# C/C++ sample applications for demonstrating onnxruntime usage

1. (Windows and Linux) fns_candy_style_transfer: A C application that uses the FNS-Candy style transfer model to re-style images. It is written purely in C, no C++.
2. (Windows only) MNIST: A windows GUI application for doing handwriting recognition
3. (Windows only) imagenet: An end-to-end sample for the [ImageNet Large Scale Visual Recognition Challenge 2012](http://www.image-net.org/challenges/LSVRC/2012/) - requires ATL libraries to be installed as a part of the VS Studio installation.
4. model-explorer: A commandline C++ application that generates random data and performs model inference. A second C++ application demonstrates how to perform batch processing. (TODO: Add CI build for it)
5. OpenVINO_EP: Using OpenVino execution provider on the squeezenet model (TODO: Add CI build for it)
6. opschema_lib_use (TODO: Add CI build for it)

## How to build

### Prerequisites

1. Visual Studio 2015/2017/2019
2. cmake(version >=3.13)
3. (optional) [libpng 1.6](https://libpng.sourceforge.io/)

### Install ONNX Runtime

#### Option 1: Download release version

Download an ONNX Runtime release from https://github.com/microsoft/onnxruntime/releases/. 

For example, you may download onnxruntime-win-x64-\*\*\*.zip and unzip it to any folder.

#### Option 2: Build from source

If you'd like to build it by yourself, [build instructions are here](https://www.onnxruntime.ai/docs/build/). Please note you need to add the "--build_shared_lib" flag to your build command. Like this:

Open Developer Command Prompt for Visual Studio version you are going to use. This will setup necessary environment for the compiler and other things to be found.

```bat
build.bat --config RelWithDebInfo --build_shared_lib --parallel 
```

By default this will build a project with "C:\Program Files (x86)\onnxruntime" install destination. This is a protected folder on Windows. If you do not want to run installation with elevated privileges you will need to override the default installation location by passing extra CMake arguments. For example:

```bat
build.bat --config RelWithDebInfo --build_dir .\build  --build_shared_lib --parallel  --cmake_extra_defines CMAKE_INSTALL_PREFIX=c:\dev\ort_install
```

By default products of the build on Windows go to .\build\Windows\<config> folder. In the case above it would be .\build\RelWithDebInfo since the build folder is mentioned explicitly.

If you did not specify alternative installation location above you would need to open an elevated command prompt to install onnxruntime.

Run the following commands.

```bat
cmake --install .\build\RelWithDebInfo --config RelWithDebInfo
```

### Build the samples

The location of the ONNX Runtime header files is specified by the ONNXRUNTIME_ROOTDIR environment variable.

Open Developer Command Prompt for Visual Studio version you are going to use, change your current directory to samples\c_cxx, then run

```bat
mkdir build && cd build
cmake .. -A x64 -T host=x64 -DLIBPNG_ROOTDIR=C:\path\to\your\libpng\binary -DONNXRUNTIME_ROOTDIR=c:\dev\ort_install
```

You may omit the "-DLIBPNG_ROOTDIR=..." argument if you don't have the libpng library.
You may omit "-DONNXRUNTIME_ROOTDIR=..." if you installed to a default location.

You may append "-Donnxruntime_USE_CUDA=ON" or "-Donnxruntime_USE_DML=ON" to the last command args if your onnxruntime binary was built with CUDA or DirectML support respectively.

You can then either open the solution in a Visual Studio and build it from there

```bat
devenv onnxruntime_samples.sln
```

Or build it using msbuild

```bat
msbuild onnxruntime_samples.sln /p:Configuration=Debug|Release
cmake --install .\build\Debug|Release --config Debug
```

To run the samples make sure that your Install Folder Bin is in the path so your sample executable can find onnxruntime dll and libpng if you used it.

