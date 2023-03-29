# C/C++ sample applications for demonstrating onnxruntime usage

1. (Windows and Linux) fns_candy_style_transfer: A C application that uses the FNS-Candy style transfer model to re-style images. It is written purely in C, no C++.
2. (Windows only) MNIST: A windows GUI application for doing handwriting recognition
3. (Windows only) imagenet: An end-to-end sample for the [ImageNet Large Scale Visual Recognition Challenge 2012](http://www.image-net.org/challenges/LSVRC/2012/) - requires ATL libraries to be installed as a part of the VS Studio installation.
4. model-explorer: A commandline C++ application that generates random data and performs model inference. A second C++ application demonstrates how to perform batch processing. (TODO: Add CI build for it)
5. OpenVINO_EP: Using OpenVino execution provider on the squeezenet model (TODO: Add CI build for it)
6. opschema_lib_use (TODO: Add CI build for it)

## How to build

Note: These build instructions are for the Windows examples only.

### Prerequisites

1. Visual Studio 2019 or 2022
2. cmake(version >=3.13)
3. (optional) [libpng 1.6](https://libpng.sourceforge.io/)

## Install ONNX Runtime

### Option 1: Download a release package

* Download an onnxruntime package (onnxruntime-win-x64-\*\*\*.zip) from https://github.com/microsoft/onnxruntime/releases/.

* Unzip to any folder

* Set ONNXRUNTIME_ROOTDIR to the root of the folder as an absolute path

### Option 2: build from source

If you'd like to build from source, the full build instructions are [here](https://www.onnxruntime.ai/docs/build/).

Please note you need to include the "--build_shared_lib" flag in your build command. 

e.g. from Developer Command Prompt or Developer PowerShell for the Visual Studio version you are going to use,
build ONNX Runtime at least these options:

```bat
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel
```

By default the build output will go in the `.\build\Windows\<config>` folder, and
"C:\Program Files\onnxruntime" will be the installation location.

You can override the installation location by specifying CMAKE_INSTALL_PREFIX via the cmake_extra_defines parameter. For example:

```bat
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel --cmake_extra_defines CMAKE_INSTALL_PREFIX=D:\onnxruntime
```

Run the below command to install onnxruntime. If installing to "C:\Program Files\onnxruntime" you will need to run the command from an elevated command prompt.

```bat
cmake --install .\build\Windows\RelWithDebInfo --config RelWithDebInfo
```

### Build the samples

The location of the ONNX Runtime header files is specified by the ONNXRUNTIME_ROOTDIR environment variable.

Open Developer Command Prompt or Developer PowerShell for the Visual Studio version you are going to use, change your current directory to samples\c_cxx, and run the following command:

```bat
mkdir build && cd build
cmake .. -A x64 -T host=x64 -DONNXRUNTIME_ROOTDIR=D:\onnxruntime -DLIBPNG_ROOTDIR=C:\path\to\your\libpng\binary
```

* You can omit the "-DLIBPNG_ROOTDIR=..." argument if you don't have the libpng library.
* You can omit "-DONNXRUNTIME_ROOTDIR=..." if you installed to "C:\Program Files\onnxruntime", otherwise adjust the value to match your ONNX Runtime install location.
* Append "-Donnxruntime_USE_CUDA=ON" or "-Donnxruntime_USE_DML=ON" to the last command args if your onnxruntime binary was built with CUDA or DirectML support respectively.

You can then open and build the solution using Visual Studio:

```bat
devenv onnxruntime_samples.sln
```

or build it from the command line using msbuild

```bat
msbuild onnxruntime_samples.sln /p:Configuration=Debug|Release
```
