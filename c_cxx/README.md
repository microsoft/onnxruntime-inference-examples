# C/C++ sample applications

These applications demonstrate the ONNX Runtime [C/C++ API](https://onnxruntime.ai/docs/api/c)

|Application|Description|API|Targets|
|-----------|-----------|---|-----------------|
| [Style transfer](fns_candy_style_transfer) | Apply a 'candy' style to any image |C| Windows, Linux|
| [Image classification using Inceptionv3](Snpe_EP) | Classify an image | C++ | Windows, Android |
| [Image classification using Inceptionv4](imagenet) | Classify a batch of images and optionally train with your own data | C++ | Any|
| [Number recognition](MNIST) | Recognize number with a GUI |C++|Windows|
| [Image classification with Squeezenet](OpenVINO_EP/Windows/squeezenet_classification) | Classify individual images |C++|Linux, Windows on Intel hardware|
| [Model explorer](OpenVINO_EP/Windows/model-explorer/)| Explore model inputs and outputs |C++|Linux, Windows|

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
