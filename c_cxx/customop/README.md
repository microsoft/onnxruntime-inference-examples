This directory contains one C/C++ sample for demonstrating onnxruntime custom operators:

## Prerequisites
1. download onnxruntime binaries (onnxruntime-linux-xx.tgz or onnxruntime-win-x64-xx.zip) from [onnxruntime release site](https://github.com/microsoft/onnxruntime/releases)
2. cmake(version >=3.13)
3. onnx

## How to build and run the sample
1. Run python kenerl.py to generate the onnx model file which contains the custom op.
2. Unzip the onnxruntime binaries to any folder. The folder you unzip it to will be your ONNXRUNTIME_ROOTDIR path.
3. Open a terminal and change your current directory to samples/c_cxx/customop, and run the below command.
   - mkdir build && cd build
   - cmake .. -DONNXRUNTIME_ROOTDIR=/path/to/your/onnxruntime
   - cmake --build . --config Release
4. Add the path of the ONNXRUNTIME_ROOTDIR in `LD_LIBRARY_PATH` on Linux or `PATH` on Windows.
5. Copy the model generated in step 1 and Run the executable file in the build folder.
