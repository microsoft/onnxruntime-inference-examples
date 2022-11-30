# Image classification with Squeezenet in CPP using OpenVINO Execution Provider:

1. The image classification uses a Squeezenet Deep Learning ONNX Model from the ONNX Model Zoo. Currently this sample is only supported on Linux platforms.

2. The sample involves presenting an image to the ONNX Runtime (RT), which uses the OpenVINO Execution Provider for ONNX RT to run inference on various Intel hardware devices like Intel CPU, GPU, VPU and more. The sample uses OpenCV for image processing and ONNX Runtime OpenVINO EP for inference. After the sample image is inferred, the terminal will output the predicted label classes in order of their confidence.

The source code for this sample is available [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP/Linux/squeezenet_classification/squeezenet_cpp_app.cpp).

3. There is one more sample [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP/Linux/squeezenet_classification/squeezenet_cpp_app_io.cpp) with IO Buffer optimization enabled. With IO Buffer interfaces we can avoid any memory copy overhead when plugging OpenVINO™ inference into an existing GPU pipeline. It also enables OpenCL kernels to participate in the pipeline to become native buffer consumers or producers of the OpenVINO™ inference. Refer [here](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API.html) for more details. This sample is for GPUs only.

# How to build

## Prerequisites
1. [The Intel<sup>®</sup> Distribution of OpenVINO toolkit](https://docs.openvinotoolkit.org/latest/index.html)
2. Use opencv (use the same opencv package that comes builtin with Intel<sup>®</sup> Distribution of OpenVINO toolkit)
3. Use opencl for IO buffer sample (squeezenet_cpp_app_io.cpp).
4. Use any sample image as input to the sample.
5. Download the latest Squeezenet model from the ONNX Model Zoo.
   This example was adapted from [ONNX Model Zoo](https://github.com/onnx/models).Download the latest version of the [Squeezenet](https://github.com/onnx/models/tree/master/vision/classification/squeezenet) model from here.


## Install ONNX Runtime for OpenVINO Execution Provider

## Build steps
[build instructions](https://onnxruntime.ai/docs/build/eps.html#openvino)

## Reference Documentation
[Documentation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

If you build it by yourself, you must append the "--build_shared_lib" flag to your build command. Also for running the sample with IO Buffer optimization feature, make sure you set the OpenCL paths. For example if you are setting the path from openvino source build folder, the paths will be like:
```
export OPENCL_LIBS=path/to/your/directory/openvino/bin/intel64/Debug/lib/
export OPENCL_INCS=path/to/your/directory/openvino/thirdparty/ocl/clhpp_headers/include/
```

```
./build.sh --config Release --use_openvino CPU_FP32 --build_shared_lib
```

## Build the sample C++ Application
1. Navigate to the directory /onnxruntime/build/Linux/Release/

2. Now copy all the files required to run this sample at this same location (/onnxruntime/build/Linux/Release/)

3. compile the sample

   - For general sample
      ```
      g++ -o run_squeezenet squeezenet_cpp_app.cpp -I ../../../include/onnxruntime/core/session/ -I /path/to/opencv/include/ -I /path/to/opencv/lib/ -L ./ -lonnxruntime_providers_openvino -lonnxruntime_providers_shared -lonnxruntime -L /path/to/opencv/lib/ -lopencv_imgcodecs -lopencv_dnn -lopencv_core -lopencv_imgproc
      ```
      **Note:**
      If you are using the opencv from openvino package, below are the paths:
      * For latest version (2022.1.0), run download_opencv.sh in /path/to/openvino/extras/script and the opencv folder will be downloaded at /path/to/openvino/extras.
      * For older openvino version, opencv folder is available at openvino directory itself.

   - For the sample using IO Buffer Optimization feature
      Set the OpenCL lib and headers path. For example if you are setting the path from openvino source build folder, the paths will be like:
      ```
      export OPENCL_LIBS=path/to/your/directory/openvino/bin/intel64/Debug/lib/
      export OPENCL_INCS=path/to/your/directory/openvino/thirdparty/ocl/clhpp_headers/include/
      ```
      Now based on the above path, compile command will be:
      ```
      g++ -o run_squeezenet squeezenet_cpp_app_io.cpp -I ../../../include/onnxruntime/core/session/ -I $OPENCL_INCS -I $OPENCL_INCS/../../cl_headers/ -I /path/to/opencv/include/ -I /path/to/opencv/lib/ -L ./ -lonnxruntime_providers_openvino -lonnxruntime_providers_shared -lonnxruntime -L /path/to/opencv/lib/ -lopencv_imgcodecs -lopencv_dnn -lopencv_core -lopencv_imgproc -L $OPENCL_LIBS -lOpenCL
      ```
      **Note:**
      If you are using the opencv from openvino package, below are the paths:
      * For latest version (2022.1.0), run download_opencv.sh in /path/to/openvino/extras/script and the opencv folder will be downloaded at /path/to/openvino/extras.
      * For older openvino version, opencv folder is available at openvino directory itself.
      * The current cmake files are adjusted with the opencv folders coming along with openvino packages. Plase make sure you are updating the opencv paths according to your custom builds.
      
      For the squeezenet IO buffer sample:
      Make sure you are creating the opencl context for the right GPU device in a multi-GPU environment.

4. Run the sample

   - To Run the general sample
      (using Intel OpenVINO-EP)
      ```
      ./run_squeezenet --use_openvino <path_to_onnx_model> <path_to_sample_image> <path_to_labels_file>
      ```
      Example:
      ```
      ./run_squeezenet --use_openvino squeezenet1.1-7.onnx demo.jpeg synset.txt  (using Intel OpenVINO-EP)
      ```
      (using Default CPU)
      ```
      ./run_squeezenet --use_cpu <path_to_onnx_model> <path_to_sample_image> <path_to_labels_file>
      ```
   - To Run the sample for IO Buffer Optimization feature
      ```
      ./run_squeezenet <path_to_onnx_model> <path_to_sample_image> <path_to_labels_file>
      ```

## References:

[OpenVINO Execution Provider](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/faster-inferencing-with-one-line-of-code.html)

[Other ONNXRT Reference Samples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx)
