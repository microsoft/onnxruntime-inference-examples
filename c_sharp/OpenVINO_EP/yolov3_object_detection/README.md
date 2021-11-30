# Object detection with YOLOv3 in C# using OpenVINO Execution Provider:

1. The object detection sample uses YOLOv3 Deep Learning ONNX Model from the ONNX Model Zoo.

2. The sample involves presenting an image to the ONNX Runtime (RT), which uses the OpenVINO Execution Provider for ONNX RT to run inference on Intel<sup>®</sup> NCS2 stick (MYRIADX device). The sample uses ImageSharp for image processing and ONNX Runtime OpenVINO EP for inference.

The source code for this sample is available [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_sharp/OpenVINO_EP/yolov3_object_detection).

# How to build

## Prerequisites
1. Install [.NET 5.0](https://dotnet.microsoft.com/download/dotnet/5.0) or higher for your OS (Mac, Windows or Linux).
2. [The Intel<sup>®</sup> Distribution of OpenVINO toolkit](https://docs.openvinotoolkit.org/latest/index.html)
3. Use any sample Image as input to the sample.
4. Download the latest YOLOv3 model from the ONNX Model Zoo.
   This example was adapted from [ONNX Model Zoo](https://github.com/onnx/models).Download the latest version of the [YOLOv3](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3) model from here.

## Install ONNX Runtime for OpenVINO Execution Provider

## Build steps
[build instructions](https://onnxruntime.ai/docs/build/eps.html#openvino)

## Reference Documentation
[Documentation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

To build nuget packages of onnxruntime with openvino flavour
```
./build.sh --config Release --use_openvino MYRIAD_FP16 --build_shared_lib --build_nuget
```
## Build the sample C# Application
1. Create a new console project
```
dotnet new console
```
2. Install Nuget Packages of Onnxruntime and [ImageSharp](https://www.nuget.org/packages/SixLabors.ImageSharp)
    1. Open the Visual C# Project file (.csproj) using VS19.
    2. Right click on project, navigate to manage Nuget Packages.
    3. Install SixLabors.ImageSharp, SixLabors.Core, SixLabors.Fonts and SixLabors.ImageSharp.Drawing Packages from nuget.org.
    4. Install Microsoft.ML.OnnxRuntime.Managed and Microsoft.ML.OnnxRuntime.Openvino from your build directory nuget-artifacts.

3. Compile the sample
```
dotnet build
```

4. Run the sample
```
dotnet run [path-to-model] [path-to-image] [path-to-output-image]
```

## References:

[OpenVINO Execution Provider](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/faster-inferencing-with-one-line-of-code.html)

[Get started with ORT for C#](https://onnxruntime.ai/docs/get-started/with-csharp.html)

[fasterrcnn_csharp](https://onnxruntime.ai/docs/tutorials/fasterrcnn_csharp.html))

[resnet50_csharp](https://onnxruntime.ai/docs/tutorials/resnet50_csharp.html)

