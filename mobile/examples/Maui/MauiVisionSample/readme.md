# ONNX Runtime MAUI Vision Sample

The [MAUI Vision Sample](MauiVisionSample.sln) demonstrates the use of two different vision models from the [ONNX Model Zoo collection](https://github.com/onnx/models/tree/main), by a [MAUI](https://docs.microsoft.com/en-us/dotnet/maui/what-is-maui) app.

## Overview
The app enables you to take/pick a photo on the device or use a sample image to explore the following models.

### [Mobilenet](https://github.com/onnx/models/tree/main/vision/classification/mobilenet)

Classifies the major object in the image into 1,000 pre-defined classes.

### [Ultraface](https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface)

Lightweight face detection model designed for edge computing devices providing detection boxes and scores for a given image.

This model is included in the repository, but has been updated using the onnxruntime python package tools to:
- remove unused initializers,
  - `python -m onnxruntime.tools.optimize_onnx_model --opt_level basic <model>.onnx <updated_model>.onnx`
- make the initializers constant
  - requires a script from the ONNX Runtime repository. this can be downloaded from [here](https://github.com/microsoft/onnxruntime/blob/master/tools/python/remove_initializer_from_input.py)
  - `python remove_initializer_from_input.py --input <model>.onnx --output <updated_model>.onnx`

in order to enable more runtime optimizations to occur.

The sample also demonstrates how to use the default **CPU EP ([Execution Provider](https://onnxruntime.ai/docs/execution-providers))** as well as add a  platform-specific execution provider. In this case, [NNAPI](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html) for Android and [CoreML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) for iOS.


## Building

To build the sample Visual Studio 2022 Preview is currently required, as .net 6 support (which includes MAUI) is only available in that version.

The MAUI workload should be installed. See [here](https://docs.microsoft.com/en-us/dotnet/maui/get-started/first-app) for more information.

If you get any errors during building about missing platform workloads a simple way to resolve is to open a terminal from Visual Studio 2022 Preview (View->Terminal) and run the command `dotnet workload restore MauiVisionSample.sln`

## ONNX Runtime Nuget Packages

There are two ONNX Runtime NuGet packages used. 
 - Microsoft.ML.OnnxRuntime
   - contains the native libraries for all supported platforms, including various architectures of Windows, Linux, Mac, iOS and Android
 - Microsoft.ML.OnnxRuntime.Managed
   - contains the C# bindings to use the native libraries

## ONNX Runtime Usage

The key parts demonstrating the integration of ONNX Runtime are:

### Model

The ONNX models are included in the [Raw](MauiVisionSample/Resources/Raw) resources. 

The model bytes are loaded using `OpenAppPackageFileAsync` in `Utils::LoadResource`. Note that it is necessary to copy to a byte[] at runtime as the model may be compressed in the app on platforms such as Android.

### Microsoft.ML.OnnxRuntime.InferenceSession

The `InferenceSession` reads the ONNX model bytes, optimizes the model, and handles model execution.

An `InferenceSession` instance can be used to execute the model multiple times, including concurrently, so should only ever be created once per model. 

Creating an `InferenceSession` that uses the CPU Execution Provider is initially done in `VisionSampleBase::Initialize`.

We also demonstrate the steps to enable an additional execution provider (if available for the platform) in `VisionSampleBase::UpdateExecutionProviderAsync`. 
As execution providers must be selected prior to the creation of the `InferenceSession` this necessitates an expensive re-creation of the session in the sample app. Typically you would pre-determine which execution providers you wanted enabled and create the session once with that information.

The CPU Execution Provider will be able to run all models.

In some scenarios it may be beneficial to use the NNAPI Execution Provider on Android, or the CoreML Execution Provider on iOS. 
This is highly dependent on your model, and also the device (particularly for NNAPI), so it is necessary for you to performance test.

## Image acquisition

There are 3 potential ways to acquire the image to process in the sample app.

- If the sample image is selected the bytes from the jpg are loaded from the Raw resources.
  - see `GetSampleImageAsync` in [MainPage.xaml.cs](MauiVisionSample/MainPage.xaml.cs)
- If we use the MAUI MediaPicker to select/capture an image we make sure the orientation is correct and convert to a byte[] of jpg format for consistency.
  - see `TakePhotoAsync` and `PickPhotoAsync` in [MainPage.xaml.cs](MauiVisionSample/MainPage.xaml.cs)

### Pre-processing

Pre-processing the image involves converting it into a Tensor in the same way the input data used during model training was converted. These steps are model specific given they have to match how the individual model was trained.

At a high level there are two steps for the preprocessing:

- First is to convert the byte[] to an SKBitmap using SKBitmap.Decode, and apply any image level changes like resizing and cropping to the bitmap.
- Second is to convert the bitmap into the Tensor. 
  - for these models that involves converting the byte data in the bitmap to a float value for each R, G and B value in each pixel, 
normalizing those values, and arranging in the NCHW format that ONNX uses.
  - [MobilenetImageProcessor.cs](MauiVisionSample/Models/Mobilenet/MobilenetImageProcessor.cs) has comments explaining these steps and the NCHW format in more detail.

Each model has an image processor to implement the model specific logic required for these steps. 
We leverage the `SkiaSharpImageProcessor` implementation of the `IImageProcessor` interface for common tasks.

### Model Execution

Once we have converted our input image into a Tensor we can execute the model by calling`InferenceSession::Run` with the Tensor.
This is done in the implementations of `IVisionSample::GetPredictions`. 

### Post-processing

The output for each model is always model specific. The output is first processed into a meaningful format in `IVisionSample::GetPredictions`.

*Mobilenet*: The results involve 1000 scores in the same order as the 1000 labels. We use Softmax to convert the scores to probabilities and return the matching name from the labels and probability for the top 3 matches.

*Ultraface*: The results have confidence scores and bounding boxes for each match. We select the best match, if one is found with a confidence score > 50%, and return the bounding box and confidence score. Additionally, `ApplyPredictionsToImage` will draw the bounding box and score on the input image so we can display the result to the user.

