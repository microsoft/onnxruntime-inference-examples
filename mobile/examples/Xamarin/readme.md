# ONNX Runtime Xamarin Sample

The [VisionSample](VisionSample/VisionSample.sln) demonstrates the use of several [vision-related models](https://github.com/onnx/models/tree/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde#vision), from the [ONNX Model Zoo collection](https://github.com/onnx/models/tree/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde#onnx-model-zoo), by a [Xamarin.Forms](https://dotnet.microsoft.com/apps/xamarin/xamarin-forms) app.

## Overview
The sample enables you to take/pick a photo on the device or use a sample image, if one is provided, to explore the following models.

### [ResNet](https://github.com/onnx/models/tree/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/classification/resnet#resnet)

Classifies the major object in the image into 1,000 pre-defined classes.

### [Ultraface](https://github.com/onnx/models/tree/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/body_analysis/ultraface#ultra-lightweight-face-detection-model)

Lightweight face detection model designed for edge computing devices providing detection boxes and scores for a given image.
This model is included in the repository, but has been updated to remove unused initializers, and to make the initializers constant to enable more runtime optimizations to occur.

The sample also demonstrates how to switch between the default **CPU EP ([Execution Provider](https://onnxruntime.ai/docs/execution-providers))** and platform-specific options. In this case, [NNAPI](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html) for Android and [Core ML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) for iOS.

## Getting Started

> [!IMPORTANT]
> There are some [known issues](#known-issues) that could impact aspects of the sample on specific devices or environments. See [known issues section](#known-issues) for workarounds.

The [VisionSample](VisionSample/VisionSample.sln) should build and run as-is, and looks for model files in a folder in this directory called **Models**.
The Ultraface model should already exist in that directory and to able to be used.

Additionally the ResNet model can be added if desired.
With **Models** set as the current directory, you can use [wget](https://www.gnu.org/software/wget) to download it.

From this directory:
```
> cd Models
> wget <model_url>
```

| MODEL  | DOWNLOAD URL | Size   |
| ------ | ------------ | ------ |
| ResNet  | https://github.com/onnx/models/raw/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/classification/resnet/model/resnet50-v2-7.onnx | 97.7 MB |

> [!NOTE]
> You may need to reload [VisionSample.csproj](VisionSample/VisionSample/VisionSample.csproj) before newly added model files will appear in [Visual Studio Solution Explorer](https://docs.microsoft.com/visualstudio/ide/use-solution-explorer?view=vs-2022).

### Use ONNX Runtime prerelease nuget packages
If you want to use a prerelease version of ONNX Runtime nuget packages from the integration repository, update the [nuget.config](nuget.config)
```diff
   <add key="NuGetOrg" value="https://api.nuget.org/v3/index.json" />
+   <add key="INT NuGetOrg" value="https://apiint.nugettest.org/v3/index.json" />
```
And choose the prerelease version of ONNX Runtime nuget packages

- Microsoft.ML.OnnxRuntime.Managed for [VisionSample.csproj](VisionSample/VisionSample/VisionSample.csproj)
- Microsoft.ML.OnnxRuntime for [VisionSample.Forms.Android.csproj](VisionSample/VisionSample.Forms.Android/VisionSample.Forms.Android.csproj) and [VisionSample.Forms.iOS.csproj](VisionSample/VisionSample.Forms.iOS/VisionSample.Forms.iOS.csproj)

## Known Issues

### Several open issues relating to Xamarin Media components

The sample leverages [Xamarin.Essetials MediaPicker APIs](https://docs.microsoft.com/xamarin/essentials/media-picker?context=xamarin%2Fxamarin-forms&tabs=android) and [Xam.Plugin.Media](https://github.com/jamesmontemagno/MediaPlugin#media-plugin-for-xamarin-and-windows) to handle taking and picking photos in a cross-platform manner. There are several open issues which may impact the ability to use these components on specific devices.

- [Xamarin.Essentials](https://github.com/xamarin/Essentials/issues)
- [Xam.Plugin.Media](https://github.com/jamesmontemagno/MediaPlugin/issues)

The take and capture photo options are provided as a convenience but are not directly related to the use of [ONNX Runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) packages by a [Xamarin.Forms](https://dotnet.microsoft.com/apps/xamarin/xamarin-forms) app. If you're unable to use those options, you can explore use of the models using the sample image option instead.

### [MissingMethodException](https://docs.microsoft.com/dotnet/api/system.missingmethodexception) related to [ReadOnlySpan&lt;T>](https://docs.microsoft.com/dotnet/api/system.readonlyspan-1)

In [Visual Studio 2022](https://visualstudio.microsoft.com), [Hot Reload](https://docs.microsoft.com/xamarin/xamarin-forms/xaml/hot-reload) loads some additional dependencies including [System.Memory](https://www.nuget.org/packages/System.Memory) and [System.Buffers](https://www.nuget.org/packages/System.Buffers) which may cause conflicts with packages such as [ONNX Runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Managed). The workaround is to [Disable Hot Reload](https://docs.microsoft.com/xamarin/xamarin-forms/xaml/hot-reload#enable-xaml-hot-reload-for-xamarinforms) until the [issue](https://developercommunity.visualstudio.com/t/bug-in-visual-studio-2022-xamarin-signalr-method-n/1528510#T-N1585809) has been addressed.