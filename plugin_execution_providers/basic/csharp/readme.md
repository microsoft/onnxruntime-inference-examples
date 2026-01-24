# Basic Plugin Execution Provider with C#

## Contents
- `Contoso.ML.OnnxRuntime.EP.Basic/`: Contains files for the basic plugin EP C# NuGet package. `BasicEp.cs` provides helper functions to get the EP library path and the EP name.
- `SampleApp/`: Contains a sample C# application showing example usage of the basic plugin EP C# NuGet package.
- `setup.bat`: Batch script to generate the NuGet package.

## Build Instructions
This example currently only supports Windows x64 and Windows ARM64.

### Build the native plugin EP library

Follow instructions [here](../readme.md#build-instructions) to build the native library.

### Build the C\# NuGet Package

Set the environment variable `BASIC_PLUGIN_EP_LIBRARY_PATH` to the path to the native plugin EP shared library. E.g., `basic_plugin_ep.dll`.

Run `setup.bat` from this directory. Pass the build configuration (e.g., Release or Debug) as an argument.

```
.\setup.bat Release
```

The generated NuGet package will be copied to the `./local_feed` directory.

## Build and run the sample application

Build and run the sample application.

```
dotnet build .\SampleApp\SampleApp.csproj -c Release
dotnet run --project .\SampleApp\SampleApp.csproj -c Release
```
