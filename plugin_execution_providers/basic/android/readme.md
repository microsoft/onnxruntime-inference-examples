# Basic Plugin Execution Provider on Android

## Contents
- `basicpluginep`: An Android package containing the native basic plugin EP library. In addition to providing the EP library files, it also provides helper functions to get the EP library path and the EP name.
- `basicpluginepusage`: An example application showing how to use the basic plugin EP Android package. It registers the basic plugin EP library with ONNX Runtime and then runs inference using that EP.

## Build Instructions

### Android Studio
This directory can be opened with Android Studio. Build and run `basicpluginepusage`.

### Command Line
Use Gradle to build the project:

```
./gradlew build
```

The AAR files for `basicpluginep` will be generated in the `./basicpluginep/build/outputs/aar` directory.
