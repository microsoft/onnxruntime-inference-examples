# ORT Objective-C Basic Usage Example

This example contains some simple code that uses the Objective-C API.
- [Swift example code](OrtBasicUsage/SwiftOrtBasicUsage.swift)
- [Objective-C++ example code](OrtBasicUsage/ObjcOrtBasicUsage.mm)

## Set up

### Generate the model

The model should be generated in this location: `<this directory>/OrtBasicUsage/model`

See instructions [here](../model/readme.md) for how to generate the model.

For example, with the model generation script dependencies installed, from this directory, run:

```bash
../model/gen_model.sh ./OrtBasicUsage/model
```

### Install the Pod dependencies

From this directory, run:

```bash
pod install
```

## Build and run

Open the generated OrtBasicUsage.xcworkspace file in Xcode to build and run the example.
