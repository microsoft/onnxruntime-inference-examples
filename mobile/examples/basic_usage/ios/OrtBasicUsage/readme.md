# ORT Objective-C Basic Usage Example

This example contains some simple code that uses the Objective-C API.

## Set up

### Generate the model

The model should be generated in this location: `<this directory>/OrtBasicUsage/model`

Install dependencies as described [here](../../model/readme.md).

Then, from this directory, run:

```bash
../../model/gen_model.sh ./OrtBasicUsage/model
```

### Install the Pod dependencies

From this directory, run:

```bash
pod install
```

## Build and run

Open the generated .xcworkspace file in XCode to build and run the example.
