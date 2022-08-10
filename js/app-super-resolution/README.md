# RUNNING SUPER-RESOLUTION ON IMAGES ACROSS VARIOUS PLATFORMS USING ONNX RUNTIME MODELS

## OVERVIEW

This example contains an application capable of performing super-resolution on input images using ONNX Runtime and the application can be run on both Android and Web platforms. The IOS implementation is not available yet but will be added soon.

This project makes use of the [Expo framework](https://docs.expo.dev/) which is a free and open source toolchain built around React Native to help you build native projects using JavaScript and React.

For steps on how to build this project, refer to [this](instructions.md).

**NOTE:** **_This application makes use of a Web API that may not be supported in some browsers. Check Browser Compatibility_** [here](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas#browser_compatibility)

## ONNX Runtime Usage

### Model

For this example, the model used will be in [ORT format](https://onnxruntime.ai/docs/reference/ort-format-models.html#what-is-the-ort-model-format). This format is usually used in size-constrained environments hence why it is used here.

The ORT format model used can be found [here](https://github.com/VictorIyke/super_resolution_MW/blob/main/cross_plat/assets/super_resnet12.ort).

### Pre-processing

- ### `Mobile`

      ...

- ### `Web`

      ...

### Post-processing

- ### `Mobile`

      ...

- ### `Web`

      ...

## Examples

1. MOBILE (Android)

   <img width="318" alt="image" src="https://user-images.githubusercontent.com/106185642/181639530-9c808167-d68c-49d4-8e89-72aeeb11164e.png">

2. WEB

   <img width="1215" alt="image" src="https://user-images.githubusercontent.com/106185642/181638855-f341e52e-dfc1-4362-b93a-0117f0cfd65a.png">
