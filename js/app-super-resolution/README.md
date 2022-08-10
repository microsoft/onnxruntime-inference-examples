# RUNNING SUPER-RESOLUTION ON IMAGES ACROSS VARIOUS PLATFORMS USING ONNX RUNTIME MODELS

## **OVERVIEW**

This example contains an application capable of performing super-resolution on input images using ONNX Runtime and the application can be run on both Android and Web platforms. The IOS implementation is not available yet but will be added soon.

This project makes use of the [Expo framework](https://docs.expo.dev/) which is a free and open source toolchain built around React Native to help you build native projects using JavaScript and React.

For steps on how to build this project, refer to [this](instructions.md).

**NOTE:** **_This application makes use of a Web API that may not be supported in some browsers. Check Browser Compatibility_** [here](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas#browser_compatibility)

## **ONNX Runtime Usage**

### **Model**

Lightweight model capable of increasing the resolution of images.

For this example, the model used will be in [ORT format](https://onnxruntime.ai/docs/reference/ort-format-models.html#what-is-the-ort-model-format). This format is usually used in size-constrained environments hence why it is used here. The ORT format model used can be found [here](https://github.com/VictorIyke/super_resolution_MW/blob/main/cross_plat/assets/super_resnet12.ort).

### Limitations:

1. The model do not work properly with full HD images. The output becomes more pixelated.
2. For very unclear blurry images, the models’ output has little to no noticeable change.
3. The input image size ratio should be close to 1:1. The model works on input images with identical height and width, so an image that does not meet this standard will have to be awfully shrunk (or stretched) to fit the ratio.

### **Pre-processing**

The pre-processing method used here consists of:

1. Resizing/Cropping the image selected by the User to a fixed dimension (224x224)
2. Obtaining the Y'Cb'Cr' pixel data of the image.
3. Feeding only the Y' channel of the pixel data to the model as an ORT tensor.

### **Post-processing**

The post-processing method used here consists of converting the Y' pixels from the model output to RGB format and creating an image from the RGB pixels to be displayed to the User.

### **Platform Differences**

Depending on the platform, there are different ways image data is obtained from the input image and output image is obtained from the image data.

- `For Mobile:` The mobile implementation makes use of [Android Native Modules](https://reactnative.dev/docs/next/native-modules-intro) written in Java. These `native modules` have functions responsible for obtaining image data given an image source and also creating an image given its image data.

- `For Web:` The web implementation makes use of the [Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) and [Offscreen Canvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas) Web API which makes working with images easier on web. With this API, the input images and the output images can be drawn on a `canvas` and the image data can be obtained from the canvas.

## Examples

1. MOBILE (Android)

   <img width="318" alt="image" src="https://user-images.githubusercontent.com/106185642/181639530-9c808167-d68c-49d4-8e89-72aeeb11164e.png">

2. WEB

   <img width="1215" alt="image" src="https://user-images.githubusercontent.com/106185642/181638855-f341e52e-dfc1-4362-b93a-0117f0cfd65a.png">
