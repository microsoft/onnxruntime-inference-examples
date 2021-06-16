# ONNX Runtime Mobile object detection iOS sample application

This is an example app uses object detection which is able to continuously detect the objects in the frames seen by your iOS device's back camera and display the detected object bounding boxes, detected class and corresponding inference confidence on the screen.

This example is loosely based on [Google Tensorflow lite - Object Detection Examples](https://github.com/tensorflow/examples/)

### Model
[//]: # (Add the TF mobilenet SSD v2_300 float model link here)
We use pre-trained MobileNet SSD V2 model from TensorFlow in this sample app. 

## Requirements
- Install Xcode 12.5 and above (preferably latest version)
- A valid Apple Developer ID
- A real iOS device with a camera (preferably iphone 12/iphone 12 pro)
- Xcode command line tools `xcode-select --install`
- Clone the `onnxruntime-inference-examples` source code repo

## Build And Run

0. [Optional] Prepare ORT format model from TF -> ONNX -> ORT
[//]: # (Add the information from TF->ONNX->ORT here)

1. Install CocoaPods. `sudo gem install cocoapods`

2. Run `pod install` to generate the workspace file under `examples/object_detections/ios/`. 
- At the end of this step, you should get a file called `ORTObjectDetection.xcworkspace`.

3. Download and copy the SSDMobileNetV2 ORT model to `examples/object_detections/ios/ORTObjectDetection/ModelsAndData/`. The ORT format ssd model can be downloaded here. 
[//]: # (Add the ort format model link here)

4. Open `ORTObjectDetection.xcworkspace` in xcworkspace and make sure to select your corresponding development team under `Target-General-Signing` for a proper codesign procedure to run the app.

5. Connect your iOS device, build and run the app. You'll have to grant permissions for the app to use the device's camera.


### iOS App related information

This app uses [ONNX Runtime Objective-C API](https://www.onnxruntime.ai/docs/reference/api/objectivec-api.html) for performing object detection function.

It is written entirely in Swift and uses a bridgingheader file for the framework to be used in a Swift app.





