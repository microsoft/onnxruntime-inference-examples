# ONNX Runtime Mobile Examples

These examples demonstrate how to use ONNX Runtime (ORT) Mobile in mobile applications.

## General Prerequisites

These are some general prerequisites.
Examples may specify other requirements if applicable.
Please refer to the instructions for each example.

### Get the Code

Clone this repo.

```bash
git clone https://github.com/microsoft/onnxruntime-inference-examples.git
```

### iOS Example Prerequisites

- Xcode 12.5+
- CocoaPods
- A valid Apple Developer ID if you want to run the example on a device

## Examples

### Basic Usage

The example app shows basic usage of the ORT APIs.

- [iOS Basic Usage](examples/basic_usage/ios)

### Image Classification

The example app uses image classification which is able to continuously classify the objects it sees from the device's camera in real-time and displays the most probable inference results on the screen.

- [Android Image Classifier](examples/image_classifications/android)

### Speech Recognition

The example app uses speech recognition to transcribe speech from audio recorded by the device.

- [iOS Speech Recognition](examples/speech_recognition/ios)

### Object Detection

The example app uses object detection which is able to continuously detect the objects in the frames seen by your iOS device's back camera and display the detected object bounding boxes, detected class and corresponding inference confidence on the screen.

- [iOS Object Detection](examples/object_detection/ios)
