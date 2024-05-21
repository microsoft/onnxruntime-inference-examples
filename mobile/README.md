# ONNX Runtime Mobile Examples

These examples demonstrate how to use ONNX Runtime (ORT) in mobile applications.

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

- [Android Image Classifier](examples/image_classification/android)

### Speech Recognition

The example app uses speech recognition to transcribe speech from audio recorded by the device.

- [iOS Speech Recognition](examples/speech_recognition/ios)

### Object Detection

The example app uses object detection which is able to continuously detect the objects in the frames seen by your iOS device's back camera and display the detected object bounding boxes, detected class and corresponding inference confidence on the screen.

- [iOS Object Detection](examples/object_detection/ios)
- [Android Object Detection](examples/object_detection/android)

### Xamarin VisionSample

The [Xamarin.Forms](https://dotnet.microsoft.com/apps/xamarin/xamarin-forms) example app demonstrates the use of several vision-related models, from the ONNX Model Zoo collection.

- [Xamarin VisionSample](examples/Xamarin)

### Super Resolution

The example application accomplishes the task of recovering a high resolution (HR) image from its low resolution counterpart with Ort-Extensions support for pre/post processing. Currently supports on platform Android and iOS.

- [Android Super Resolution](examples/super_resolution/android)
- [iOS Super Resolution](examples/super_resolution/ios)

### Question Answering

The example app gives a demo of introducing question answering models with pre/post processing into mobile scenario. Currently supports on platform Android and iOS.

- [Android Question answering](examples/question_answering/android)
- [iOS Question answering](examples/question_answering/ios)

### Whisper (Cloud/Local)

This example shows how to use ORT to do speech recognition using the [Whisper](https://github.com/openai/whisper) model. One version (Cloud) calls the OpenAI Whisper endpoint using the Azure custom op. The other uses a local Whisper model.

- [Azure Whisper](examples/whisper/azure/android)
- [Local Whisper](examples/whisper/local/android)

### React Native Expo Sample

This is an example React Native Expo project which demonstrates basic usage of ORT such as loading onnx models and creating inference sessions, etc.

- [React Native Basic Sample](examples/React_Native/ort-rn-basic-usage/)

### Phi-3 Sample

This is a basic Phi-3 (Android for now) example application with ONNX Runtime mobile and ONNX Runtime Generate() API with support for efficiently running generative AI models. This app demonstrates the usage of phi-3 model in a simple question answering chatbot mode.

- [Android Phi-3](examples/phi-3/android)
