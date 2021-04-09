# ONNX Runtime Mobile image classification Android sample application

## Overview
This is an example application for [ONNX Runtime](https://github.com/microsoft/onnxruntime) on Android. The demo app uses image classfication which is able to continuously classify the objects it sees from the device's camera in real-time and displays the most probable inference result on the screen.

This example is loosely based on [Google CodeLabs - Getting Started with CameraX](https://codelabs.developers.google.com/codelabs/camerax-getting-started)

### Model
We use classic MobileNetV2(float) model and MobileNetV2 (uint8) in this sample app.

## Requirements
- Android Studio 4.1+ (installed on Mac/Windows/Linux)
- Android SDK 29+
- Android NDK r21+
- Android device in developer mode and enable USB debugging

## Build And Run
### Prerequisites
-  MobileNetV2 ort format model
-  labels text file (used for image classification)
-  Prebuilt ONNX Runtime arm64 Android Archive(AAR) files, which can be directly imported in Android Studio

The above three files are provided and can be downloaded [here](https://1drv.ms/u/s!Auaxv_56eyubgQX-S_kTP0AP66Km?e=e8YMX1).

[Optional] You can also build your own ONNX Runtime arm64 AAR files for Android. (See [build instructions here](https://www.onnxruntime.ai/docs/how-to/build.html#android) and [Build Android Archive(AAR)](https://www.onnxruntime.ai/docs/how-to/build.html#build-android-archive-aar)). 


### Step 1. Clone the ONNX Runtime Mobile examples source code and download required model files
Clone this ORT Mobile examples GitHub repository to your computer to get the sample application.

Download the packages provided in `Prerequisites`.

- Copy MobileNetV1 onnx model and the labels file to `example/image_classification/android/app/src/main/res/raw/`

- Create `/libs` directory under `app/` and copy the AAR file `onnxruntime-release-1.7.0.aar` to `app/libs`

Then open the sample application in Android Studio. To do this, open Android Studio and select `Open an existing project`, browse folders and open the folder `examples/image_classification/android/`.

<img width=60% src="images/screenshot_1.png"/>

### Step 2. Build the sample application in Android Studio

Select `Build-Make Project` in the top toolbar in Android Studio and check the projects has built successfully.

<img width=60% src="images/screenshot_3.png" alt="App Screenshot"/>

<img width=60% src="images/screenshot_4.png" alt="App Screenshot"/>

### Step 3. Connect your Android Device and run the app

Connect your Android Device to the computer and select your device in the top-down device bar.

<img width=60% src="images/screenshot_5.png" alt="App Screenshot"/>

<img width=60% src="images/screenshot_6.png" alt="App Screenshot"/>

Then Select `Run-Run app` and this will prompt the app to be installed on your device. 

Now you can test and try by opening the app `ort_image_classifier` on your device. The app may request your permission for using the camera.


#
Here's an example screenshot of the app. 

<img width=20% src="images/screenshot_2.jpg" alt="App Screenshot" />



