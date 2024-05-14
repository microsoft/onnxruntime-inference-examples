# ONNX Runtime Mobile Phi-3 Android sample application with ORT GenAI library support

## Overview

This is a basic [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) example application for [ONNX Runtime](https://github.com/microsoft/onnxruntime) on Android with [Ort-GenAI](https://github.com/microsoft/onnxruntime-genai) support for running generative AI models. This app demonstrates the usage of phi-3 model in a simple question answering chatbot mode.

### Model
The model used here is from the source: [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx#phi-3-mini-4k-instruct-onnx-models)
and optimized for int4 CPU and Mobile usage.

### Requirements
- Android Studio Giraffe | 2022.3.1 or later (installed on Mac/Windows/Linux)
- Android SDK 29+
- Android NDK r22+
- An Android device or an Android Emulator

## Build And Run

### Step 1: Clone the ONNX runtime mobile examples source code

Clone this repository to get the sample application. 

`git@github.com:microsoft/onnxruntime-inference-examples.git`

### [Optional] Step 2: Prepare the model

The current set up supports downloading Phi-3-mini model directly from Huggingface repo to the android device folder. However, it takes time since the model data is >2.5G.

You can also follow this link to download **Phi-3-mini**: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
and manually copy to the android device file directory following the below instructions:

#### Steps for manual copying models to android device directory:
From Android Studio:
  - create (if necessary) and run your emulator/device
    - make sure it has at least 8GB of internal storage
  - debug/run the app so it's deployed to the device and creates it's `files` directory
    - expected to be `/data/data/ai.onnxruntime.genai.demo/files`
      - this is the path returned by `getFilesDir()`
  - Open Device Explorer in Android Studio
  - Navigate to `/data/data/ai.onnxruntime.genai.demo/files`
    - adjust as needed if the value returned by getFilesDir() differs for your emulator or device
  - copy the whole [phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4) model folder to the `files` directory

### Step 3: Connect Android Device and Run the app
  Connect your Android Device to your computer or select the Android Emulator in Android Studio Device manager.

  Then select `Run -> Run app` and this will prompt the app to be built and installed on your device or emulator.

  Now you can try giving some sample prompt questions and test the chatbot android app by clicking the ">" action button.

#
Here are some sample example screenshots of the app.

<img width=20% src="images/Local_LLM_1.jpg" alt="App Screenshot 1" />

<img width=20% src="images/Local_LLM_2.jpg" alt="App Screenshot 2" />

<img width=20% src="images/Local_LLM_3.jpg" alt="App Screenshot 3" />

