# Android Speech Recognition Example with Azure Custom Op

This example creates a version to show how to use ORT to do speech recognition using the [Whisper](https://github.com/openai/whisper) model and calls the OpenAI Whisper endpoint using the Azure custom op.

The application lets the user make an audio recording, then recognizes the speech from that recording and displays a transcript.

Example App Scrennshot - A Successful result returned from OpenAI endpoint:

<img width=25% src="images/Img_successful_endpoint_res.png" alt="App Screenshot" />

## Set up

### Prerequisites

See the general prerequisites [here](../../../README.md#General-Prerequisites).

Additionally, you will need to be able to record audio, either on an emulator or a device.

This example was developed with Android Studio Giraffe | 2022.3.1 Patch 1.
It is recommended to use that version or a newer one.

### Generate the model

This is optional. Currently a usable model file is checked in.

Model was created with [create_openai_whisper_transcriptions.py](https://github.com/microsoft/onnxruntime-extensions/blob/main/test/data/azure/create_openai_whisper_transcriptions.py) script. If any changes are required they should be simple adjust. (e.g. change default audio format from wav to mp3).

Copy the model to `app/src/main/res/raw/openai_whisper_transcriptions.onnx`.

### Prepare an OpenAI Auth Token

It's required in the app to provide an OpenAI auth token as the first input to the model.

For local testing, you should be able to create an account with API key and initial free credits.

[OpenAI authentication](https://platform.openai.com/docs/plugins/authentication)
## Build and run

Open this directory in Android Studio to build and run the example.
