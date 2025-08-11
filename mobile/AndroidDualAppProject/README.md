# Android Dual App Project

This project demonstrates a dual-app architecture for on-device AI inference using ONNX Runtime on Android.

## Overview

The project consists of two separate Android applications:

1. **UI App**: Provides user interface for input collection and result display.
2. **AI App**: Handles the AI model loading and inference using ONNX Runtime and ONNX Runtime GenAI.

## Architecture

The apps communicate using Android's Inter-Process Communication (IPC) mechanism via AIDL (Android Interface Definition Language).

```
+-------------+           +-------------+
|             |           |             |
|   UI App    |<--------->|   AI App    |
| (User Input)|    AIDL   | (Inference) |
|             |           |             |
+-------------+           +-------------+
                               |
                               v
                          +----------+
                          |  ONNX    |
                          |  Model   |
                          +----------+
```

### Why Two Separate Apps?

This dual-app architecture provides several benefits:

1. **Memory Isolation**: The AI inference process runs in a separate memory space from the UI, preventing the UI from becoming unresponsive during inference.

2. **Process Separation**: If the model inference causes crashes or excessive memory usage, it won't crash the UI app.

3. **Resource Management**: The AI app can be configured to run at a different priority level than the UI app.

4. **Update Flexibility**: The AI model implementation can be updated independently of the UI application.

5. **Security**: Sensitive model operations are isolated from the user-facing app.

### Communication Flow

1. The UI App binds to the AIApp's InferenceService using AIDL.
2. User enters a prompt in the UI App.
3. The prompt is sent via AIDL to the AIApp.
4. The AIApp loads the model (if not already loaded) and performs inference.
5. The AIApp returns the generated response back to the UI App.
6. The UI App displays the response to the user.

## Key Components

### UI App

- `MainActivity`: Manages the user interface and communication with the AI App.
- `IInferenceService.aidl`: Interface definition for communicating with the AI App's service.

### AI App

- `InferenceService`: Android Service that manages the AI model and processes inference requests.
- `ModelManager`: Handles the ONNX Runtime model initialization and inference.
- `ModelDownloader`: Utility for downloading or extracting model files.
- `IInferenceService.aidl`: Interface implementation for the service.

## Dependencies

- ONNX Runtime for Android: `com.microsoft.onnxruntime:onnxruntime-android:1.16.1`
- ONNX Runtime GenAI: `onnxruntime-genai-android-0.4.0-dev.aar`

## Setup and Running

### Building the Apps

1. Open the project in Android Studio
2. Build both the AIApp and UIApp modules
3. Make sure you have the `onnxruntime-genai-android-0.4.0-dev.aar` file in the `AIApp/libs/` folder

### Setting up the Model Files

1. Download the Microsoft Phi-3.5-mini model files (ONNX format)
2. Use the provided PowerShell script to push the model to your device:

```powershell
.\push_model.ps1 -modelDir "path\to\model\directory"
```

Or manually push the files via ADB:

```powershell
adb shell mkdir -p /sdcard/phi-3-model
adb push path\to\model-file.onnx /sdcard/phi-3-model/
adb push path\to\tokenizer.json /sdcard/phi-3-model/
```

### Running the Apps

1. Install both apps on your Android device
2. Launch the AI App first to initialize the service
3. Grant necessary storage permissions when prompted
4. Start the service using the button in the AIApp
5. Launch the UI App which will automatically connect to the AI service
6. Enter prompts in the UIApp and receive model responses

### Monitoring Performance

The apps provide detailed performance logs via Android logcat. Connect your device to a computer and run:

```powershell
adb logcat -s ModelManager:D MemoryMonitor:D InferenceService:D UIApp:D
```

## Permissions

The application requires the following permissions:

- `READ_EXTERNAL_STORAGE` / `WRITE_EXTERNAL_STORAGE` - For accessing model files in external storage
- `READ_MEDIA_*` (Android 13+) - For accessing model files on newer Android versions

For Android 10+ devices, the app uses `requestLegacyExternalStorage="true"` to ensure compatibility with older storage access methods.

## Model Setup

### Pushing Model Files to Device using ADB

The app is configured to load model files from `/sdcard/phi-3-model/` directory on your Android device. Follow these steps to push the model files:

1. Download the Phi-3.5 model files (the ONNX format files) from Microsoft or HuggingFace
2. Connect your Android device to your computer
3. Enable USB debugging on your device
4. Open a terminal and use these ADB commands:

```bash
# Create directory on device
adb shell mkdir -p /sdcard/phi-3-model

# Push model files to device
adb push path/to/model-file.onnx /sdcard/phi-3-model/
adb push path/to/tokenizer.json /sdcard/phi-3-model/
# Push any other required model files
```

5. Verify files were transferred correctly:
```bash
adb shell ls -la /sdcard/phi-3-model/
```

### Alternative Setup Methods

For a production app, you might want to:
1. Copy the model files to the app's assets folder
2. Extract them at runtime to the app's internal storage
3. Or download them from a server at runtime

## License

[Your license information here]
