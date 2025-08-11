# Phi-3.5 Model Integration Notes

This document provides detailed instructions for integrating the Microsoft Phi-3.5 model with the Android dual app architecture.

## Model Requirements

The Phi-3.5-mini model has the following requirements:

1. ONNX Runtime GenAI library (`onnxruntime-genai-android-0.4.0-dev.aar`)
2. Model files:
   - Main model file (quantized): `phi-3.5-mini-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx`
   - Tokenizer config file: `tokenizer.json`

## Model Location for Testing

The model files should be placed in:
```
/sdcard/phi-3-model/
```

## Model Setup Steps

### 1. Download the Model Files

1. Download the Phi-3.5-mini-instruct model files from Microsoft or HuggingFace
2. Ensure you have both the quantized ONNX model file and the tokenizer configuration

### 2. Push Model Files to Android Device

```powershell
# Create directory on device
adb shell mkdir -p /sdcard/phi-3-model

# Push model files to device
adb push path\to\phi-3.5-mini-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx /sdcard/phi-3-model/
adb push path\to\tokenizer.json /sdcard/phi-3-model/

# Verify files were transferred correctly
adb shell ls -la /sdcard/phi-3-model/
```

### 3. Configure Runtime Permissions

Both apps are configured to request necessary storage permissions at runtime. Make sure to:
1. Grant storage permissions when prompted
2. For Android 10+, ensure the app can access the external storage path

## Important Notes for Production Deployment

For a production Android deployment, the current implementation would need to be modified:

1. **Model Distribution**: 
   - Package smaller models with the app using the assets folder
   - Implement a download manager for larger models with progress tracking and error handling
   - Consider cloud-based inference for very large models

2. **Asset Management**:
   - Store models in the app's private storage area for better security
   - Implement model versioning and updates

3. **Permissions**:
   - Follow the principle of least privilege
   - Consider scoped storage for Android 10+ compatibility

## App Workflow

1. The AIApp loads the model in `ModelManager.kt` using the ONNX Runtime GenAI library
2. The InferenceService exposes an AIDL interface for remote model inference
3. The UIApp binds to this service and sends prompts for processing

## Usage Flow

1. Install both the AIApp and UIApp on your Android device
2. Push the model files to `/sdcard/phi-3-model/` as described above
3. Launch the AIApp first and start the service (grant permissions when prompted)
4. Launch the UIApp which will connect to the AI service
5. Enter prompts in the UIApp and receive model responses

## Testing Strategy

### Local Testing (Desktop)

Before deploying to a device, you can test the model locally using:

1. **JUnit Tests**: Run `ModelManagerLocalTest.kt` to verify model loading and inference
2. **Standalone Java App**: Use `ModelTest.java` for interactive testing without Android dependencies
3. **Helper Scripts**:
   - `test_model_local.ps1`: Verifies model files and updates test configurations
   - `push_model.ps1`: Pushes verified model files to an Android device

For detailed instructions, see `local_testing_guide.md`.

### Device Testing

For testing on Android devices:

1. First test with the emulator by placing the model in a location accessible by the app
2. Use a smaller quantized model version for testing on real devices
3. Implement proper error handling for cases where the model can't be loaded

## Model Performance Considerations

- The Phi-3.5-mini-instruct-generic-cpu model is optimized for CPU inference
- Consider monitoring device temperature and battery usage during inference
- Implement timeouts for long-running inference operations (already implemented in UIApp with a 30-second timeout)
- Add user feedback for inference progress, especially for longer prompts

### Performance Monitoring

We've added performance monitoring features to the application:

1. **Memory Usage Tracking**: The `MemoryMonitor` utility class tracks memory usage during model loading and inference.
2. **Token Generation Rate**: The ModelManager logs tokens per second during generation.
3. **Request Latency**: Both apps track and log request timing information.
4. **Error Statistics**: The InferenceService tracks success and failure rates for inference requests.

To access these metrics:

1. View the Android logcat output with the tags:
   - `ModelManager` - For model loading and inference performance
   - `MemoryMonitor` - For detailed memory usage statistics
   - `InferenceService` - For service-level metrics and error rates
   - `UIApp` - For user-facing request timings

## Troubleshooting

If you encounter issues:

### Model Loading Errors

Check that:
1. All model files are properly pushed to the correct location
2. The app has necessary storage permissions
3. Look at the AIApp's logs for specific error messages

### Service Connection Issues

Check that:
1. The AIApp service is running
2. The intent filter and action names match between both apps
3. The AIDL interface definition is identical in both apps

### Inference Errors

Check that:
1. The prompt format is correct for the Phi-3.5 model
2. There's enough memory available for model inference
3. The model parameters (temperature, maxLength, etc.) are reasonable

## References

- [ONNX Runtime GenAI Documentation](https://onnxruntime.ai/docs/genai/)
- [Microsoft Phi-3.5 Model Documentation](https://huggingface.co/microsoft/phi-3.5-mini-instruct-4k)
- [Android AIDL Documentation](https://developer.android.com/develop/background-work/services/aidl)
