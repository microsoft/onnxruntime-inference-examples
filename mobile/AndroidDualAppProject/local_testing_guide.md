# Local Model Testing Guide

This guide provides instructions for testing the Phi-3.5 model locally before deploying to an Android device.

## Prerequisites

1. Download the Phi-3.5 model files:
   - ONNX model file: `phi-3.5-mini-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx`
   - Tokenizer configuration: `tokenizer.json`
   - You can download these from the Microsoft/Hugging Face repositories

2. Required libraries:
   - ONNX Runtime GenAI Java library (`onnxruntime-genai-1.16.1.jar` or newer)
   - JUnit and Mockito for unit testing

## Test Methods Available

This project provides multiple ways to test the model locally:

### 1. Standalone Java Application

The `ModelTest.java` file contains a standalone Java application that can run without Android dependencies.

To run:
```powershell
# First, update the MODEL_PATH variable in the file or use the test_model_local.ps1 script
# Compile and run with the required libraries in classpath
javac -cp "path\to\onnxruntime-genai.jar" ModelTest.java
java -cp ".;path\to\onnxruntime-genai.jar" com.example.modeltest.ModelTest
```

This application provides a simple command-line interface to test prompts against the model.

### 2. JUnit Tests

The project includes `ModelManagerLocalTest.kt` which contains JUnit tests for the `ModelManager` class.

These tests check:
- Model initialization
- Text generation
- Performance metrics

To run:
- Open the project in Android Studio
- Right-click on `ModelManagerLocalTest.kt` and select 'Run ModelManagerLocalTest'
- Or run from command line with: `./gradlew AIApp:testDebugUnitTest --tests "com.example.aiapp.ModelManagerLocalTest"`

### 3. Helper Scripts

#### Test Model Locally

The `test_model_local.ps1` script helps you:
- Verify your model files
- Update test files with the correct model path
- Provide guidance on next steps

```powershell
.\test_model_local.ps1 -modelDir "C:\path\to\model\directory"
```

#### Push Model to Device

Once local testing passes, use the `push_model.ps1` script to push your models to an Android device:

```powershell
.\push_model.ps1 -modelDir "C:\path\to\model\directory"
```

## Troubleshooting Local Tests

### Memory Issues
If you encounter memory issues during testing, try:
```powershell
# For Java application
java -Xmx4g -cp ".;path\to\onnxruntime-genai.jar" com.example.modeltest.ModelTest

# For Gradle tests
./gradlew AIApp:testDebugUnitTest -Dorg.gradle.jvmargs="-Xmx4g"
```

### Library Not Found
Ensure the ONNX Runtime GenAI library is properly included:
- For standalone testing: Include in the classpath
- For unit tests: Make sure it's in the build.gradle.kts dependencies

### Model File Issues
If you see errors about model files:
1. Check that both the .onnx file and tokenizer.json are in the same directory
2. Verify file permissions (should be readable)
3. Check that the model is the correct version (Phi-3.5 mini)

## Expected Model Behavior

When testing, you should expect:
- Initial loading time: 5-30 seconds (depending on hardware)
- Token generation rate: ~1-10 tokens per second on a typical CPU
- Memory usage: ~1-2GB during inference
- Sample responses should be coherent and contextually relevant to prompts
