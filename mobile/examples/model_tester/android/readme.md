setup instructions:

1. manually set up native onnxruntime libraries. unzip onnxruntime-android.aar, then put shared libraries and headers into these directories:

under mobile/examples/model_tester/android/app/src/main/cpp:
include/onnxruntime_session_options_config_keys.h
include/nnapi_provider_factory.h
include/cpu_provider_factory.h
include/onnxruntime_lite_custom_op.h
include/onnxruntime_run_options_config_keys.h
include/onnxruntime_float16.h
include/onnxruntime_cxx_inline.h
include/onnxruntime_cxx_api.h
include/onnxruntime_c_api.h
lib/armeabi-v7a/libonnxruntime.so
lib/x86/libonnxruntime.so
lib/arm64-v8a/libonnxruntime.so
lib/x86_64/libonnxruntime.so

2. copy an onnx model file to mobile/examples/model_tester/android/app/src/main/res/raw/model.onnx
