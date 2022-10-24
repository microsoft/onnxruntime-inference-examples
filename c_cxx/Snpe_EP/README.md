# Image classification with Inception v3 in CPP using SNPE Execution Provider
1.  This image classification sample uses the Inception v3 model from [SNPE turtorial](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_inceptionv3.html) with DLC format. Wrap it as a custom node in an ONNX model to inference with SNPE Execution Provider.

2.  The sample uses the Onnxruntime SNPE Execution Provider to run inference on various Qualcomm devices like Qualcomm CPU, GPU DSP, AIP, etc. It supports Windows ARM64, and Android.

# Prerequisites
1. Setup a Linux environment by [WSL2](https://learn.microsoft.com/en-us/windows/wsl/)
2. Download SNPE SDK from Qualcomm's developer site [here](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)

3. Setup SNPE on the Linux environment (WSL2). Setup environment for the tutorial. Follow the Tutorials and Examples for [Tensorflow inceptionv3](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_inceptionv3.html)
4. Get the model [Inception v3](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_setup.html#tutorial_setup_inception_v3)

    ```
    python $SNPE_ROOT/models/inception_v3/scripts/setup_inceptionv3.py -a ~/tmpdir -d -r dsp
    ```

    The generated DLC file inception_v3.dlc and inception_v3_quantized.dlc can be found at $SNPE_ROOT/models/inception_v3/dlc/.

    The data chairs.raw, notice_sign.raw, plastic_cup.raw and trash_bin.raw can be found at $SNPE_ROOT/models/inception_v3/data/. The sample applicatioin use these raw file as input.

5. Create ONNX model from DLC file


    Create a script gen_onnx_model.py in folder $SNPE_ROOT/models/inception_v3 with the code below:

    ```
    import onnx
    from onnx import helper
    from onnx import TensorProto

    with open('./dlc/inception_v3_quantized.dlc','rb') as file:
        file_content = file.read()

    input1 = helper.make_tensor_value_info('input:0', TensorProto.FLOAT, [1, 299, 299, 3])
    output1 = helper.make_tensor_value_info('InceptionV3/Predictions/Reshape_1:0', TensorProto.FLOAT, [1, 1001])

    snpe_node = helper.make_node('Snpe', name='Inception v3', inputs=['input:0'], outputs=['InceptionV3/Predictions/Reshape_1:0'], DLC=file_content, snpe_version='1.61.0', target_device='DSP', notes='quantized dlc model.', domain='com.microsoft')

    graph_def = helper.make_graph([snpe_node], 'Inception_v3', [input1], [output1])
    model_def = helper.make_model(graph_def, producer_name='tesorflow', opset_imports=[helper.make_opsetid('', 13)])
    onnx.save(model_def, 'snpe_inception_v3.onnx')
    ```

    Run the script to generate the Onnx model with the DLC content embed in a Onnx node.
    Change the data type for input & output from TensorProto.FLOAT to TensorProto.UINT8 if you want to run model inference with quantized data. It's helpful to save the bandwidth (expecially for application with CS mode, data need to be transmitted across the network). Also need to change the options_values for buffer_type from FLOAT to TF8 in the main.cpp.
    ```
    std::vector<const char*> options_keys = {"runtime", "buffer_type"};
    std::vector<const char*> options_values = {"CPU", "TF8"}; // set to TF8 if use quantized data

    g_ort->SessionOptionsAppendExecutionProvider(session_options, "SNPE", options_keys.data(),
                                                 options_values.data(), options_keys.size());
    ```
    Please refers to the unit test case [Snpe_ConvertFromAbs.QuantizedModelTf8Test](https://github.com/microsoft/onnxruntime/blob/5ecfaef042380995fb15587ccf6ff77f9d3a01d2/onnxruntime/test/contrib_ops/snpe_op_test.cc#L209-L251) for more details.

# How to build

## Windows
1. [Build Onnxruntime with SNPE SNPE Execution Provider](https://onnxruntime.ai/docs/execution-providers/SNPE-ExecutionProvider.html)
    ```
    build.bat --use_snpe --snpe_root=[location-of-SNPE_SDK] --build_shared_lib --cmake_generator "Visual Studio 16 2019" --skip_submodule_sync --config Release --build_dir \build\Windows
    ```

2. Build the sample application
    ```
    cmake.exe -S . -B build\ -G "Visual Studio 16 2019" -DONNXRUNTIME_ROOTDIR=[location-of-Onnxruntime]
    ```

    build snpe_ep_sample.sln with x64 platform to run on host without Qualcomm NPU, build with ARM64 platform to run on device with Qualcomm NPU.

3. Run the sample
    Copy files below to folder which has snpe_ep_sample.exe
    onnxruntime.dll -- from Onnxruntime build folder
    SNPE.dll -- from $SNPE_ROOT/lib
    chairs.raw -- from $SNPE_ROOT/models/inception_v3/data/cropped
    imagenet_slim_labels.txt -- from $SNPE_ROOT/models/inception_v3/data

    Run
    ```
    snpe_ep_sample.exe --cpu chairs.raw
    ```

    it will output:

    ```
    832, 0.299591, studio couch
    ```

## Android
1. [Build Onnxruntime with SNPE SNPE Execution Provider](https://onnxruntime.ai/docs/execution-providers/SNPE-ExecutionProvider.html)
    ```
    build.bat --build_shared_lib --skip_submodule_sync --android --config Release --use_snpe --snpe_root [location-of-SNPE_SDK] --android_sdk_path [location-of-android_SDK] --android_ndk_path [location-of-android_NDK] --android_abi arm64-v8a --android_api [api-version] --cmake_generator Ninja --build_dir build\Android
    ```

2. Build the sample application

    ```
    cmake.exe -S . -B build_android\ -G Ninja -DONNXRUNTIME_ROOTDIR=[location-of-Onnxruntime] -DCMAKE_TOOLCHAIN_FILE=[location-of-android_NDK\build\cmake\android.toolchain.cmake] -DANDROID_PLATFORM=android-27 -DANDROID_MIN_SDK=27 -DANDROID_ABI=arm64-v8a
    cmake.exe --build build_android\ --config Release
    ```

3. Run sample on Android device with DSP/HTP
    Follow instruction [Running on Android using DSP Runtime](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_inceptionv3.html)

    push file to Android device
    ```
    adb shell "mkdir /data/local/tmp/snpeexample"
    adb push [$SNPE_ROOT]/lib/aarch64-android-clang6.0/*.so /data/local/tmp/snpeexample
    adb push [$SNPE_ROOT]/lib/dsp/*.so /data/local/tmp/snpeexample
    adb push [$Onnxruntime_ROOT]/build/Android/Release/libonnxruntime.so /data/local/tmp/snpeexample    
    adb push [$SNPE_ROOT]/models/inception_v3/data/cropped/chairs.raw /data/local/tmp/snpeexample
    adb push [$SNPE_ROOT]/models/inception_v3/data/imagenet_slim_labels.txt /data/local/tmp/snpeexample
    adb push [$SNPE_ROOT]/models/inception_v3/snpe_inception_v3.onnx /data/local/tmp/snpeexample
    adb push ./onnxruntime-inference-examples/c_cxx/Snpe_EP/build_android/snpe_ep_sample /data/local/tmp/snpeexample
    ```

    Run sample

    ```
    adb shell
    cd /data/local/tmp/snpeexample
    chmod +x *
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample
    export PATH=$PATH:/data/local/tmp/snpeexample
    snpe_ep_sample --cpu chairs.raw
    snpe_ep_sample --dsp chairs.raw
    ```

    it will output:
    ```
    832, 0.299591, studio couch
    ```