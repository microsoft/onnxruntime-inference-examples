# Image classification with VGG16 in C# using SNPE Execution Provider

This sample shows how to use ONNX Runtime with SNPE execution provider to run models optimized for Qualcomm silicon on Windows Arm64 devices. It uses the VGG16 model from [Qualcomm's tutorial](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_onnx.html) and turns it into a SNPE-optiimized ONNX model that can be inferenced with ONNX Runtime with SNPE execution provider.

# Install Qualcomm SDK for Windows
Download the Qualcomm Neural Processing SDK for Windows on Snapdragon from [Qualcomm's developer site](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/windows-on-snapdragon).
   * You currently need to request access to the packages and you will need to create a Qualcomm account to access the packages
   * SNPE v1.61.46 and v1.61.48 have been qualified with ONNX Runtime so far.

# Prepare the SNPE-optimized ONNX file 
Currently the Qualcomm model tools are only runnable in a Linux environment even though their output will be used in Windows.
1. Setup a Linux environment with [WSL2](https://learn.microsoft.com/en-us/windows/wsl/)
2. Download and install the Linux [Qualcomm Neural Processing SDK](https://developer.qualcomm.com/downloads/qualcomm-neural-processing-sdk-ai-v1660) in the WSL2 environment.
   * Note, you will need to login with your Qualcomm account to access.
3. Follow steps 1 and 2 in [Qualcomm's tutorial](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_onnx.html) to setup the environment and download the source model
4. Add post-processing steps to the model
   * Copy add_softmax.py to $SNPE_ROOT/models/VGG/onnx folder and run it to apply Softmax node to model output. 
5. Follow steps 3 through 5 in [Qualcomm's tutorial](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_onnx.html)
   * These will generate a vgg16.dlc file in $SNPE_ROOT/models/VGG/dlc/
6. Apply quantization to the model
   * Run command below to generate the quantized DLC file vgg16_q.dlc.
    ```
	cd $SNPE_ROOT/models/VGG
	snpe-dlc-quantize --input_dlc dlc/vgg16.dlc --output_dlc dlc/vgg16_q.dlc --input_list data/cropped/raw_list.txt
    ```    
7. Create the SNPE-optimized ONNX model
    * Copy the script WrapDLCintoOnnx.py to the $SNPE_ROOT/models/VGG/dlc folder and run it. This will generate an ONNX model with the DLC content embedded.
	
# Build & run the sample app
The sample app uses the ONNX file you created above

1. Install [.NET 6.0](https://dotnet.microsoft.com/download/dotnet/6.0) or higher for Arm64.
2. Install Microsoft.ML.OnnxRuntime.Snpe nuget package from [nuget.org](https://www.nuget.org/)
3. Open image_classification.csproj with Visual Studio. Right click on the solution and click Restore Nuget Packages if the nuget is not installed.
4. Build the sample application, making sure to select the ARM64 platform target.
5. Copy additional files needed to run the sample to the folder that contains image_classification.exe
    * onnxruntime.dll -- from the win-arm64 build folder (for example, vgg16_image_classification\bin\Debug\net6.0\runtimes\win-arm64\nativenet6.0/rutimes/win-arm64/native)
    * SNPE.dll, snpe_dsp_domains_v3.dll -- from $SNPE_ROOT/lib/aarch64-windows-vc19
    * libsnpe_dsp_v68_domains_v3_skel.so -- from $SNPE_ROOT/lib/dsp, this is required for DSP inference
    * kitten.raw -- from $SNPE_ROOT/models/VGG/data/cropped
    * synset.txt -- from $SNPE_ROOT/models/VGG/data
6. Make sure [VC redist arm64](https://aka.ms/vs/17/release/vc_redist.arm64.exe) is installed. You can check the list of installed Apps in Settings to see if it is already there.
   * If it is not installed, you will encounter the error "The application was unable to start correctly" when you try to run the sample app.
7. Run the sample app
    ```
    image_classification.exe vgg16_dlc_q.onnx kitten.raw synset.txt
    ```

    It should output:

    ```
	vgg16_image_classification>image_classification.exe vgg16_dlc_q.onnx kitten.raw synset.txt
	probability=0.3443572 ; class=n02123045 tabby, tabby cat
	probability=0.3175425 ; class=n02124075 Egyptian cat
	probability=0.3175425 ; class=n02124075 Egyptian cat
	probability=0.0127016995 ; class=n02127052 lynx, catamount
	probability=0.0028225998 ; class=n02129604 tiger, Panthera tigris
    ```

    This indicates the model has been successfully run by ONNX Runtime with SNPE execution provider on the Qualcomm NPU.
