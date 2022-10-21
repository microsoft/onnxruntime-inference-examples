# Image classification with VGG16 in C# using SNPE Execution Provider
1.  This image classification sample uses the VGG16 model from [SNPE turtorial](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_onnx.html) with DLC format. Wrap it as a custom node in an ONNX model to inference with SNPE Execution Provider.

2.  The sample uses the Onnxruntime SNPE Execution Provider to run inference on various Qualcomm devices like Qualcomm CPU, DSP, etc. It supports Windows ARM64.

# Prerequisites
1. Setup a Linux environment by [WSL2](https://learn.microsoft.com/en-us/windows/wsl/)
2. Download SNPE SDK from Qualcomm's developer site [here](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/windows-on-snapdragon). Developer needs to request access to the package.

3. Setup SNPE on the Linux environment (WSL2). Setup environment for the tutorial. Follow the Tutorials and Examples for [ONNX VGG](https://developer.qualcomm.com/sites/default/files/docs/snpe/tutorial_onnx.html)
4. Get the model
    Run command below to get the VGG16 Onnx model.
    ```
    cd $SNPE_ROOT/models/VGG
    wget https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx
    ```

    Offload the Softmax from post-processing to model inference (on NPU). Copy add_softmax.py to $SNPE_ROOT/models/VGG/onnx folder and run it, to apply Softmax node to model output.

    Follow step 3 ~ 5 to generate the DLC file vgg16.dlc. 
	
	Runn command below to generate the quantized DLC file vgg16_q.dlc.
    ```
	cd $SNPE_ROOT/models/VGG
	snpe-dlc-quantize --input_dlc dlc/vgg16.dlc --output_dlc dlc/vgg16_q.dlc --input_list data/cropped/raw_list.txt
    ```
	
    The generated DLC file vgg16.dlc and vgg16_q.dlc can be found at $SNPE_ROOT/models/VGG/dlc/.

    The data kitten.raw can be found at $SNPE_ROOT/models/VGG/data/cropped. The synset.txt can be found at $SNPE_ROOT/models/VGG/data. The sample application use this raw file as input.

5. Create ONNX model from DLC file

    Run script WrapDLCintoOnnx.py in folder $SNPE_ROOT/models/VGG/dlc to generate the Onnx model with the DLC content embed in a Onnx node.
	
	Sample code to enable SNPE ExecutionProvider:
    ```
    var options = new SessionOptions { LogId = "SnpeImageClassificationSample" };

    // Add SNPE EP
    var providerOptions = new Dictionary<string, string>();
    providerOptions.Add("runtime", "DSP"); // CPU, DSP
    providerOptions.Add("buffer_type", "FLOAT");
    options.AppendExecutionProvider("SNPE", providerOptions);

    using var session = new InferenceSession(modelFilePath, options);
    ```

# Build & Run

## Windows
1. Install [.NET 6.0](https://dotnet.microsoft.com/download/dotnet/6.0) or higher and download nuget. Install SDK on build machine, install Arm64 runtime on target device.
2. Install Microsoft.ML.OnnxRuntime.Snpe nuget package from [nuget.org](https://www.nuget.org/)
   Open image_classification.csproj with Visual Studio. Right click on the solution and click Restore Nuget Packages if the nuget is not installed.

3. Build the sample application
    
    build image_classification project with x64 platform to run without Qualcomm NPU, build with ARM64 platform to run on device with Qualcomm NPU.

4. Run the sample
    Copy files below to folder which has image_classification.exe
    onnxruntime.dll -- from Onnxruntime build folder
    SNPE.dll and other dll if exist -- from $SNPE_ROOT/lib
    *.so -- from $SNPE_ROOT/lib/lib, this is required for DSP inference
    kitten.raw -- from $SNPE_ROOT/models/VGG/data/cropped
    synset.txt -- from $SNPE_ROOT/models/VGG/data

    Run
    ```
    image_classification.exe vgg16_dlc_q.onnx kitten.raw synset.txt
    ```

    it will output:

    ```
	vgg16_image_classification>image_classification.exe vgg16_dlc_q.onnx kitten.raw synset.txt
	probability=0.3443572 ; class=n02123045 tabby, tabby cat
	probability=0.3175425 ; class=n02124075 Egyptian cat
	probability=0.3175425 ; class=n02124075 Egyptian cat
	probability=0.0127016995 ; class=n02127052 lynx, catamount
	probability=0.0028225998 ; class=n02129604 tiger, Panthera tigris
    ```
