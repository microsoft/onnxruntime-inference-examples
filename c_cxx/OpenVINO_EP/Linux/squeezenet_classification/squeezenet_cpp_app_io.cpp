/*
Copyright (C) 2021, Intel Corporation
SPDX-License-Identifier: Apache-2.0

Portions of this software are copyright of their respective authors and released under the MIT license:
- ONNX-Runtime-Inference, Copyright 2020 Lei Mao. For licensing see https://github.com/leimao/ONNX-Runtime-Inference/blob/main/LICENSE.md
*/

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept> // To use runtime_error
#include <CL/cl2.hpp>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

struct OpenCL {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;

    explicit OpenCL(std::shared_ptr<std::vector<cl_context_properties>> media_api_context_properties = nullptr) {
        // get Intel iGPU OCL device, create context and queue
        {
            const unsigned int refVendorID = 0x8086;
            cl_uint n = 0;
            clGetPlatformIDs(0, NULL, &n);

            // Get platform list
            std::vector<cl_platform_id> platform_ids(n);
            clGetPlatformIDs(n, platform_ids.data(), NULL);

            for (auto& id : platform_ids) {
                cl::Platform platform = cl::Platform(id);
                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    if (refVendorID == d.getInfo<CL_DEVICE_VENDOR_ID>()) {
                        _device = d;
                        _context = cl::Context(_device);
                        break;
                    }
                }
            }
            cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            _queue = cl::CommandQueue(_context, _device, props);
        }
    }

    explicit OpenCL(cl_context context) {
        // user-supplied context handle
        _context = cl::Context(context);
        _device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0]);

        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        _queue = cl::CommandQueue(_context, _device, props);
    }
};

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Function to validate the input image file extension.
bool imageFileExtension(std::string str)
{
  // is empty throw error
  if (str.empty())
    throw std::runtime_error("[ ERROR ] The image File path is empty");

  size_t pos = str.rfind('.');
  if (pos == std::string::npos)
    return false;

  std::string ext = str.substr(pos+1);

  if (ext == "jpg" || ext == "jpeg" || ext == "gif" || ext == "png" || ext == "jfif" || 
        ext == "JPG" || ext == "JPEG" || ext == "GIF" || ext == "PNG" || ext == "JFIF") {
            return true;
  }

  return false;
}

// Function to read the labels from the labelFilepath.
std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

// Function to validate the input model file extension.
bool checkModelExtension(const std::string& filename)
{
    if(filename.empty())
    {
        throw std::runtime_error("[ ERROR ] The Model file path is empty");
    }
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "onnx")
        return true;
    return false;
}

// Function to validate the Label file extension.
bool checkLabelFileExtension(const std::string& filename)
{
    size_t pos = filename.rfind('.');
    if (filename.empty())
    {
        throw std::runtime_error("[ ERROR ] The Label file path is empty");
    }
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "txt") {
        return true;
    } else {
        return false;
    }
}

//Handling divide by zero
float division(float num, float den){
   if (den == 0) {
      throw std::runtime_error("[ ERROR ] Math error: Attempted to divide by Zero\n");
   }
   return (num / den);
}

void printHelp() {
    std::cout << "To run the model, use the following command:\n";
    std::cout << "Example: ./run_squeezenet <path_to_the_model> <path_to_the_image> <path_to_the_classes_file>" << std::endl;
    std::cout << "\n Example: ./run_squeezenet squeezenet1.1-7.onnx demo.jpeg synset.txt \n" << std::endl;
}

int main(int argc, char* argv[])
{

    if(argc == 2) {
        std::string option = argv[1];
        if (option == "--help" || option == "-help" || option == "--h" || option == "-h") {
            printHelp();
        }
        return 0;
    } else if(argc != 4) {
        std::cout << "[ ERROR ] you have used the wrong command to run your program." << std::endl;
        printHelp();
        return 0;
    }

    std::string instanceName{"image-classification-inference"};

    std::string modelFilepath = argv[1]; // .onnx file

    //validate ModelFilePath
    checkModelExtension(modelFilepath);
    if(!checkModelExtension(modelFilepath)) {
        throw std::runtime_error("[ ERROR ] The ModelFilepath is not correct. Make sure you are setting the path to an onnx model file (.onnx)");
    }
    std::string imageFilepath = argv[2];

    // Validate ImageFilePath
    imageFileExtension(imageFilepath);
    if(!imageFileExtension(imageFilepath)) {
        throw std::runtime_error("[ ERROR ] The imageFilepath doesn't have correct image extension. Choose from jpeg, jpg, gif, png, PNG, jfif");
    }
    std::ifstream f(imageFilepath.c_str());
    if(!f.good()) {
        throw std::runtime_error("[ ERROR ] The imageFilepath is not set correctly or doesn't exist");
    }

    // Validate LabelFilePath
    std::string labelFilepath = argv[3];
    if(!checkLabelFileExtension(labelFilepath)) {
        throw std::runtime_error("[ ERROR ] The LabelFilepath is not set correctly and the labels file should end with extension .txt");
    }

    std::vector<std::string> labels{readLabels(labelFilepath)};

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);

    auto ocl_instance = std::make_shared<OpenCL>();

    //Appending OpenVINO Execution Provider API
    // Using OPENVINO backend
    OrtOpenVINOProviderOptions options;
    options.device_type = "GPU_FP32"; //Another option is: GPU_FP16
    options.context = (void *) ocl_instance->_context.get() ; 
    std::cout << "OpenVINO device type is set to: " << options.device_type << std::endl;
    sessionOptions.AppendExecutionProvider_OpenVINO(options);
    
    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations ( Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    //Creation: The Ort::Session is created here
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo info_gpu("OpenVINO_GPU", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    auto inputNodeName = session.GetInputNameAllocated(0, allocator);
    const char* inputName = inputNodeName.get();
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    auto outputNodeName = session.GetOutputNameAllocated(0, allocator);
    const char* outputName = outputNodeName.get();
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;
    //pre-processing the Image
    // step 1: Read an image in HWC BGR UINT8 format.
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

    // step 2: Resize the image.
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(inputDims.at(3), inputDims.at(2)),
               cv::InterpolationFlags::INTER_CUBIC);

    // step 3: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    // step 4: Convert the image to HWC RGB float format by dividing each pixel by 255.
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    // step 5: Split the RGB channels from the image.   
    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    //step 6: Normalize each channel.
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;

    //step 7: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);

    // step 8: Convert the image to CHW RGB float format.
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    
    //Run Inference
    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};

    /* To run inference using ONNX Runtime, the user is responsible for creating and managing the 
    input and output buffers. The buffers are IO Binding Buffers created on Remote Folders to create Remote Blob*/

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());

    size_t imgSize = inputTensorSize*4;
    cl_int err;
    cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imgSize, NULL, &err);
    {
        void *buffer = (void *)preprocessedImage.ptr();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imgSize, buffer);
    }

    //To pass to OrtValue wrap the buffer in shared buffer
    void *shared_buffer_void = static_cast<void *>(&shared_buffer);
    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);
 
    Ort::Value inputTensors = Ort::Value::CreateTensor(
        info_gpu, shared_buffer_void, imgSize, inputDims.data(),
        inputDims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    assert(("Output tensor size should equal to the label set size.",
            labels.size() == outputTensorSize));
    
    size_t imgSizeO = outputTensorSize*4;
    cl::Buffer shared_buffer_out(ocl_instance->_context, CL_MEM_READ_WRITE, imgSizeO, NULL, &err);
   
   //To pass the ORT Value wrap the output buffer in shared buffer
    void *shared_buffer_out_void = static_cast<void *>(&shared_buffer_out);
       Ort::Value outputTensors = Ort::Value::CreateTensor(
        info_gpu, shared_buffer_out_void, imgSizeO, outputDims.data(),
        outputDims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    std::cout << "Before Running\n";
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensors, 1, outputNames.data(), &outputTensors, 1);

    int predId = 0;
    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;

    uint8_t *ary = (uint8_t*) malloc(imgSizeO);
    ocl_instance->_queue.enqueueReadBuffer(shared_buffer_out, true, 0, imgSizeO, ary);

    float outputTensorArray[outputTensorSize] ;

    std::memcpy(outputTensorArray, ary, imgSizeO);
    /* The inference result could be found in the buffer for the output tensors, 
    which are usually the buffer from std::vector instances. */

    for (int i = 0; i < labels.size(); i++) {
        activation = outputTensorArray[i];
        expSum += std::exp(activation);
        if (activation > maxActivation)
        {
            predId = i;
            maxActivation = activation;
        }
    }
    std::cout << "Predicted Label ID: " << predId << std::endl;
    std::cout << "Predicted Label: " << labels.at(predId) << std::endl;
    float result;
    try {
      result = division(std::exp(maxActivation), expSum);
      std::cout << "Uncalibrated Confidence: " << result << std::endl;
    }
    catch (std::runtime_error& e) {
      std::cout << "Exception occurred" << std::endl << e.what();
    }

    // Measure latency
    int numTests{10};
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    //Run: Running the session is done in the Run() method:
    for (int i = 0; i < numTests; i++) {
       // session.Run(Ort::RunOptions{nullptr}, binding);
      session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensors, 1, outputNames.data(), &outputTensors, 1);
      //session.Run(Ort::RunOptions{nullptr}, binding);
    }
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "Minimum Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / static_cast<float>(numTests)
              << " ms" << std::endl;
    return 0;
}