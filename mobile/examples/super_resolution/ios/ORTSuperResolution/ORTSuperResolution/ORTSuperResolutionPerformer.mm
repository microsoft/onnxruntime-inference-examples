// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ORTSuperResolutionPerformer.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_extensions.h>


@implementation ORTSuperResolutionPerformer

+ (nullable UIImage*)performSuperResolutionWithError:(NSError **)error {
    
    UIImage* output_image = nil;
    
    try {
        
        // Register custom ops
        
        const auto ort_log_level = ORT_LOGGING_LEVEL_INFO;
        auto ort_env = Ort::Env(ort_log_level, "ORTSuperResolution");
        auto session_options = Ort::SessionOptions();
        
        if (RegisterCustomOps(session_options, OrtGetApiBase()) != nullptr) {
            throw std::runtime_error("RegisterCustomOps failed");
        }
        
        // Step 1: Load model
        
        NSString *model_path = [NSBundle.mainBundle pathForResource:@"pytorch_superresolution_with_pre_post_processing_opset18"
                                                             ofType:@"onnx"];
        if (model_path == nullptr) {
            throw std::runtime_error("Failed to get model path");
        }
        
        // Step 2: Create Ort Inference Session
        
        auto sess = Ort::Session(ort_env, [model_path UTF8String], session_options);
        
        // Read input image
        
        // note: need to set Xcode settings to prevent it from messing with PNG files:
        // in "Build Settings":
        // - set "Compress PNG Files" to "No"
        // - set "Remove Text Metadata From PNG Files" to "No"
        NSString *input_image_path =
        [NSBundle.mainBundle pathForResource:@"cat_224x224" ofType:@"png"];
        if (input_image_path == nullptr) {
            throw std::runtime_error("Failed to get image path");
        }
        
        // Step 3: Prepare input tensors and input/output names
        
        NSMutableData *input_data =
        [NSMutableData dataWithContentsOfFile:input_image_path];
        const int64_t input_data_length = input_data.length;
        const auto memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        
        const auto input_tensor = Ort::Value::CreateTensor(memoryInfo, [input_data mutableBytes], input_data_length,
                                                           &input_data_length, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
        
        constexpr auto input_names = std::array{"image"};
        constexpr auto output_names = std::array{"image_out"};
        
        // Step 4: Call inference session run
        
        const auto outputs = sess.Run(Ort::RunOptions(), input_names.data(),
                                      &input_tensor, 1, output_names.data(), 1);
        if (outputs.size() != 1) {
            throw std::runtime_error("Unexpected number of outputs");
        }
        
        // Step 5: Analyze model outputs
        
        const auto &output_tensor = outputs.front();
        const auto output_type_and_shape_info = output_tensor.GetTensorTypeAndShapeInfo();
        const auto output_shape = output_type_and_shape_info.GetShape();
        
        if (const auto output_element_type =
            output_type_and_shape_info.GetElementType();
            output_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            throw std::runtime_error("Unexpected output element type");
        }
        
        const uint8_t *output_data_raw = output_tensor.GetTensorData<uint8_t>();
        
        // Step 6: Convert raw bytes into NSData and return as displayable UIImage
        
        NSData *output_data = [NSData dataWithBytes:output_data_raw length:(output_shape[0])];
        output_image = [UIImage imageWithData:output_data];
        
    } catch (std::exception &e) {
        NSLog(@"%s error: %s", __FUNCTION__, e.what());
        
        static NSString *const kErrorDomain = @"ORTSuperResolution";
        constexpr NSInteger kErrorCode = 0;
        if (error) {
            NSString *description =
            [NSString stringWithCString:e.what() encoding:NSASCIIStringEncoding];
            *error =
            [NSError errorWithDomain:kErrorDomain
                                code:kErrorCode
                            userInfo:@{NSLocalizedDescriptionKey : description}];
        }
        return nullptr;
    }
    
    if (error) {
        *error = nullptr;
    }
    return output_image;
}

@end
