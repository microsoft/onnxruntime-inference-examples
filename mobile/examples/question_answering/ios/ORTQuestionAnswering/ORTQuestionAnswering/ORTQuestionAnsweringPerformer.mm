// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ORTQuestionAnsweringPerformer.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_extensions.h>


@implementation ORTQuestionAnsweringPerformer


+ (nullable NSString *)performQuestionAnswering:(NSString *)input_user_question context:(NSString *)input_article error:(NSError **)error{
    
    NSString *output_text = nil;
    
    try {
        
        // Register custom ops
        
        const auto ort_log_level = ORT_LOGGING_LEVEL_INFO;
        auto ort_env = Ort::Env(ort_log_level, "ORTQuestionAnswering");
        auto session_options = Ort::SessionOptions();
        
        if (RegisterCustomOps(session_options, OrtGetApiBase()) != nullptr) {
            throw std::runtime_error("RegisterCustomOps failed");
        }
        
        // Step 1: Load model
        
        NSString *model_path = [NSBundle.mainBundle pathForResource:@"csarron_mobilebert_uncased_squad_v2_quant_with_pre_post_processing"
                                                             ofType:@"onnx"];
        if (model_path == nullptr) {
            throw std::runtime_error("Failed to get model path");
        }
        
        // Step 2: Create Ort Inference Session
        
        auto sess = Ort::Session(ort_env, [model_path UTF8String], session_options);
        
        // Step 3: Prepare input tensors and input/output names
        
        std::vector<int64_t> input_dims{1, 2};
        Ort::AllocatorWithDefaultOptions ortAllocator;
        
        auto input_tensor = Ort::Value::CreateTensor(ortAllocator, input_dims.data(), input_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
        
        std::vector<std::string> question_article_vec;
        question_article_vec.reserve(2);
        question_article_vec.push_back([input_user_question UTF8String]);
        question_article_vec.push_back([input_article UTF8String]);
        
        std::vector<const char*> p_str;
        for (const auto& s : question_article_vec) {
            p_str.push_back(s.c_str());
        }
        
        input_tensor.FillStringTensor(p_str.data(), p_str.size());
        
        constexpr auto input_names = std::array{"input_text"};
        constexpr auto output_names = std::array{"text"};
        
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
            output_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
            throw std::runtime_error("Unexpected output element type");
        }
        
        // Step 6: Set output answer text
        
        const std::string* output_string_raw = output_tensor.GetTensorData<std::string>();
        output_text = [NSString stringWithCString:output_string_raw->c_str() encoding:NSUTF8StringEncoding];
        
    } catch (std::exception &e) {
        NSLog(@"%s error: %s", __FUNCTION__, e.what());
        
        static NSString *const kErrorDomain = @"ORTQuestionAnswering";
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
    return output_text;
}

@end
