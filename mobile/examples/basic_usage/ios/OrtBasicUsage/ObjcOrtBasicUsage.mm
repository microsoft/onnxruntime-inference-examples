// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ObjcOrtBasicUsage.h"

#import <Foundation/Foundation.h>

#import <onnxruntime.h>

static_assert(__has_feature(objc_arc), "ARC must be enabled.");

NS_ASSUME_NONNULL_BEGIN

@implementation ObjcOrtBasicUsage

+ (nullable ORTValue*)CreateOrtValueForFloat:(float*)fp error:(NSError**)error {
  // `data` will hold the memory of the input ORT value.
  // We set it to refer to the memory of the given float (*fp).
  NSMutableData* data = [[NSMutableData alloc] initWithBytes:fp length:sizeof(float)];
  // This will create a value with a tensor with the given float's data, of
  // type float, and with shape [1].
  ORTValue* ortValue = [[ORTValue alloc] initWithTensorData:data
                                                elementType:ORTTensorElementDataTypeFloat
                                                      shape:@[ @1 ]
                                                      error:error];
  return ortValue;
}

+ (nullable NSString*)GetAddModelPathWithError:(NSError**)error {
  // There are two model formats:
  // - The standard ONNX format, only supported in a full ORT build
  //   (onnxruntime-objc Pod).
  // - ORT format, supported by minimal and full ORT builds
  //   (onnxruntime-mobile-objc and onnxruntime-objc Pods).
  // Define the ORT_BASIC_USAGE_USE_ONNX_FORMAT_MODEL symbol to use the ONNX
  // format model.
  NSString* const path =
#if defined(ORT_BASIC_USAGE_USE_ONNX_FORMAT_MODEL)
      [NSBundle.mainBundle pathForResource:@"single_add" ofType:@"onnx"];
#else
      [NSBundle.mainBundle pathForResource:@"single_add" ofType:@"ort"];
#endif

  if (!path) {
    if (error) {
      *error = [NSError errorWithDomain:@"ObjcOrtBasicUsage.ErrorDomain"
                                   code:1
                               userInfo:@{NSLocalizedDescriptionKey : @"Failed to get model path."}];
    }
    return nil;
  }

  return path;
}

+ (nullable NSNumber*)AddA:(NSNumber*)aNumber B:(NSNumber*)bNumber error:(NSError**)error {
  // We will run a simple model which adds two floats.
  // The inputs are named `A` and `B` and the output is named `C` (A + B = C).
  // All inputs and outputs are float tensors with shape [1].
  NSString* const addModelPath = [ObjcOrtBasicUsage GetAddModelPathWithError:error];
  if (!addModelPath) return nil;

  // Regarding error handling:
  // If an error occurs, ORT APIs will return `nil` or `NO` and set the
  // optional NSError** parameter if provided.
  // Here, we will use the NSError** value from the caller and check the return
  // value of each ORT API, returning nil on error.

  // First, we create the ORT environment.
  // The environment is required in order to create an ORT session.
  // ORTLoggingLevelWarning should show us only important messages.
  ORTEnv* ortEnv = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning error:error];
  if (!ortEnv) return nil;

  float a = aNumber.floatValue, b = bNumber.floatValue;

  // Next, we will create some ORT values for our input tensors. We have two
  // floats, `a` and `b`.
  // See CreateOrtValueForFloat:error for details.
  ORTValue* aInputValue = [ObjcOrtBasicUsage CreateOrtValueForFloat:&a error:error];
  if (!aInputValue) return nil;
  ORTValue* bInputValue = [ObjcOrtBasicUsage CreateOrtValueForFloat:&b error:error];
  if (!bInputValue) return nil;

  // Now, we will create an ORT session to run our model.
  // One can configure session options with a session options object
  // (ORTSessionOptions).
  // We use the default options with sessionOptions:nil.
  ORTSession* session = [[ORTSession alloc] initWithEnv:ortEnv modelPath:addModelPath sessionOptions:nil error:error];
  if (!session) return nil;

  // With a session and input values, we have what we need to run the model.
  // We provide a dictionary mapping from input name to value and a set of
  // output names.
  // This run method will run the model, allocating the output(s), and return
  // them in a dictionary mapping from output name to value.
  // As with session creation, it is possible to configure run options with a
  // run options object (ORTRunOptions).
  // We use the default options with runOptions:nil.
  NSDictionary<NSString*, ORTValue*>* outputs = [session runWithInputs:@{@"A" : aInputValue, @"B" : bInputValue}
                                                           outputNames:[NSSet setWithArray:@[ @"C" ]]
                                                            runOptions:nil
                                                                 error:error];
  if (!outputs) return nil;

  // After running the model, we will get the output.
  ORTValue* cOutputValue = outputs[@"C"];

  // We know the output value is a float tensor with shape [1] so we will just
  // access it directly.
  NSData* cData = [cOutputValue tensorDataWithError:error];
  if (!cData) return nil;

  // Since we called run without pre-allocated outputs, ORT owns the output
  // values. We must not access an output value's memory after it is
  // deinitialized. So, we will copy the data here.
  float c;
  memcpy(&c, cData.bytes, sizeof(float));

  return [NSNumber numberWithFloat:c];
}

@end

NS_ASSUME_NONNULL_END
