// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>
#import <React/RCTLog.h>
#import "MobileNetDataHandler.h"
#import "OnnxruntimeModule.h"
#import "TensorHelper.h"

NS_ASSUME_NONNULL_BEGIN

static NSDictionary* CocoClasses = @{
  @1: @"person",
  @2: @"bicycle",
  @3: @"car",
  @4: @"motorcycle",
  @5: @"airplane",
  @6: @"bus",
  @7: @"train",
  @8: @"truck",
  @9: @"boat",
  @10: @"traffic light"
};

static OnnxruntimeModule* onnxruntimeModule = nil;
static NSString* onnxruntimeModuleKey = nil;

@implementation MobileNetDataHandler

+(void)initialize {
  onnxruntimeModule = [[OnnxruntimeModule alloc] init];
  NSString* modelPath = [[NSBundle mainBundle] pathForResource:@"ssd_mobilenet_v1.f" ofType:@"onnx"];
  NSFileManager* fileManager = [NSFileManager defaultManager];
  if ([fileManager fileExistsAtPath:modelPath]) {
    onnxruntimeModuleKey = modelPath;
  }
}

RCT_EXPORT_MODULE(MobileNetDataHandler)

// It returns mode path in local device,
// so that onnxruntime is able to load a model using a given path.
RCT_EXPORT_METHOD(getLocalModelPath:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  @try {
    if (onnxruntimeModuleKey != nil) {
      resolve(onnxruntimeModuleKey);
    } else {
      reject(@"mobilenet", @"no such a model", nil);
    }
  }
  @catch(NSException* exception) {
    reject(@"mobilenet", @"no such a model", nil);
  }
}

// It returns image path.
RCT_EXPORT_METHOD(getImagePath:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  @try {
    NSString* imagePath = [[NSBundle mainBundle] pathForResource:@"mobilenet" ofType:@"jpg"];
    NSFileManager* fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:imagePath]) {
      resolve(imagePath);
    } else {
      reject(@"mobilenet", @"no such an image", nil);
    }
  }
  @catch(NSException* exception) {
    reject(@"mobilenet", @"no such an image", nil);
  }
}

// It gets raw input data, which can be uri or byte array and others,
// returns cooked data formatted as input of a model.
RCT_EXPORT_METHOD(preprocess:(NSString*)uri
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary* inputDataMap = [self preprocess:uri];
    resolve(inputDataMap);
  }
  @catch(NSException* exception) {
    reject(@"mobilenet", @"can't load an image", nil);
  }
}

// It gets a result from onnxruntime and a duration of session time for input data,
// returns output data formatted as React Native map.
RCT_EXPORT_METHOD(postprocess:(NSDictionary*)result
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary* cookedMap = [self postprocess:result];
    resolve(cookedMap);
  }
  @catch(NSException* exception) {
    reject(@"mobilenet", @"can't pose-process an image", nil);
  }
}

RCT_EXPORT_METHOD(run:(NSString*)input
               output:(NSArray*)output
              options:(NSDictionary*)options
              resolve:(RCTPromiseResolveBlock)resolve
               reject:(RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary* inputMap = [self preprocess:input];
    NSDictionary* resultMap = [onnxruntimeModule run:onnxruntimeModuleKey input:inputMap output:output options:options];
    NSDictionary* outputMap = [self postprocess:resultMap];
    resolve(outputMap);
  }
  @catch(NSException* exception) {
    reject(@"mobilenet", @"can't run a model", nil);
  }
}

-(NSDictionary*)preprocess:(NSString*)uri {
  UIImage *image = [UIImage imageNamed:@"mobilenet.jpg"];
  
  CGImageRef imageRef = [image CGImage];
  NSUInteger width = CGImageGetWidth(imageRef);
  NSUInteger height = CGImageGetHeight(imageRef);
  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

  const NSUInteger rawDataSize = height * width * 4;
  std::vector<unsigned char> rawData(rawDataSize);
  NSUInteger bytesPerPixel = 4;
  NSUInteger bytesPerRow = bytesPerPixel * width;
  CGContextRef context = CGBitmapContextCreate(rawData.data(),
                                               width,
                                               height,
                                               8,
                                               bytesPerRow,
                                               colorSpace,
                                               kCGImageAlphaPremultipliedLast|kCGImageByteOrder32Big);
  CGColorSpaceRelease(colorSpace);
  CGContextSetBlendMode(context, kCGBlendModeCopy);
  CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
  CGContextRelease(context);
  
  const NSInteger channelSize = 3;
  const NSInteger dimSize = height * width * channelSize;
  const NSInteger byteBufferSize = dimSize * sizeof(float);

  unsigned char* byteBuffer = static_cast<unsigned char*>(malloc(byteBufferSize));
  NSData* byteBufferRef = [NSData dataWithBytesNoCopy:byteBuffer length:byteBufferSize];
  float* floatPtr = (float*)[byteBufferRef bytes];
  for (NSUInteger h = 0; h < height; ++h) {
    for (NSUInteger w = 0; w < width; ++w) {
      NSUInteger byteIndex = (bytesPerRow * h) + w * bytesPerPixel;
      *floatPtr++ = rawData[byteIndex];
      *floatPtr++ = rawData[byteIndex + 1];
      *floatPtr++ = rawData[byteIndex + 2];
    }
  }
  floatPtr = (float*)[byteBufferRef bytes];
  
  NSMutableDictionary* inputDataMap = [NSMutableDictionary dictionary];
  
  NSMutableDictionary* inputTensorMap = [NSMutableDictionary dictionary];

  // dims
  NSArray* dims = @[[NSNumber numberWithInt:1],
                    [NSNumber numberWithInt:static_cast<int>(height)],
                    [NSNumber numberWithInt:static_cast<int>(width)],
                    [NSNumber numberWithInt:channelSize]];
  inputTensorMap[@"dims"] = dims;

  // type
  inputTensorMap[@"type"] = JsTensorTypeFloat;

  // encoded data
  NSString* data = [byteBufferRef base64EncodedStringWithOptions:0];
  inputTensorMap[@"data"] = data;
  
  inputDataMap[@"image_tensor:0"] = inputTensorMap;

  return inputDataMap;
}

-(NSDictionary*)postprocess:(NSDictionary*)result {
  NSMutableString* detectionResult = [NSMutableString string];
  NSInteger numDetections = 0;
  
  {
    NSDictionary* outputTensor = [result objectForKey:@"num_detections:0"];

    NSString* data = [outputTensor objectForKey:@"data"];
    NSData* buffer = [[NSData alloc] initWithBase64EncodedString:data options:0];
    float* floatBuffer = (float*)[buffer bytes];
    numDetections = (int)floatBuffer[0];
    
    detectionResult = [NSMutableString stringWithFormat:@"%@Number of detections: %ld\r", detectionResult, numDetections];
  }
  
  {
    NSDictionary* outputTensor = [result objectForKey:@"detection_classes:0"];
    NSMutableArray* dataArray = [NSMutableArray array];

    NSString* data = [outputTensor objectForKey:@"data"];
    NSData* buffer = [[NSData alloc] initWithBase64EncodedString:data options:0];
    float* floatBuffer = (float*)[buffer bytes];
    for (int i = 0; i < numDetections; ++i) {
      int value = static_cast<int>(*floatBuffer++);
      [dataArray addObject:[NSNumber numberWithInt:value]];
    }
    
    NSMutableDictionary* detected = [NSMutableDictionary dictionary];
    for (NSNumber* i in dataArray) {
      NSString* name = [self getCoCoClass:[i integerValue]];
      NSNumber* count = [detected objectForKey:name];
      if (count != nil) {
        detected[name] = [NSNumber numberWithInt:[count intValue] + 1];
      } else {
        detected[name] = @1;
      }
    }
    
    NSEnumerator* enumerator = [detected keyEnumerator];
    id key;
    while ((key = [enumerator nextObject])) {
      detectionResult = [NSMutableString stringWithFormat:@"%@%ld %@ detected\r", detectionResult,
                         [[detected objectForKey:key] integerValue], key];
    }
    detectionResult = [NSMutableString stringWithFormat:@"%@\r", detectionResult];
  }
  
  NSDictionary* cookedMap = @{@"result": detectionResult};
  return cookedMap;
}

- (NSString*)getCoCoClass:(NSInteger)classId {
  NSString* category = [CocoClasses objectForKey:[NSNumber numberWithInteger:classId]];
  if (category != nil) {
    return category;
  } else {
    return @"(not in category)";
  }
}

@end

NS_ASSUME_NONNULL_END
