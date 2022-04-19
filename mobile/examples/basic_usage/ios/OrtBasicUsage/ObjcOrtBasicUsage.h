// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ObjcOrtBasicUsage_h
#define ObjcOrtBasicUsage_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ObjcOrtBasicUsage : NSObject

/**
 * Gets the path to the model used to add two numbers.
 */
+ (nullable NSString*)GetAddModelPathWithError:(NSError**)error;

/**
 * Adds `a` and `b` using ONNX Runtime.
 */
+ (nullable NSNumber*)AddA:(NSNumber*)a B:(NSNumber*)b error:(NSError**)error NS_SWIFT_NAME(add(_:_:));

@end

NS_ASSUME_NONNULL_END

#endif /* ObjcOrtBasicUsage_h */
