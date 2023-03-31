// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORTQuestionAnsweringPerformer_h
#define ORTQuestionAnsweringPerformer_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface ORTQuestionAnsweringPerformer : NSObject

+ (nullable NSString*)performQuestionAnsweringWithError:(NSError**)error;

@end

NS_ASSUME_NONNULL_END

#endif /* ORTQuestionAnsweringPerformer_h */
