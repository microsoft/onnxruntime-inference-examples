// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORTSuperResolutionPerformer_h
#define ORTSuperResolutionPerformer_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface ORTSuperResolutionPerformer : NSObject

+ (nullable UIImage*)performSuperResolutionWithError:(NSError**)error;

@end

NS_ASSUME_NONNULL_END

#endif
