// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef GenAIGenerator_h
#define GenAIGenerator_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface GenAIGenerator : NSObject

- (void)generate:(NSString *)input_user_question;

@end

NS_ASSUME_NONNULL_END

#endif /* GenAIGenerator_h */
