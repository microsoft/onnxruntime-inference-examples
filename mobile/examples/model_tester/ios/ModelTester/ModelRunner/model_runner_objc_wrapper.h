#import <Foundation/Foundation.h>
#include <stdint.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * This class is an Objective-C wrapper around the C++ model runner functionality.
 */
@interface ModelRunner : NSObject

+ (nullable NSString*)runWithModelPath:(NSString*)modelPath
                         numIterations:(uint32_t)numIterations
                                 error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
