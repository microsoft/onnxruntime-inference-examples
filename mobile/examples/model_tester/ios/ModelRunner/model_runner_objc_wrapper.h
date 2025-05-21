#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * This class is an Objective-C wrapper around the C++ `model_runner::RunConfig` structure.
 */
@interface ModelRunnerRunConfig : NSObject

- (void)setModelPath:(NSString*)modelPath;

- (void)setNumIterations:(NSUInteger)numIterations;

- (void)setExecutionProvider:(NSString*)providerName
                     options:(nullable NSDictionary<NSString*, NSString*>*)providerOptions;

@end

/**
 * This class is an Objective-C wrapper around the C++ model runner functions.
 */
@interface ModelRunner : NSObject

+ (nullable NSString*)runWithConfig:(ModelRunnerRunConfig*)config
                              error:(NSError**)error NS_SWIFT_NAME(run(config:));

@end

NS_ASSUME_NONNULL_END
