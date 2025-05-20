#import "model_runner_objc_wrapper.h"

#include "model_runner.h"

NS_ASSUME_NONNULL_BEGIN

@interface ModelRunnerRunConfig ()

- (const model_runner::RunConfig&)cppRunConfig;

@end

@implementation ModelRunnerRunConfig {
  model_runner::RunConfig _runConfig;
}

- (void)setModelPath:(nonnull NSString*)modelPath {
  _runConfig.model_path = modelPath.UTF8String;
}

- (void)setNumIterations:(NSUInteger)numIterations {
  _runConfig.num_iterations = static_cast<size_t>(numIterations);
}

- (void)setExecutionProvider:(NSString*)providerName
                     options:(nullable NSDictionary<NSString*, NSString*>*)providerOptions {
  model_runner::RunConfig::EpConfig ep_config{};
  ep_config.provider_name = providerName.UTF8String;
  if (providerOptions != nil) {
    for (NSString* optionName in providerOptions) {
      NSString* optionValue = providerOptions[optionName];
      ep_config.provider_options.emplace(optionName.UTF8String,
                                         optionValue.UTF8String);
    }
  }
  _runConfig.ep = std::move(ep_config);
}

- (const model_runner::RunConfig&)cppRunConfig {
  return _runConfig;
}

@end

@implementation ModelRunner

+ (nullable NSString*)runWithConfig:(ModelRunnerRunConfig*)objcConfig
                              error:(NSError**)error {
  try {
    const auto& config = [objcConfig cppRunConfig];

    auto result = model_runner::Run(config);

    auto summary = model_runner::GetRunSummary(config, result);

    return [NSString stringWithUTF8String:summary.c_str()];
  } catch (const std::exception& e) {
    if (error) {
      NSString* description = [NSString stringWithCString:e.what()
                                                 encoding:NSUTF8StringEncoding];

      *error = [NSError errorWithDomain:@"ModelRunner"
                                   code:0
                               userInfo:@{NSLocalizedDescriptionKey : description}];
    }
    return nil;
  }
}

@end

NS_ASSUME_NONNULL_END
