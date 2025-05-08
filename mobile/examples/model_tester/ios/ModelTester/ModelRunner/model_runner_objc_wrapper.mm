#import "model_runner_objc_wrapper.h"

#include "model_runner.h"

@implementation ModelRunner

+ (nullable NSString*)runWithModelPath:(NSString*)modelPath
                         numIterations:(uint32_t)numIterations
                                 error:(NSError**)error {
  try {
    model_runner::RunConfig config{};
    config.model_path = modelPath.UTF8String;
    config.num_iterations = numIterations;
    config.num_warmup_iterations = 1;

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
