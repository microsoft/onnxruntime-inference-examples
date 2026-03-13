// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#import "GenAIGenerator.h"
#include <chrono>
#include <vector>
#include "LocalLLM-Swift.h"
#include "ort_genai.h"
#include "ort_genai_c.h"

const size_t kMaxTokens = 200;

@interface GenAIGenerator () {
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaTokenizer> tokenizer;
  NSString* modelPath;
}
@end

@implementation GenAIGenerator

typedef std::chrono::steady_clock Clock;
typedef std::chrono::time_point<Clock> TimePoint;

- (instancetype)init {
  self = [super init];
  if (self) {
    self->model = nullptr;
    self->tokenizer = nullptr;
  }
  return self;
}

- (void)setModelFolderPath:(NSString*)modelPath {
  @synchronized(self) {
    self->modelPath = [modelPath copy];
    NSLog(@"Model folder path set to: %@", modelPath);
    
    try {
      [self loadModelFromPath];
    } catch (const std::exception& e) {
      NSString* errorMessage = [NSString stringWithUTF8String:e.what()];
      NSLog(@"Error loading model: %@", errorMessage);
      
      // Notify the UI about the error
      NSDictionary* errorInfo = @{@"error" : errorMessage};
      dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:@"GenAIError" object:nil userInfo:errorInfo];
      });
    }
  }
}

- (void)loadModelFromPath {
  @synchronized(self) {
    NSLog(@"Creating model...");
    self->model = OgaModel::Create(self->modelPath.UTF8String);  // throws exception
    NSLog(@"Creating tokenizer...");
    self->tokenizer = OgaTokenizer::Create(*self->model);  // throws exception
  }
}

- (void)generate:(nonnull NSString*)input_user_question {
  std::vector<long long> tokenTimes;  // per-token generation times
  tokenTimes.reserve(kMaxTokens);
  TimePoint startTime, firstTokenTime, tokenStartTime;

  try {
    if (!self->modelPath) {
      self->modelPath = [[NSBundle mainBundle] resourcePath];
      NSLog(@"No folder path provided. Using the default folder path: %@", self->modelPath);
      [self loadModelFromPath];
    }

    NSLog(@"Starting token generation...");
    auto tokenizer_stream = OgaTokenizerStream::Create(*self->tokenizer);

    // Construct the prompt
    NSString* promptString = [NSString stringWithFormat:@"<|user|>\n%@<|end|>\n<|assistant|>", input_user_question];
    const char* prompt = [promptString UTF8String];

    // Encode the prompt
    auto sequences = OgaSequences::Create();
    self->tokenizer->Encode(prompt, *sequences);

    size_t promptTokensCount = sequences->SequenceCount(0);

    NSLog(@"Setting generator parameters...");
    auto params = OgaGeneratorParams::Create(*self->model);
    params->SetSearchOption("max_length", kMaxTokens);

    auto generator = OgaGenerator::Create(*self->model, *params);

    bool isFirstToken = true;
    NSLog(@"Starting token generation loop...");

    startTime = Clock::now();
    generator->AppendTokenSequences(*sequences);
    while (!generator->IsDone()) {
      tokenStartTime = Clock::now();
      generator->GenerateNextToken();

      if (isFirstToken) {
        firstTokenTime = Clock::now();
        isFirstToken = false;
      }

      // Get the sequence data and decode the token
      const int32_t* seq = generator->GetSequenceData(0);
      size_t seq_len = generator->GetSequenceCount(0);
      const char* decode_tokens = tokenizer_stream->Decode(seq[seq_len - 1]);

      if (!decode_tokens) {
        throw std::runtime_error("Token decoding failed.");
      }

      // Measure token generation time excluding logging
      TimePoint tokenEndTime = Clock::now();
      auto tokenDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tokenEndTime - tokenStartTime).count();
      tokenTimes.push_back(tokenDuration);
      NSString* decodedTokenString = [NSString stringWithUTF8String:decode_tokens];
      [SharedTokenUpdater.shared addDecodedToken:decodedTokenString];
    }

    TimePoint endTime = Clock::now();
    // Log token times
    NSLog(@"Per-token generation times: %@", [self formatTokenTimes:tokenTimes]);

    // Calculate metrics
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    auto firstTokenDuration = std::chrono::duration_cast<std::chrono::milliseconds>(firstTokenTime - startTime).count();

    double promptProcRate = (double)promptTokensCount * 1000.0 / firstTokenDuration;
    double tokenGenRate = (double)(tokenTimes.size() - 1) * 1000.0 / (totalDuration - firstTokenDuration);

    NSLog(@"Token generation completed. Total time: %lld ms, First token time: %lld ms, Total tokens: %zu",
          totalDuration, firstTokenDuration, tokenTimes.size());
    NSLog(@"Prompt tokens: %zu, Prompt Processing Rate: %f tokens/s", promptTokensCount, promptProcRate);
    NSLog(@"Generated tokens: %zu, Token Generation Rate: %f tokens/s", tokenTimes.size(), tokenGenRate);

    NSDictionary* stats = @{@"tokenGenRate" : @(tokenGenRate), @"promptProcRate" : @(promptProcRate)};
    // notify main thread that token generation is complete
    dispatch_async(dispatch_get_main_queue(), ^{
      [[NSNotificationCenter defaultCenter] postNotificationName:@"TokenGenerationStats" object:nil userInfo:stats];
      [[NSNotificationCenter defaultCenter] postNotificationName:@"TokenGenerationCompleted" object:nil];
    });

    NSLog(@"Token generation completed.");

  } catch (const std::exception& e) {
    NSString* errorMessage = [NSString stringWithUTF8String:e.what()];
    NSLog(@"Error during generation: %@", errorMessage);

    // Send error to the UI
    NSDictionary* errorInfo = @{@"error" : errorMessage};
    dispatch_async(dispatch_get_main_queue(), ^{
      [[NSNotificationCenter defaultCenter] postNotificationName:@"GenAIError" object:nil userInfo:errorInfo];
    });
  }
}

// Utility function to format token times for logging
- (NSString*)formatTokenTimes:(const std::vector<long long>&)tokenTimes {
  NSMutableString* formattedTimes = [NSMutableString string];
  for (size_t i = 0; i < tokenTimes.size(); i++) {
    [formattedTimes appendFormat:@"%lld ms, ", tokenTimes[i]];
  }
  return [formattedTimes copy];
}

@end
