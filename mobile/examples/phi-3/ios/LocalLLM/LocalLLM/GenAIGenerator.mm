// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "GenAIGenerator.h"
#include "LocalLLM-Swift.h"
#include "ort_genai.h"
#include "ort_genai_c.h"
#include <chrono>

@implementation GenAIGenerator

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> TimePoint;

+ (void)generate:(nonnull NSString*)input_user_question {
    NSLog(@"Starting token generation...");
    
    NSString* llmPath = [[NSBundle mainBundle] resourcePath];
    const char* modelPath = llmPath.cString;
    
    // Log model creation
    NSLog(@"Creating model ...");
    auto model = OgaModel::Create(modelPath);
    if (!model) {
        NSLog(@"Failed to create model.");
        return;
    }
    
    NSLog(@"Creating tokenizer...");
    auto tokenizer = OgaTokenizer::Create(*model);
    if (!tokenizer) {
        NSLog(@"Failed to create tokenizer.");
        return;
    }
    
    auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
    
    // Construct the prompt
    NSString* promptString = [NSString stringWithFormat:@"<|user|>\n%@<|end|>\n<|assistant|>", input_user_question];
    const char* prompt = [promptString UTF8String];
    
    NSLog(@"Encoding prompt...");
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt, *sequences);
    
    // Log parameters
    NSLog(@"Setting generator parameters...");
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 200);
    params->SetInputSequences(*sequences);
    
    NSLog(@"Creating generator...");
    auto generator = OgaGenerator::Create(*model, *params);
    
    bool isFirstToken = true;
    TimePoint startTime = Clock::now();
    TimePoint firstTokenTime;
    int tokenCount = 0;
    
    NSLog(@"Starting token generation loop...");
    while (!generator->IsDone()) {
        generator->ComputeLogits();
        generator->GenerateNextToken();
        
        if (isFirstToken) {
            NSLog(@"First token generated.");
            firstTokenTime = Clock::now();
            isFirstToken = false;
        }
        
        // Get the sequence data
        const int32_t* seq = generator->GetSequenceData(0);
        size_t seq_len = generator->GetSequenceCount(0);
        
        // Decode the new token
        const char* decode_tokens = tokenizer_stream->Decode(seq[seq_len - 1]);
        
        // Check for decoding failure
        if (!decode_tokens) {
            NSLog(@"Token decoding failed.");
            break;
        }
        
        NSLog(@"Decoded token: %s", decode_tokens);
        tokenCount++;
        
        // Convert token to NSString and update UI on the main thread
        NSString* decodedTokenString = [NSString stringWithUTF8String:decode_tokens];
        [SharedTokenUpdater.shared addDecodedToken:decodedTokenString];
    }


    TimePoint endTime = Clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    auto firstTokenDuration = std::chrono::duration_cast<std::chrono::milliseconds>(firstTokenTime - startTime).count();
    
    NSLog(@"Token generation completed. Total time: %lld ms, First token time: %lld ms, Total tokens: %d", totalDuration, firstTokenDuration, tokenCount);

    NSDictionary *stats = @{
        @"totalTime": @(totalDuration),
        @"firstTokenTime": @(firstTokenDuration),
        @"tokenCount": @(tokenCount)
    };

    // notify main thread that token generation is complete 
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:@"TokenGenerationCompleted" object:nil];
        [[NSNotificationCenter defaultCenter] postNotificationName:@"TokenGenerationStats" object:nil userInfo:stats];
    });
    NSLog(@"Token generation completed.");
}

@end
