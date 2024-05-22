// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "GenAIGenerator.h"
#include "ort_genai_c.h"
#include "ort_genai.h"
#include "LocalLLM-Swift.h"

@implementation GenAIGenerator

+ (void)generate:(nonnull NSString *)input_user_question {

    NSString *llmPath = [[NSBundle mainBundle] resourcePath];
    char const *modelPath = llmPath.cString;

    auto model =  OgaModel::Create(modelPath);

    auto tokenizer = OgaTokenizer::Create(*model);

    NSString *promptString = [NSString stringWithFormat:@"<|user|>\n%@<|end|>\n<|assistant|>", input_user_question];
    const char* prompt = [promptString UTF8String];
    
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt, *sequences);

    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 200);
    params->SetInputSequences(*sequences);

//  Streaming Output to generate token by token
    auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

    auto generator = OgaGenerator::Create(*model, *params);

     while (!generator->IsDone()) {
         generator->ComputeLogits();
         generator->GenerateNextToken();

         const int32_t* seq = generator->GetSequenceData(0);
         size_t seq_len = generator->GetSequenceCount(0);
         const char* decode_tokens = tokenizer_stream->Decode(seq[seq_len-1]);
         
         NSLog(@"Decoded tokens: %s", decode_tokens);
         
//      Add decoded token to SharedTokenUpdater
        NSString *decodedTokenString = [NSString stringWithUTF8String:decode_tokens];
        [SharedTokenUpdater.shared addDecodedToken:decodedTokenString];
         
//      Introduce a delay to allow the UI to update
        [NSThread sleepForTimeInterval:0.1]; // Adjust the delay as needed
     }
}
@end
