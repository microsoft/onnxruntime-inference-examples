#import "ViewController.h"
#include "ort_genai_c.h"
#include "ort_genai.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    NSString *llmPath = [[NSBundle mainBundle] resourcePath];
    char const *modelPath = llmPath.cString;
//
    auto model =  OgaModel::Create(modelPath);
//
    auto tokenizer = OgaTokenizer::Create(*model);
//
    const char* prompt = "<|system|>You are a helpful AI assistant.<|end|><|user|>Can you introduce yourself?<|end|><|assistant|>";
//
    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt, *sequences);
//
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 200);
    params->SetInputSequences(*sequences);
//

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
     }
}
@end
