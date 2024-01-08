Step-by-Step
============

This example confirms INT4 weight only quantization (WOQ) accuracy for Llama-2 models on [lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai).

As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that can meet the computational demands of these modern architectures while maintaining the accuracy. Compared to normal quantization like W8A8, weight only quantization is probably a better trade-off to balance the performance and the accuracy.

Two weight only algorithms are provided in this example. Round-to-nearest (RTN) is the most straightforward way to quantize weight using scale maps. GPTQ algorithm provides more accurate quantization but requires more computational resources.

# Prerequisite

## 1. Environment
```shell
SKIP_RUNTIME=True pip install -r requirements.txt
```

## 2. Prepare Model

Note that this README.md uses meta-llama/Llama-2-7b-hf as an example. There are other models available that can be used for INT4 weight only quantization. The following table shows a few models' configurations:

| Model | Num Hidden Layers| Num Attention Heads | Hidden Size |
| --- | --- | --- | --- |
| [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) | 32 | 32 | 4096 |
| [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | 32 | 32 | 4096 |
| [meta-llama/Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b) | 40 | 40 | 5120 |
| [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | 40 | 40 | 5120 |
| [meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b) | 80 | 64 | 8192 |
| [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | 80 | 64 | 8192 |

Export to ONNX model:
```bash
optimum-cli export onnx --model meta-llama/Llama-2-7b-hf --task text-generation-with-past ./Llama-2-7b-hf
```

> Note: require `optimum>=1.14.0`.

# Run

## 1. Quantization

```bash
bash run_quant.sh --model_input=/folder/of/model \ # folder path of onnx model, config and tokenizer
                  --model_output=/folder/of/quantized/model \ # folder path to save onnx model
                  --algorithm=RTN \ # or GPTQ
```

## 2. Benchmark

```bash
bash run_benchmark.sh --model_input=/folder/of/model \ # folder path of onnx model, config and tokenizer
                      --tasks=lambada_openai
```


