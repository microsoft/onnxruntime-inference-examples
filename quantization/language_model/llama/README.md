Step-by-Step
============

This folder contains example code for quantizing LLaMa model.

# Prerequisite

## 1. Environment
```shell
pip install -r requirements.txt
```

## 2. Prepare Model

```bash
optimum-cli export onnx --model decapoda-research/llama-7b-hf --task causal-lm-past ./llama_7b
optimum-cli export onnx --model decapoda-research/llama-13b-hf --task causal-lm-past ./llama_13b
```

# Run

## 1. Quantization

```bash
bash run_quantization.sh --input_model=/path/to/model \ # folder path of onnx model
                         --output_model=/path/to/model_tune \ # folder path to save onnx model
                         --batch_size=batch_size # optional \
                         --dataset NeelNanda/pile-10k \
                         --alpha 0.6 \ # 0.6 for llama-7b, 0.8 for llama-13b
                         --quant_format="QOperator" # or QDQ, optional
```

## 2. Benchmark

Accuracy:

```bash
bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                      --batch_size=batch_size \ # optional 
                      --mode=accuracy \
                      --tasks=lambada_openai
```

Performance:
```bash
numactl -m 0 -C 0-3 bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                                          --mode=performance \
                                          --batch_size=batch_size # optional \
                                          --intra_op_num_threads=4
```
