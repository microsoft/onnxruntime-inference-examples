# Evaluating ORT Causal Large Language Models (LLMs)

This folder contains an implementation example to evaluate the performance and accuracy of ORT LLMs. 
- Performance evaluations measure complete runtime on different input/output lengths.
- Accuracy evaluations are based on benchmarks from the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), a unified framework to test generative language models on a large number of different evaluation tasks.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Accuracy Tasks](#tasks)

## Getting Started

To get started with the evaluation, follow these steps:

1. Clone the repository
2. Navigate to the parent folder and install the required dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare ONNX model using optimum-cli:
   ```bash
   optimum-cli export onnx --model decapoda-research/llama-7b-hf --task causal-lm-with-past --for-ort --device cpu llama-7b-onnx   
   ```

## Usage
- Evaluate performance for ONNX Causal model:
   ```bash
   python main.py --model_args pretrained=llama-7b-onnx \
                  --perf_batch 10 \
                  --mode perf
   ```
   You can also get the model time profile by using the --profiling flag.

- Evaluate accuracy for ONNX Causal model:
   ```bash
   python main.py --model_args pretrained=llama-7b-onnx \
                  --device cpu \
                  --tasks hendrycksTest-marketing,lambada_openai,arc_easy \
                  --mode acc
   ```

Additional arguments can be provided to the model constructor using the `--model_args` flag.

## Evaluation Metrics

During the evaluation, the following metrics can measured:
### For Performance Evaluations
- **Per Token Cost**: Inferred runtime for model to generate each output token.

- **Prompt Cost**: Inferred runtime for model to process input prompts.

### For Accuracy Evaluations 
- **Accuracy**: The proportion of correct predictions over the total number of samples.

- **Perplexity**: Calculated based on the probability distribution assigned by the model to each token in a sequence. The lower the perplexity value, the better the language model's performance.

## Accuracy Tasks

Available tasks to run can be found in the [lm-evaluation-harness tool](https://github.com/EleutherAI/lm-evaluation-harness/blob/4fbbd60fa3573dcbf61eb79492f772adeb969157/lm_eval/tasks/__init__.py#L99) hosted by EleutherAI



