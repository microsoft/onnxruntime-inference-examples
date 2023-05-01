# Introduction
In these notebooks we are demonstrating quantization aware training using Neural Networks Compression Framework and inference of quantized model using OpenVINO Execution provider through Optimum library.
We are using a BERT Model(bert-large-uncased-whole-word-masking-finetuned-squad) with Stanford Question Answering Dataset (SQuAD) dataset from the HuggingFace hub (transformers library) for Question-Answering usecase.

The question answer scenario takes a question and a piece of text called a context, and produces answer to the question extracted from the context. The questions & contexts are tokenized and encoded, fed as inputs into the transformer model. The answer is extracted from the output of the model which is the most likely start and end tokens in the context, which are then mapped back into words.

## Prerequisites

To run on AzureML, you need:
* Azure subscription
* Azure Machine Learning Workspace (see this notebook for creation of the workspace if you do not already have one: [AzureML configuration notebook](https://github.com/Azure/MachineLearningNotebooks/blob/56e0ebc5acb9614fac51d8b98ede5acee8003820/configuration.ipynb))
* the Azure Machine Learning SDK
* the Azure CLI and the Azure Machine learning CLI extension (> version 2.2.2)


## Quantization Aware Training using Neural Networks Compression Framework 
 - The training notebook is to demonstrate quantization aware training using Neural Networks Compression Framework [NNCF](https://github.com/AlexKoff88/nncf_pytorch/tree/ak/qdq_per_channel) through  
 [Optimum Intel](https://github.com/huggingface/optimum-intel/tree/v1.5.2).  
- The output of training is an INT8 optimized model and will stored on a local/cloud storage.  
- Inorder to run Quantization Aware Training we need to provide few arguments as an input to the training script like model name, dataset name etc.  
  <br/>For more details please refer to the training notebook.

## Inference on quantized model using OpenVINO Execution provider through Optimum ONNX Runtime
- In the Inference notebook we will be using the Finetuned model for inference.    
- Options to run inference on single and multiple inputs:  
  
    - In case of multiple input, we need to provide input csv file. The sample csv file is available [here](https://github.com/intel/nlp-training-and-inference-openvino/blob/v1.1/question-answering-bert-qat/onnxovep_optimum_inference/data/input.csv).  
    After running inference on input csv the corresponding outputs will be written onto output.csv. 
    - In case of single input, variables i.e., context and question are used.
      These variables are passed as an argument to the inference script. If they are empty strings, then the default behaviour is to read the input csv file and run inference.  

  <br/>For more details please refer to the inference notebook.
