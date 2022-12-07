# Introduction
 In these notebooks we are demonstrating quantization aware training using Neural Networks Compression Framework and run inference on quantized model using OpenVINO Execution provider.
Here we're using a BERT Model(bert-large-uncased-whole-word-masking-finetuned-squad) from the HuggingFace hub (transformers library) for Question-Answering usecase.

In the following sections, we use the HuggingFace Bert model trained with Stanford Question Answering Dataset (SQuAD) dataset as an example. The Bert model is quantized with NNCF Quantize Aware Training.

The question answer scenario takes a question and a piece of text called a context, and produces answer to the question extracted from the context. The questions & contexts are tokenized and encoded, fed as inputs into the transformer model. The answer is extracted from the output of the model which is the most likely start and end tokens in the context, which are then mapped back into words.

## Prerequisites

To run on AzureML, you need:
* Azure subscription
* Azure Machine Learning Workspace (see this notebook for creation of the workspace if you do not already have one: [AzureML configuration notebook](https://github.com/Azure/MachineLearningNotebooks/blob/56e0ebc5acb9614fac51d8b98ede5acee8003820/configuration.ipynb))
* the Azure Machine Learning SDK
* the Azure CLI and the Azure Machine learning CLI extension (> version 2.2.2)


## Quantization Aware Training using Neural Networks Compression Framework 
 - The training notebook is to demonstrate quantization aware training using Neural Networks Compression Framework [NNCF](https://github.com/AlexKoff88/nncf_pytorch/tree/ak/qdq_per_channel) through  
 OpenVINO™ Integration with [Optimum](https://github.com/huggingface/optimum-intel/tree/v1.5.2).  
- The output of the training is the INT8 optimized model and will stored on a local/cloud storage.  
- To run the Quantization Aware Training we need to pass certain arguments to the script like model name, dataset name etc.  
For more details please refer to the training notebook.

## Inference on quantized model using OpenVINO Execution provider
- In the Inference notebook we are using above Finetuned model for inferencing using OpenVINO Execution provider.    
- We have options to run inference on single and multiple inputs.  
- In case of multiple input, we need to provide input csv file. The sample csv file is available [here](https://github.com/intel/nlp-training-and-inference-openvino/blob/bert_qa_azureml/question-answering-bert-qat/onnxovep_optimum_inference/data/input.csv).  
This file will be read and inference will be performed and the corresponding outputs will be saved as outputs.csv.  
- In case of single input, the variables context and question are used to facilitate this.  
  These variables are passed as an argument to the inference script. If they are empty strings, then the default behaviour is to read the input csv file and run inference.  
For more details please refer to the inference notebook.
