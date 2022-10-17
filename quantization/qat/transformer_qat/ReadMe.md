# **Bert model QAT and onnxruntime GPU support**

## **Introduction**

This bert_imdb_finetune_qat example is to show how to:
* Do QAT training with bert model
* Export QAT trainned model to onnx model with Q/DQ operators
* Optimize the exported Q/DQ model to run with onnxruntime CUDA execution provider in int8 mode.

The base fp32 part of the sample is based on following document on web by Fabio Chiusano: 
[BERT Finetuning with Hugging Face and Training Visualizations with TensorBoard](https://medium.com/nlplanet/bert-finetuning-with-hugging-face-and-training-visualizations-with-tensorboard-46368a57fc97)

## **Pre-requisite**

Note that transfomrers and torch version are important for the sample to run correctly.
* Huggingface transformers 4.23.0
* Torch version 1.8.2 cuda
* Other deps like onnx, onnxrntime-gpu 1.13 or newer, datasets, sklearn, etc.

## **finetune, export and evaluation in float 32**

Following command will download huggingface bert uncased model and using imdb sentiment dataset to fine tune.
And export fp32 onnx model. All output will be saved under ../BertModel/ .
Evaluation of the torch and onnx model are included for comparision purpose.

```console
python bert_imdb_finetune_qat.py --do_fp32_all
```

## **finetune, export and evaluation with QAT**

Following command will use the fp32 model finetuned above, quantize it, and do quantize aware training.
Evalute the QAT model, then export onnx model with Q/DQ.

```console
python bert_imdb_finetune_qat.py --do_qat_fine_tune --do_qat_eval --do_qat_export --do_qat_onnx_eval
```

## **optimize the qdq model for onnxruntime cuda**

onnxruntime 1.13 contains transformers optimizer for int8 Q/DQ model. To fuse the qdq onnx model, run:
```console
python -m onnxruntime.transformers.optimizer  --num_heads 12 --input ../BertModel/bert-base-uncased-finetuned-imdb-qat.onnx --output ../BertModel/bert-base-uncased-finetuned-imdb-int8.onnx
```

To run and evalute the optimized int8 onnxruntime model,
```console
python bert_imdb_finetune_qat.py --do_qat_onnx_eval --qat_opt_onnx_model ../BertModel/bert-base-uncased-finetuned-imdb-int8.onnx
```
as you may encounter conflict as torch and onnxruntime here require different cuda, we suggest you creat seperate environment that contains torch cpu with onnxruntime-gpu to run the last command above.

## **How to QAT a Bert Model**

Basically, all QAT components are in the files qat_bert.py and qat_utils.py, including:
+ Fakeq() is applied on specifictensor, for huggingface, many basic module are modified, like
    * BertOutput,
    * BertSelfOutput,
    * BertSelfAttention,
    * BertIntermediate,
+ those modified module have from_float() method to convert from original fp32 module
+ module mapping are specified when quantize the model
+ minor changes in the main python file could get the qat model:

```python
from qat_bert import quantize_bert_model
from qat_utils import remove_qconfig_for_module
def remove_qconfig_before_convert(model):
    # Do not quantize subgraph that not in the bert part
    remove_qconfig_for_module(model, '', ['classifier', 'bert.pooler'], remove_subtree=True)

model = quantize_bert_model(model, remove_qconfig_before_convert)
```

## **QAT a Bert Model in your project**
* Reuse the qat_bert.py and qat_utils.py. in case the transformer version is not same, modifications may needed to those two files.
* Similar minor change as showed above in the main training python file could make the work done.
