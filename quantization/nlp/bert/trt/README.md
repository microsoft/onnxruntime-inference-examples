# BERT QDQ Quantization in ONNX for TensorRT  
There are two main steps for the quantization:
1. Calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model
2. Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model

After Qunatization is done, you can evaluate the QDQ model by running evaluation script we provide or your own method.

The **e2e_tensorrt_bert_example.py** is an end-to-end example for you to reference and run.

## Requirements
* ONNX Runtime 1.10+ ([ORT Python GPU Package](https://pypi.org/project/onnxruntime-gpu/) includes TensorRT and CUDA from ORT1.10 or you can manually [build](https://onnxruntime.ai/docs/build/eps.html#tensorrt)) 
* Python 3+
* numpy 
* The onnx model used in the script is converted from Hugging Face BERT model. https://huggingface.co/transformers/serialization.html#converting-an-onnx-model-using-the-transformers-onnx-package
* We use [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset as default dataset which is included in the repo. If you want to use other dataset to do calibration/evaluation, please either follow the format of squad/dev-1.1.json to add dataset or you have to write your own pre-processing method to parse your dataset.

Some utility functions for dataset processing, data reader and evaluation are from Nvidia TensorRT demo BERT repo,
https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT
## Model Calibration
Before running the calibration, please set the configuration properly in order to get better performance.
* **op_type_to_quantize** : Default is ['MatMul', 'Add']. One thing to remember is that even though quantizing more nodes improves inference latency, it can result in significant accuracy drop. So we don't suggest quantize all op type in the model.
* **sequence_lengths** and **doc_stride** : Please always consider them together. In order to get better accuracy result, if use sequence length 384 then choose doc stride 128. if use sequence length 128 then choose doc stride 32. Generally speaking larger sequence_lengths and doc_stride can have better accuracy.
* **calib_num** : Default is 100. It's the number of examples in dataset used for calibration.

## QDQ Model Generation
In order to get best performance from TensorRT, there are some optimizations being done when inserting Q/DQ nodes to the model.
* When inserting QDQ nodes to 'Add' node, only insert QDQ nodes to 'Add' node which is followed by ReduceMean node
* Enable per channel quantization on 'MatMul' node's weights. Please see QDQQuantizer(...) in the e2e example, the per_channel argument should be True as well as 'QDQOpTypePerChannelSupportToAxis': {'MatMul': 1} should be specified in extra_options argument. You can also modify 'QDQOpTypePerChannelSupportToAxis' to other op types and channel axis if they can increase performance.

Once QDQ model generation is done, the qdq_model.onnx will be saved.

## QDQ Model Evaluation
Remember to set env variables, ORT_TENSORRT_FP16_ENABLE=1 and ORT_TENSORRT_INT8_ENABLE=1, to run QDQ model.
We use evaluation tool from Nvidia TensorRT demo BERT repo to evaluate the result based on SQuAD v1.0 and SQuAD v2.0.

Note: The input names of model in the e2e example is based on Hugging Face Model's naming. If model input names are not correct in your model, please modify the code ort_session.run(["output_start_logits","output_end_logits"], inputs) in the example.
