# BERT QDQ Quantization for TensorRT  
There are two main steps for the quantization:
1. Calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model
2. Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model

After Qunatization is done, you can evaluate the QDQ model by running evaluation script we provide or your own method.

The e2e_tensorrt_bert_example.py is an end-to-end example for you to reference and run.

## Requirements
* ONNX runtime 1.10+
* numpy
* The onnx model used in the script is converted from Hugging Face BERT model. https://huggingface.co/transformers/serialization.html#converting-an-onnx-model-using-the-transformers-onnx-package
* We use SQuAD dataset as default dataset which is included in the repo. If you want to use other dataset to do calibration/evaluation, please either follow the format of squad/dev-1.1.json or you have to write your own pre-processing method.
## Model Calibration
Before running the calibration, please set the configuration properly in order to get better performance.
* **op_type_to_quantize** : Default is ['MatMul', 'Add']. One thing to remember is that even though quantizing more nodes improves inference latency, it can result in significant accuracy drop. So we don't suggest quantize all op type in the model.
* **sequence_lengths** and **doc_stride** : Please always consider them together. In order to get better accuracy result, if use sequence length 384 then choose doc stride 128. if use sequence length 128 then choose doc stride 32. Generally speaking larger sequence_lengths and doc_stride can have better accuracy.
* **calib_num** : Default is 100. It's the number of examples in dataset used for calibration.
