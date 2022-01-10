# BERT QDQ Quantization in ONNX for TensorRT  
There are two main steps for the quantization:
1. Calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model
2. Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model

After Quantization is done, you can evaluate the QDQ model by running evaluation script we provide or your own script.

The **e2e_tensorrt_bert_example.py** is an end-to-end example for you to reference and run.

## Requirements
* Please build from latest ONNX Runtime source (see [here](https://onnxruntime.ai/docs/build/eps.html#tensorrt)) for now.
We plan to include TensorRT QDQ support later in ONNX Runtime 1.11 for [ORT Python GPU Package](https://pypi.org/project/onnxruntime-gpu/)
* TensorRT 8.2.2.1
* Python 3+
* numpy 
* The onnx model used in the script is converted from Hugging Face BERT model. https://huggingface.co/transformers/serialization.html#converting-an-onnx-model-using-the-transformers-onnx-package
* We use [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset as default dataset which is included in the repo. If you want to use other dataset to do calibration/evaluation, please either follow the format of squad/dev-1.1.json to add dataset or you have to write your own pre-processing method to parse your dataset.

Some utility functions for dataset processing, data reader and evaluation are from Nvidia TensorRT demo BERT repo,
https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT
## Model Calibration
Before running the calibration, please set the configuration properly in order to get better performance.

* **sequence_lengths** and **doc_stride** : Always consider them together. In order to get better accuracy result, choose doc stride 128 when using sequence length 384. Choose doc stride 32 when using sequence length 128. Generally speaking larger sequence_lengths and doc_stride can have better accuracy.
* **calib_num** : Default is 100. It's the number of examples in dataset used for calibration.

When calling `create_calibrator(...)`, following parameters are also configurable.
* **op_type_to_quantize** : Default is ['MatMul', 'Add']. One thing to remember is that even though quantizing more nodes improves inference latency, it can result in significant accuracy drop. So we don't suggest quantize all op type in the model.
* **calibrate_method** : Default is CalibrationMethod.MinMax. MinMax (CalibrationMethod.MinMax), Percentile (CalibrationMethod.Percentile) and Entropy (CalibrationMethod.Entropy) are supported. Please notice that generally use entropy algorithm for object detection models and percentile algorithm for NLP BERT models.
* **extra_options** : Default is {}. It can accept `num_bins`, `num_quantized_bins` and `percentile` as options. If no options are given, it will use internal default settings to run the calibration. When using entropy algorithm, `num_bins` (means number of bins of histogram for collecting floating tensor data) and `num_quantized_bins` (means number of bins of histogram after quantization) can be set with different combinations to test and fine-tune the calibration to get optimal result, for example, {'num_bins':8001, 'num_quantized_bins':255}. When using percentile algorithm, `num_bins` and `percentile` can be set with different values to fine-tune the calibration to get better result, for example, {'num_bins':2048, 'percentile':99.999}. 

## QDQ Model Generation
In order to get best performance from TensorRT, there are some optimizations being done when inserting Q/DQ nodes to the model.
* When inserting QDQ nodes to 'Add' node, only insert QDQ nodes to 'Add' node which is followed by ReduceMean node
* Enable per channel quantization on 'MatMul' node's weights. Please see `QDQQuantizer(...)` in the e2e example, the per_channel argument should be True as well as 'QDQOpTypePerChannelSupportToAxis': {'MatMul': 1} should be specified in extra_options argument. You can also modify 'QDQOpTypePerChannelSupportToAxis' to other op types and channel axis if they can increase performance.

Once QDQ model generation is done, the qdq_model.onnx will be saved.

## QDQ Model Evaluation
Remember to set env variables, ORT_TENSORRT_FP16_ENABLE=1 and ORT_TENSORRT_INT8_ENABLE=1, to run QDQ model.
We use evaluation tool from Nvidia TensorRT demo BERT repo to evaluate the result based on SQuAD v1.0 and SQuAD v2.0.

Note: The input names of model in the e2e example is based on Hugging Face Model's naming. If model input names are not correct in your model, please modify the code ort_session.run(["output_start_logits","output_end_logits"], inputs) in the example.

## Performance
The performance results were obtained by running [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest) with TensorRT 8.2.2.1 on
NVIDIA T4 with (1x T4 32G) GPUs. 
```shell
./onnxruntime_perf_test -e tensorrt -r 10000 qdq_model.onnx -o 0 -i 'trt_fp16_enable|true trt_int8_enable|true trt_engine_cache_enable|true'
```


The accuracy results were obtained by running e2e_tensorrt_bert_example.py as above mentioned. 
#### BERT Base
| Sequence Length | Batch Size | FP32 Latency (ms) | FP16 Latency (ms) |  INT8 Latency (ms) | FP32 Accuracy (F1) | FP16 Accuracy (F1) |    INT8 Accuracy (F1) |
|-----------------|------------|----|-------------|--------|------|-------|---------|
| 128 | 1 | 7.33134 ms| 2.14245 ms | 1.65864 ms |88.259| 88.111 | 82.988 |
| 384 | 1 | 21.6183 ms| 4.55848 ms | 3.58168 ms |88.662| 88.595 | 82.988 |
#### BERT Large
| Sequence Length | Batch Size | FP32 Latency (ms) | FP16 Latency (ms) |  INT8 Latency (ms) | FP32 Accuracy (F1) | FP16 Accuracy (F1) |    INT8 Accuracy (F1) |
|-----------------|------------|----|-------------|--------|------|-------|---------|
| 128 | 1 | 24.027  ms| 5.69197 ms | 4.46386 ms |90.109| 89.692 | 88.395 |
| 384 | 1 | 70.6816 ms| 14.5353 ms | 10.6119 ms | 90.539|89.747 | 89.541 |
