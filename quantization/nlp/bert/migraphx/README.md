# BERT QDQ Quantization in ONNX for MIGraphX
There are two main steps for the quantization:
1. Calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model
2. Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model

After Quantization is done, you can evaluate the QDQ model by running evaluation script we provide or your own script.

The **e2e_migraphx_bert_example.py** is an end-to-end example for you to reference and run.

## Requirements
* Please build from latest ONNX Runtime source (see [here](https://onnxruntime.ai/docs/build/eps.html#migraphx)) for now.
* MIGraphX 2.8 and above
* ROCm 5.7 and above (For calibration data)
* Python 3+
* numpy 
* The onnx model used in the script is converted from Hugging Face BERT model. https://huggingface.co/transformers/serialization.html#converting-an-onnx-model-using-the-transformers-onnx-package
* We use [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset as default dataset which is included in the repo. If you want to use other dataset to do calibration/evaluation, please either follow the format of squad/dev-1.1.json to add dataset or you have to write your own pre-processing method to parse your dataset.

Some utility functions for dataset processing, data reader and evaluation are from Nvidia TensorRT demo BERT repo,
https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT

Code from the TensorRT example has been reused for the MIGraphX Execution Provider to showcase how simple it is to convert over CUDA and TensorRT code into MIGraphX and ROCm within Onnxruntime. Just change the desired Execution Provider and install the proper requirements (ROCm and MIGraphX) and run your script as you did with CUDA.

We've also added a few more input args to the script to help finetune the inference you'd like to run. Feel free to use the --help when running

usage: e2e_migraphx_bert_example.py [-h] [--fp16] [--int8] [--ep EP] [--cal_ep CAL_EP] [--model MODEL]
                                    [--vocab VOCAB] [--token TOKEN] [--version VERSION] [--no_eval]
                                    [--ort_verbose] [--ort_quant] [--save_load] [--batch BATCH]
                                    [--seq_len SEQ_LEN] [--query_len QUERY_LEN] [--doc_stride DOC_STRIDE]
                                    [--cal_num CAL_NUM] [--samples SAMPLES] [--verbose]

options:
  -h, --help            show this help message and exit
  --fp16                Perform fp16 quantization on the model before running inference
  --int8                Perform int8 quantization on the model before running inference
  --ep EP               The desired execution provider [MIGraphX, ROCm] are the options; Default is MIGraphX
  --cal_ep CAL_EP       The desired execution provider [MIGraphX, ROCm, CPU] for int8 quantization; Default is
                        MIGraphX
  --model MODEL         Path to the desired model to be run. Default ins ./model.onnx
  --vocab VOCAB         Path to the vocab of the model. Default is ./squad/vocab.txt
  --token TOKEN         Path to the tokenized inputs. Default is None and will be taken from vocab file
  --version VERSION     Squad dataset version. Default is 1.1. Choices are 1.1 and 2.0
  --no_eval             Turn off evaluate output result for f1 and exact match score. Default False
  --ort_verbose         Turn on onnxruntime verbose flags
  --ort_quant           Turn on Onnxruntime Quantizer instead of MIGraphX Quantizer
  --save_load           Turn on Onnxruntime Model save loading to speed up inference
  --batch BATCH         Batch size per inference
  --seq_len SEQ_LEN     sequence length of the model. Default is 384
  --query_len QUERY_LEN
                        max querry length of the model. Default is 64
  --doc_stride DOC_STRIDE
                        document stride of the model. Default is 128
  --cal_num CAL_NUM     Number of calibration for QDQ Quantiation in int8. Default is 100
  --samples SAMPLES     Number of samples to test with. Default is 0 (All the samples in the dataset)
  --verbose             Show verbose output


## Model Calibration
Before running the calibration, please set the configuration properly in order to get better performance.

* **sequence_lengths** and **doc_stride** : Always consider them together. In order to get better accuracy result, choose doc stride 128 when using sequence length 384. Choose doc stride 32 when using sequence length 128. Generally speaking larger sequence_lengths and doc_stride can have better accuracy.
* **calib_num** : Default is 100. It's the number of examples in dataset used for calibration.

When calling `create_calibrator(...)`, following parameters are also configurable.
* **op_type_to_quantize** : Default is ['MatMul', 'Add']. One thing to remember is that even though quantizing more nodes improves inference latency, it can result in significant accuracy drop. So we don't suggest quantize all op type in the model.
* **calibrate_method** : Default is CalibrationMethod.MinMax. MinMax (CalibrationMethod.MinMax), Percentile (CalibrationMethod.Percentile) and Entropy (CalibrationMethod.Entropy) are supported. Please notice that generally use entropy algorithm for object detection models and percentile algorithm for NLP BERT models.
* **extra_options** : Default is {}. It can accept `num_bins`, `num_quantized_bins` and `percentile` as options. If no options are given, it will use internal default settings to run the calibration. When using entropy algorithm, `num_bins` (means number of bins of histogram for collecting floating tensor data) and `num_quantized_bins` (means number of bins of histogram after quantization) can be set with different combinations to test and fine-tune the calibration to get optimal result, for example, {'num_bins':8001, 'num_quantized_bins':255}. When using percentile algorithm, `num_bins` and `percentile` can be set with different values to fine-tune the calibration to get better result, for example, {'num_bins':2048, 'percentile':99.999}. 

## QDQ Model Generation
In order to get best performance from MIGraphX, there are some optimizations being done when inserting Q/DQ nodes to the model.
* When inserting QDQ nodes to 'Add' node, only insert QDQ nodes to 'Add' node which is followed by ReduceMean node
* Enable per channel quantization on 'MatMul' node's weights. Please see `QDQQuantizer(...)` in the e2e example, the per_channel argument should be True as well as 'QDQOpTypePerChannelSupportToAxis': {'MatMul': 1} should be specified in extra_options argument. You can also modify 'QDQOpTypePerChannelSupportToAxis' to other op types and channel axis if they can increase performance.

Once QDQ model generation is done, the qdq_model.onnx will be saved.

## QDQ Model Evaluation
Remember to set env variables, ORT_MIGRAPHX_FP16_ENABLE=1 and ORT_MIGRAPHX_INT8_ENABLE=1, to run QDQ model.
We use evaluation tool from Nvidia TensorRT demo BERT repo to evaluate the result based on SQuAD v1.0 and SQuAD v2.0.

Note: The input names of model in the e2e example is based on Hugging Face Model's naming. If model input names are not correct in your model, please modify the code ort_session.run(["output_start_logits","output_end_logits"], inputs) in the example.

