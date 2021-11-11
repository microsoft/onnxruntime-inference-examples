import os
import onnx
import onnxruntime
import numpy as np
import json
import sys
import logging
from pathlib import Path
# from onnxruntime.quantization import create_calibrator, CalibrationMethod, write_calibration_table, QuantType, QuantizationMode, QLinearOpsRegistry, QDQQuantizer
from onnxruntime.quantization import create_calibrator, CalibrationMethod, write_calibration_table, QuantType, QuantizationMode, QDQQuantizer
from hf_helper import HFBertDataReader 
from helper import BertDataReader, BertEvaluater

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    '''
    BERT QDQ Quantization for TensorRT.

    There are three steps for the quantization,
    first, calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model,
    second, Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model,
    finally, evaluate the qdq model.

    The onnx model used in the script is converted from Hugging Face BERT model,
    https://huggingface.co/transformers/serialization.html#converting-an-onnx-model-using-the-transformers-onnx-package

    Some utility functions for dataset processing, data reader and evaluation are from Nvidia TensorRT demo BERT repo,
    https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT
    '''

    # Model, dataset and quantization settings
    model_path = "./model.onnx"
    model_path = "./finetuned_fp32_mode_base.onnx"
    squad_json = "./bert-squad/dev-v1.1.json"
    vocab_file = "./bert-squad/vocab.txt"
    augmented_model_path = "./augmented_model.onnx"
    qdq_model_path = "./qdq_model.onnx"
    sequence_lengths = 128
    doc_stride = 32
    calib_num = 100
    batch_size = 8 
    op_types_to_quantize = ['MatMul', 'Add']
    op_types_to_exclude_output_quantization = op_types_to_quantize # don't add qdq to node's output to avoid accuracy drop
    is_using_hf = True 
    # is_using_hf = False 

    # Create data reader
    if is_using_hf:
        # convert to Huggingface transformers parameters
        trainer_args_dict = {
            'model_name_or_path': 'bert-base-uncased',
            'dataset_name': 'squad',
            'per_device_eval_batch_size': batch_size, # default is 8 
            'max_seq_length': sequence_lengths, 
            'doc_stride': doc_stride,
            'output_dir': 'output',
        }

        calib_args_dict = {
            # we currently don't use HF's pytorch quantization for calibration, so leave it blank.  
        }
    
        # data_reader = HFBertDataReader(trainer_args_dict, calib_args_dict, start_index=0, end_index=calib_num) 
    else:
        batch_size = 1
        data_reader = BertDataReader(model_path, squad_json, vocab_file, batch_size, sequence_lengths)


    # Generate INT8 calibration cache
    logger.info("*** Calibration starts ***")
    # if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        # op_types_to_quantize = list(QLinearOpsRegistry.keys())

    calibrator = create_calibrator(model_path, op_types_to_quantize, augmented_model_path=augmented_model_path, calibrate_method=CalibrationMethod.Percentile)
    calibrator.set_execution_providers(["CUDAExecutionProvider"]) 

    '''
    We can use one data reader to do data pre-processing, however,
    some machines don't have sufficient memory to hold all dataset and all intermediate output,
    especially using 'Entropy' or 'Percentile' calibrator which collects histogram for tensors.
    So let multiple data readers to handle different stride of dataset to avoid OOM.
    '''
    stride = 10 
    for i in range(0, calib_num, stride):
        if is_using_hf:
            data_reader = HFBertDataReader(trainer_args_dict, calib_args_dict, start_index=i, end_index=i+stride) 
        else:
            data_reader.update_load_range(i, i+stride) 
        calibrator.collect_data(data_reader)

    compute_range = calibrator.compute_range()
    write_calibration_table(compute_range)
    logger.info("Calibration is done. Calibration cache is saved to calibration.json")

    # with open('calibration.json', 'r') as f:
        # compute_range=json.load(f)

    # Generate QDQ model
    logger.info("*** Generate QDQ model ***")
    mode = QuantizationMode.QLinearOps
    model = onnx.load_model(Path(model_path), False)
    quantizer = QDQQuantizer(
        model,
        True, #per_channel
        False, #reduce_range
        mode,
        True,  #static
        QuantType.QInt8, #weight_type
        QuantType.QInt8, #activation_type
        compute_range,
        [], #nodes_to_quantize
        [], #nodes_to_exclude
        op_types_to_quantize,
        {'ActivationSymmetric' : True, 'AddQDQPairToWeight' : True, 'AddQDQToAddNodeFollowedByReduceMeanNode': True, 'OpTypesToExcludeOutputQuantizatioin': op_types_to_quantize, 'DedicatedQDQPair': True, 'QDQChannelAxis': 1}) #extra_options
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(qdq_model_path, False)
    logger.info("QDQ model is saved to " + qdq_model_path)

    # QDQ model inference and get SQUAD prediction 
    logger.info("*** QDQ model Evaluate ***")
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable TRT FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable TRT INT8 precision
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(qdq_model_path, sess_options=sess_options, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
    if is_using_hf:
        data_reader = HFBertDataReader(trainer_args_dict, calib_args_dict, session)
        trainer = data_reader.trainer
        metrics = trainer.evaluate()

        max_eval_samples = data_reader.data_args.max_eval_samples if data_reader.data_args.max_eval_samples is not None else len(trainer.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(trainer.eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    else:
        batch_size = 1 
        data_reader = BertDataReader(qdq_model_path, squad_json, vocab_file, batch_size, sequence_lengths)
        BertEvaluater(session, data_reader).evaluate()
