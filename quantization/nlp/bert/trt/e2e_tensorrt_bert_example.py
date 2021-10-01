import os
import onnx
import glob
import scipy.io
import numpy as np
import logging
from PIL import Image
import json
import collections
import six
import unicodedata
import onnx
import onnxruntime
import sys
import data_processing as dp
import tokenization
from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table, QuantType, QuantizationMode, QLinearOpsRegistry, QDQQuantizer
from squad.evaluate_v1_1 import evaluate 

class BertDataReader(CalibrationDataReader):
    def __init__(self, model_path, squad_json, vocab_file, batch_size, max_seq_length, num_inputs):
        self.model_path = model_path
        self.data = dp.read_squad_json(squad_json)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.current_index = 0
        self.num_inputs = min(num_inputs, len(self.data))
        self.tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.doc_stride = 128
        self.max_query_length = 64
        self.enum_data_dicts = iter([])
        self.feature_list = []
        self.token_list = []
        self.example_id_list = []

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index, self.batch_size, self.num_inputs))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} sentences".format(current_batch, self.batch_size))

        input_ids = []
        input_mask = []
        segment_ids = []
        for i in range(self.batch_size):
            example = self.data[self.current_index + i]
            features = dp.convert_example_to_features(example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
            self.example_id_list.append(example.id)
            self.feature_list.append(features)
            self.token_list.append(example.doc_tokens)
            if len(input_ids) and len(segment_ids) and len(input_mask):
                input_ids = np.vstack([input_ids, features[0].input_ids])
                input_mask = np.vstack([input_mask, features[0].input_mask])
                segment_ids = np.vstack([segment_ids, features[0].segment_ids])
            else:
                input_ids = np.expand_dims(features[0].input_ids, axis=0)
                input_mask = np.expand_dims(features[0].input_mask, axis=0)
                segment_ids = np.expand_dims(features[0].segment_ids, axis=0)
        data = [{"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids":segment_ids}]
        self.current_index += self.batch_size
        self.enum_data_dicts = iter(data)
        return next(self.enum_data_dicts, None)

def inference(ort_session, features, example):

    _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
            "NetworkOutput",
            ["start_logits", "end_logits", "feature_index"])
    networkOutputs = []

    for i, feature in enumerate(features):
        input_ids = np.expand_dims(features[i].input_ids, axis=0)
        input_mask = np.expand_dims(features[i].input_mask, axis=0)
        segment_ids = np.expand_dims(features[i].segment_ids, axis=0)

        data = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids":segment_ids}
        output = ort_session.run(["start_logits","end_logits"], data)

        start_logits = output[0][0]
        end_logits = output[1][0]

        networkOutputs.append(_NetworkOutput(
            start_logits = start_logits,
            end_logits = end_logits,
            feature_index = i 
            ))

    # Total number of n-best predictions to generate in the nbest_predictions.json output file
    n_best_size = 20

    # The maximum length of an answer that can be generated. This is needed
    # because the start and end predictions are not conditioned on one another
    max_answer_length = 30

    prediction, nbest_json, scores_diff_json = dp.get_predictions(example.doc_tokens, features,
            networkOutputs, n_best_size, max_answer_length)
    
    return prediction


if __name__ == '__main__':
    '''
    BERT QDQ Quantization for TensorRT.

    There are two steps for the quantization,
    first, calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model
    second, Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model

    The onnx model used in the script is converted from Hugging Face BERT model,
    https://huggingface.co/transformers/serialization.html#converting-an-onnx-model-using-the-transformers-onnx-package

    Some utility functions for dataset processing, data reader and evaluation are from Nvidia TensorRT demo BERT repo,
    https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT
    '''

    # Model, dataset and quantization settings
    model_path = "./model.onnx"
    squad_json = "./squad/dev-v1.1.json"
    vocab_file = "./squad/vocab.txt"
    augmented_model_path = "./augmented_model.onnx"
    qdq_model_path = "./qdq_model.onnx"
    sequence_lengths = [384]
    calib_num = 100
    op_types_to_quantize = ['MatMul', 'Transpose', 'Add']

    # Generate INT8 calibration cache
    print("Calibration starts ...")
    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())
    calibrator = create_calibrator(model_path, op_types_to_quantize, augmented_model_path=augmented_model_path)
    calibrator.set_execution_providers(["CUDAExecutionProvider"]) 
    data_reader = BertDataReader(model_path, squad_json, vocab_file, 2, sequence_lengths[-1], calib_num)
    calibrator.collect_data(data_reader)
    compute_range = calibrator.compute_range()
    write_calibration_table(compute_range)
    print("Calibration is done. Calibration cache is saved to calibration.json")

    # Generate QDQ model
    mode = QuantizationMode.QLinearOps

    model = onnx.load_model(Path(model_path), False)
    quantizer = QDQQuantizer(
        model,
        False, #per_channel
        False, #reduce_range
        mode,
        True,  #static
        QuantType.QInt8, #weight_type
        QuantType.QInt8, #activation_type
        compute_range,
        [], #nodes_to_quantize
        [], #nodes_to_exclude
        op_types_to_quantize,
        {'ActivationSymmetric' : True}) #extra_options
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(qdq_model_path, False)
    print("QDQ model is saved to ", qdq_model_path)

    # QDQ model inference and get SQUAD prediction 
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable TRT FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable TRT INT8 precision
    session = onnxruntime.InferenceSession(qdq_model_path, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])

    examples = dp.read_squad_json(squad_json)
    tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
    all_predictions = collections.OrderedDict()
    max_seq_length = sequence_lengths[-1] 
    doc_stride = 128
    max_query_length = 64
    for i, example in enumerate(examples):
        features = dp.convert_example_to_features(example.doc_tokens, example.question_text, tokenizer, max_seq_length, doc_stride, max_query_length)
        prediction = inference(session, features, example)
        all_predictions[example.id] = prediction

    prediction_file = "./prediction.json"
    with open(prediction_file, "w") as f:
        f.write(json.dumps(all_predictions, indent=4))
        print("\nOutput dump to {}".format(prediction_file))


    # Evaluate QDQ model for SQUAD
    # exact match and F1 metrics will be calculated 
    expected_version = "1.1"
    dataset_file = squad_json 
    f1_acc = 90 # Reference Accuracy 

    with open(dataset_file) as f:
        dataset_json = json.load(f)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(prediction_file) as f:
        predictions = json.load(f)
        f1_acc = float(f1_acc)
    print(json.dumps(evaluate(dataset, predictions, f1_acc)))
