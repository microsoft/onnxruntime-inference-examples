import os
import onnx
import onnxruntime
import numpy as np
import json
import collections
import data_processing as dp
import tokenization
import subprocess
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, CalibrationMethod, write_calibration_table, QuantType, QuantizationMode, QDQQuantizer
import argparse
import time

class BertDataReader(CalibrationDataReader):
    def __init__(self,
                 model_path,
                 squad_json,
                 tokens,
                 batch_size,
                 max_seq_length,
                 max_query_length,
                 doc_stride,
                 start_index=0,
                 end_index=0):
        self.model_path = model_path
        self.data = squad_json # Input list to read from
        self.tokenizer = tokens
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.example_stride = batch_size # number of examples as one example stride. (set to equal to batch size) 
        self.start_index = start_index # squad example index to start with
        self.end_index = len(self.data) if end_index == 0 else end_index 
        self.current_example_index = start_index
        self.current_feature_index = 0 # global feature index (one example can have more than one feature) 
        self.doc_stride = doc_stride 
        self.max_query_length = max_query_length
        self.enum_data_dicts = iter([])
        self.features_list = []
        self.token_list = []
        self.example_id_list = []
        self.start_of_new_stride = False # flag to inform that it's a start of new example stride

    def get_next(self, verbose=False):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            self.start_of_new_stride= False
            return iter_data

        self.enum_data_dicts = None
        if self.current_example_index >= self.end_index:
            print("Reading dataset is done. Total examples is {:}".format(self.end_index-self.start_index))
            return None
        elif self.current_example_index + self.example_stride > self.end_index:
            self.example_stride = self.end_index - self.current_example_index
            
        if verbose:
            current_batch = int(self.current_example_index / self.batch_size) 
            print("Example stride:" + str(self.example_stride))
            print("Reading example index {:}, batch {:}, containing {:} sentences".format(self.current_example_index, current_batch, self.batch_size))

        # example could have more than one feature
        # we collect all the features of examples and process them in one example stride
        features_in_current_stride = []
        if verbose:
            print("prev_example_index: " + str(self.current_example_index))
            print("prev_feature_index: " + str(self.current_feature_index))
            print("features in prev stride: " + str(len(features_in_current_stride)))

        for i in range(self.example_stride):
            example = self.data[self.current_example_index+ i]
            features = dp.convert_example_to_features(example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
            self.example_id_list.append(example.id)
            self.features_list.append(features)
            self.token_list.append(example.doc_tokens)
            features_in_current_stride += features
        self.current_example_index += self.example_stride
        self.current_feature_index+= len(features_in_current_stride)

        if verbose:
            print("current_example_index: " + str(self.current_example_index))
            print("current_feature_index: " + str(self.current_feature_index))
            print("features in current stride: " + str(len(features_in_current_stride)))

        # following layout shows three examples as example stride with batch size 2:
        # 
        # start of new example stride 
        # |
        # |
        # v
        # <--------------------- batch size 2 ---------------------->
        # |...example n, feature 1....||...example n, feature 2.....| 
        # |...example n, feature 3....||...example n+1, feature 1...| 
        # |...example n+1, feature 2..||...example n+1, feature 3...|
        # |...example n+1, feature 4..||...example n+2, feature 1...|

        data = []
        for feature_idx in range(0, len(features_in_current_stride), self.batch_size):
            input_ids = []
            input_mask = []
            segment_ids = []

            for i in range(self.batch_size):
                if feature_idx + i >= len(features_in_current_stride):
                    # Pad with same/last feature so we don't have to recompile the model due to varying inputs
                    # It appears this means we're running with a dynamic batch
                    if i < self.batch_size:
                        for j in range(self.batch_size - i):
                            input_ids = np.vstack([input_ids, feature.input_ids])
                            input_mask = np.vstack([input_mask, feature.input_mask])
                            segment_ids = np.vstack([segment_ids, feature.segment_ids])
                    if verbose:
                        print("End of features, repeating last feature as padding")
                        print("Break")
                    break
                feature = features_in_current_stride[feature_idx + i]
                if verbose:
                    print("Feature Index:" + str(feature_idx))
                if len(input_ids) and len(segment_ids) and len(input_mask):
                    if verbose:
                        print("vstack")
                    input_ids = np.vstack([input_ids, feature.input_ids])
                    input_mask = np.vstack([input_mask, feature.input_mask])
                    segment_ids = np.vstack([segment_ids, feature.segment_ids])
                else:
                    if verbose:
                        print("expand_dims")
                        print(feature.input_ids.shape)
                    input_ids = np.expand_dims(feature.input_ids, axis=0)
                    input_mask = np.expand_dims(feature.input_mask, axis=0)
                    segment_ids = np.expand_dims(feature.segment_ids, axis=0)

            if verbose:
                print("Input ID shape " + str(input_ids.shape))
                print("input mask shape " + str(input_mask.shape))
                print("Segment ID shape" + str(segment_ids.shape))
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            if flags.version == 1.1:
                data.append({"input_ids": input_ids, "input_mask": input_mask, "segment_ids":segment_ids})
            elif flags.version >= 2.0:
                data.append({"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids":segment_ids})


        self.enum_data_dicts = iter(data)
        self.start_of_new_stride = True
        return next(self.enum_data_dicts, None)

def get_predictions(example_id_in_current_stride,
                    features_in_current_stride,
                    token_list_in_current_stride,
                    batch_size,
                    outputs,
                    _NetworkOutput,
                    all_predictions):
                    
    if example_id_in_current_stride == []:
        return 

    base_feature_idx = 0
    for idx, id in enumerate(example_id_in_current_stride):
        features = features_in_current_stride[idx]
        doc_tokens = token_list_in_current_stride[idx]
        networkOutputs = []
        for i in range(len(features)):
            x = (base_feature_idx + i) // batch_size
            y = (base_feature_idx + i) % batch_size

            output = outputs[x]
            start_logits = output[0][y]
            end_logits = output[1][y]

            networkOutputs.append(_NetworkOutput(
                start_logits = start_logits,
                end_logits = end_logits,
                feature_index = i 
                ))

        base_feature_idx += len(features) 

        # Total number of n-best predictions to generate in the nbest_predictions.json output file
        n_best_size = 20

        # The maximum length of an answer that can be generated. This is needed
        # because the start and end predictions are not conditioned on one another
        max_answer_length = 30

        prediction, nbest_json, scores_diff_json = dp.get_predictions(doc_tokens, features,
                networkOutputs, n_best_size, max_answer_length)

        all_predictions[id] = prediction

def inference(data_reader, ort_session, latency, verbose=False):

    _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
            "NetworkOutput",
            ["start_logits", "end_logits", "feature_index"])
    all_predictions = collections.OrderedDict()
    
    example_id_in_current_stride = [] 
    features_in_current_stride = []  
    token_list_in_current_stride = []
    outputs = []
    while True:
        inputs = data_reader.get_next(verbose)
        if not inputs:
            break

        if data_reader.start_of_new_stride:
            if verbose:
                print("Start of new stride")
            get_predictions(example_id_in_current_stride, features_in_current_stride, token_list_in_current_stride, data_reader.batch_size, outputs, _NetworkOutput, all_predictions)

            # reset current example stride
            example_id_in_current_stride = data_reader.example_id_list[-data_reader.example_stride:]
            features_in_current_stride = data_reader.features_list[-data_reader.example_stride:] 
            token_list_in_current_stride = data_reader.token_list[-data_reader.example_stride:]
            outputs = []

        io_binding = ort_session.io_binding()

        if flags.version == 1.1:
            io_binding.bind_cpu_input('input_ids', inputs['input_ids'])
            io_binding.bind_cpu_input('input_mask', inputs['input_mask'])
            io_binding.bind_cpu_input('segment_ids', inputs['segment_ids'])
        elif flags.version == 2.0:
            io_binding.bind_cpu_input('segment_ids', inputs['token_type_ids'])
            io_binding.bind_cpu_input('input_mask', inputs['attention_mask'])
            io_binding.bind_cpu_input('input_ids', inputs['input_ids'])
        elif flags.version == 2.1:
            io_binding.bind_cpu_input('token_type_ids', inputs['token_type_ids'])
            io_binding.bind_cpu_input('attention_mask', inputs['attention_mask'])
            io_binding.bind_cpu_input('input_ids', inputs['input_ids'])


        if flags.version <= 2.0:
            io_binding.bind_output('output_start_logits')
            io_binding.bind_output('output_end_logits')
        else:
            io_binding.bind_output('start_logits')
            io_binding.bind_output('end_logits')

        if verbose:
            print("Before Inference")

        start = time.time()
        #output = ort_session.run(["output_start_logits","output_end_logits"], inputs)
        ort_session.run_with_iobinding(io_binding)
        latency.append(time.time() - start)
        output = io_binding.copy_outputs_to_cpu()
        outputs.append(output)

        if verbose:
            print("After Inference")

    # handle the last example stride
    get_predictions(example_id_in_current_stride, features_in_current_stride, token_list_in_current_stride, data_reader.batch_size, outputs, _NetworkOutput, all_predictions)

    return all_predictions

def get_op_nodes_not_followed_by_specific_op(model, op1, op2):
    op1_nodes = []
    op2_nodes = []
    selected_op1_nodes = []
    not_selected_op1_nodes = []

    for node in model.graph.node:
        if node.op_type == op1:
            op1_nodes.append(node)
        if node.op_type == op2:
            op2_nodes.append(node)

    for op1_node in op1_nodes:
        for op2_node in op2_nodes:
            if op1_node.output == op2_node.input:
                selected_op1_nodes.append(op1_node.name)
        if op1_node.name not in selected_op1_nodes:
            not_selected_op1_nodes.append(op1_node.name)

    return not_selected_op1_nodes

def custom_write_calibration_table(calibration_cache, dir="."):
    """
    Helper function to write calibration table to files.
    """

    import json
    import logging
    import flatbuffers
    import numpy as np

    import onnxruntime.quantization.CalTableFlatBuffers.KeyValue as KeyValue
    import onnxruntime.quantization.CalTableFlatBuffers.TrtTable as TrtTable
    from onnxruntime.quantization.calibrate import CalibrationMethod, TensorData, TensorsData

    logging.info(f"calibration cache: {calibration_cache}")

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (TensorData, TensorsData)):
                return obj.to_dict()
            if isinstance(obj, TensorDataWrapper):
                return obj.data_dict
            if isinstance(obj, np.ndarray):
                return {"data": obj.tolist(), "dtype": str(obj.dtype), "CLS": "numpy.array"}
            if isinstance(obj, CalibrationMethod):
                return {"CLS": obj.__class__.__name__, "value": str(obj)}
            return json.JSONEncoder.default(self, obj)

    json_data = json.dumps(calibration_cache, cls=MyEncoder)

    with open(os.path.join(dir, "calibration.json"), "w") as file:
        file.write(json_data)  # use `json.loads` to do the reverse

    # Serialize data using FlatBuffers
    zero = np.array(0)
    builder = flatbuffers.Builder(1024)
    key_value_list = []

    for key in sorted(calibration_cache.keys()):
        values = calibration_cache[key]
        d_values = values.to_dict()

        highest = d_values.get("highest", zero)
        lowest = d_values.get("lowest", zero)

        highest_val = highest.item() if hasattr(highest, "item") else float(highest)
        lowest_val = lowest.item() if hasattr(lowest, "item") else float(lowest)

        floats = [float(highest_val), float(lowest_val)]

        value = str(max(floats))

        flat_key = builder.CreateString(key)
        flat_value = builder.CreateString(value)

        KeyValue.KeyValueStart(builder)
        KeyValue.KeyValueAddKey(builder, flat_key)
        KeyValue.KeyValueAddValue(builder, flat_value)
        key_value = KeyValue.KeyValueEnd(builder)

        key_value_list.append(key_value)


    TrtTable.TrtTableStartDictVector(builder, len(key_value_list))
    for key_value in key_value_list:
        builder.PrependUOffsetTRelative(key_value)
    main_dict = builder.EndVector()

    TrtTable.TrtTableStart(builder)
    TrtTable.TrtTableAddDict(builder, main_dict)
    cal_table = TrtTable.TrtTableEnd(builder)

    builder.Finish(cal_table)
    buf = builder.Output()

    with open(os.path.join(dir, "calibration.flatbuffers"), "wb") as file:
        file.write(buf)

    # Deserialize data (for validation)
    if os.environ.get("QUANTIZATION_DEBUG", 0) in (1, "1"):
        cal_table = TrtTable.TrtTable.GetRootAsTrtTable(buf, 0)
        dict_len = cal_table.DictLength()
        for i in range(dict_len):
            key_value = cal_table.Dict(i)
            logging.info(key_value.Key())
            logging.info(key_value.Value())

    # write plain text
    with open(os.path.join(dir, "calibration.cache"), "w") as file:
        for key in sorted(calibration_cache.keys()):
            values = calibration_cache[key]
            d_values = values.to_dict()
            highest = d_values.get("highest", zero)
            lowest = d_values.get("lowest", zero)

            highest_val = highest.item() if hasattr(highest, "item") else float(highest)
            lowest_val = lowest.item() if hasattr(lowest, "item") else float(lowest)

            floats = [float(highest_val), float(lowest_val)]

            value = key + " " + str(max(floats))
            file.write(value)
            file.write("\n")

def parse_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp16",
        action="store_true",
        required=False,
        default=False,
        help='Perform fp16 quantization on the model before running inference',
    )

    parser.add_argument(
        "--int8",
        action="store_true",
        required=False,
        default=False,
        help='Perform int8 quantization on the model before running inference',
    )

    parser.add_argument(
        "--fp8",
        action="store_true",
        required=False,
        default=False,
        help='Perform fp8 quantization on the model before running inference',
    )

    parser.add_argument(
        "--ep",
        action="store",
        required=False,
        default="MIGraphX",
        type=str,
        help='The desired execution provider [MIGraphX, ROCm] are the options; Default is MIGraphX',
    )

    parser.add_argument(
        "--cal_ep",
        action="store",
        required=False,
        default="MIGraphX",
        type=str,
        help='The desired execution provider [MIGraphX, ROCm, CPU] for int8 quantization; Default is MIGraphX',
    )

    parser.add_argument(
        "--calibration_table",
        action="store",
        required=False,
        default="./bert_calibration_table_100_int8.flatbuffers",
        type=str,
        help='use a previously created calibration table" default is ./bert_calibration_table_100_int8.flatbuffers',
    )

    parser.add_argument(
        "--model",
        action="store",
        required=False,
        default="./model.onnx",
        type=str,
        help='Path to the desired model to be run. Default ins ./model.onnx',
    )

    parser.add_argument(
        "--vocab",
        action="store",
        required=False,
        default="./squad/vocab.txt",
        type=str,
        help='Path to the vocab of the model. Default is ./squad/vocab.txt',
    )

    parser.add_argument(
        "--token",
        action="store",
        required=False,
        default=None,
        type=str,
        help='Path to the tokenized inputs. Default is None and will be taken from vocab file',
    )

    parser.add_argument(
        "--version",
        required=False,
        default=1.1,
        help='Squad dataset version. Default is 1.1. Choices are 1.1 and 2.0',
        type=float
    )

    parser.add_argument(
        "--no_eval",
        action="store_true",
        required=False,
        default=False,
        help='Turn off evaluate output result for f1 and exact match score. Default False',
    )

    parser.add_argument(
        "--ort_verbose",
        action="store_true",
        required=False,
        default=False,
        help='Turn on onnxruntime verbose flags',
    )

    parser.add_argument(
        "--ort_quant",
        action="store_true",
        required=False,
        default=False,
        help='Turn on Onnxruntime Quantizer instead of MIGraphX Quantizer',
    )
    
    parser.add_argument(
        "--save_load",
        action="store_true",
        required=False,
        default=False,
        help='Turn on Onnxruntime Model save loading to speed up inference',
    )


    parser.add_argument("--batch",
                        required=False,
                        default=1,
                        help='Batch size per inference',
                        type=int)

    parser.add_argument("--seq_len",
                        required=False,
                        default=384,
                        help='sequence length of the model. Default is 384',
                        type=int)

    parser.add_argument("--query_len",
                        required=False,
                        default=64,
                        help='max querry length of the model. Default is 64',
                        type=int)

    parser.add_argument("--doc_stride",
                        required=False,
                        default=128,
                        help='document stride of the model. Default is 128',
                        type=int)

    parser.add_argument("--cal_num",
                        required=False,
                        default=100,
                        help='Number of calibration for QDQ Quantiation in int8. Default is 100',
                        type=int)

    parser.add_argument("--samples",
                        required=False,
                        default=1000,
                        help='Number of samples to test with. Default is 0 (All the samples in the dataset)',
                        type=int)


    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help='Show verbose output',
    )
    
    return parser.parse_args()

def output_run_config(flags, samples):
    print ("filename:" + flags.model)
    print ("Samples: " + str(samples) + " Batch size: " + str(flags.batch))
    print ("Sequence length: " + str(flags.seq_len))
    print ("Model Quantization: fp16:" + str(flags.fp16) + " int8:" + str(flags.int8) + " fp8:" + str(flags.fp8))
    if flags.int8:
        if flags.ort_quant:
            print ("Quantizer: Onnxruntime")
        else:
            print("Quantizer: MIGraphX")


if __name__ == '__main__':
    '''
    BERT QDQ Quantization for MIGraphX.

    There are two steps for the quantization,
    first, calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model
    second, Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model

    The onnx model used in the script is converted from MLPerf,
    https://zenodo.org/records/3733910
    
    Some utility functions for dataset processing, data reader and evaluation are from Nvidia TensorRT demo BERT repo,
    https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT
    
    '''

    flags = parse_input_args()

    # Model, dataset and quantization settings
    model_path = flags.model

    if flags.ep == "MIGraphX":
        ep = "MIGraphXExecutionProvider"
    elif flags.ep == "ROCm":
        ep = "ROCMExecutionProvider"
    else:
        print("Error: EP:" + str(flags.ep) + " Invalid")
        exit

    cal_ep = "MIGraphXExecutionProvider"
    if flags.int8:
        if flags.cal_ep == "MIGraphX":
            cal_ep = "MIGraphXExecutionProvider"
        elif flags.cal_ep == "ROCm":
            cal_ep = "ROCMExecutionProvider"
        elif flags.cal_ep == "CPU":
            cal_ep = "CPUExecutionProvider"
        else:
            print("Error: cal_ep:" + str(flags.cal_ep) + " Invalid")
            exit


    # Set squad version   
    if flags.version == 1.1:
        squad_json = "./squad/dev-v1.1.json"
    elif flags.version >= 2.0:
        squad_json = "./squad/dev-v2.0.json" # uncomment it if you want to use squad v2.0 as dataset
    else:
        print("Error: version " + str(flags.version) + " of squad dataset not a valid choice")
        exit()
    
    vocab_file = flags.vocab
    if vocab_file is None:
        vocab_file = "./squad/vocab.txt"
    augmented_model_path = "./augmented_model.onnx"
    qdq_model_path = "./qdq_model.onnx"

    sequence_lengths = [flags.seq_len] # if use sequence length 384 then choose doc stride 128. if use sequence length 128 then choose doc stride 32. 
    

    doc_stride = [flags.doc_stride]
    calib_num = flags.cal_num
    op_types_to_quantize = ['MatMul', 'Conv']
    batch_size = flags.batch

    # Preprocess input data once.
    input_dataset   = dp.read_squad_json(squad_json)

    input_tokens = flags.token
    if input_tokens is None:
        input_tokens = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

    samples = flags.samples
    if flags.samples > len(input_dataset):
        print("Selecting more samples than available defaulting to " + str(len(input_dataset)) + " samples")
        samples = len(input_dataset)

    if samples < flags.batch:
        print("Selecting less samples than target batch size. Setting to " + str(flags.batch) + " samples")
        samples = flags.batch

    model_quants = "" 
    provider_args = {}

    if flags.int8 and flags.fp8:
        print("INT8 and FP8 quantization is mutually exclusive for calibration")
        exit()

    precision=""
    if flags.int8:
        precision = precision + "_int8"
        provider_args["migraphx_int8_enable"] = str(True)

    if flags.fp8 :
        precision = precision + "_fp8"
        provider_args["migraphx_fp8_enable"] = str(True)

    if flags.int8 or flags.fp8:
        model = onnx.load_model(model_path)

        if os.path.isfile("./" + flags.calibration_table):
            print("Found previous calibration: " + flags.calibration_table + "Skipping generating table")
            provider_args["migraphx_int8_calibration_table_name"] = str(flags.calibration_table)
        else:
            calibration_table_name = "bert_calibration_table_"+ str(flags.cal_num) + precision + ".flatbuffers"
            print("Unable to find " + flags.calibration_table + " Generating Table: " + calibration_table_name)
            provider_args["migraphx_int8_calibration_table_name"] = calibration_table_name

            # Generate INT8 calibration cache
            print("Calibration data compute starts with " + str(cal_ep))
            calibrator = create_calibrator(model_path, op_types_to_quantize, augmented_model_path=augmented_model_path, calibrate_method=CalibrationMethod.Percentile)
            calibrator.set_execution_providers([cal_ep]) 

            '''
            We can use one data reader to do data pre-processing, however,
            some machines don't have sufficient memory to hold all dataset and all intermediate output,
            especially using 'Entropy' or 'Percentile' calibrator which collects histogram for tensors.
            So let multiple data readers to handle different stride of dataset to avoid OOM.
            '''
            stride = 10
            #for i in range(0, calib_num, stride):
            data_reader = BertDataReader(model_path, input_dataset, input_tokens, 1, sequence_lengths[-1], flags.query_len, doc_stride[-1], start_index=0, end_index=calib_num)
            calibrator.collect_data(data_reader)

            compute_range = calibrator.compute_data()


            calibration_table = {}
            print("Writing calibration table")
            try:
                write_calibration_table(calibration_table)
            except AttributeError as e:
                class TensorDataWrapper:
                    def __init__(self, data_dict):
                        self.data_dict = data_dict

                    def to_dict(self):
                        return self.data_dict

                    def __repr__(self):
                        return repr(self.data_dict)

                    def __serializable__(self):
                        return self.data_dict

                calibration_data = {}
                for k, v in compute_range.data.items():
                    if hasattr(v, 'to_dict'):
                        tensor_dict = v.to_dict()
                        processed_dict = {}
                        for dk, dv in tensor_dict.items():
                            if isinstance(dv, np.ndarray):
                                processed_dict[dk] = dv.item() if dv.size == 1 else dv.tolist()
                            elif isinstance(dv, np.number):
                                processed_dict[dk] = dv.item()
                            else:
                                processed_dict[dk] = dv
                        calibration_data[k] = TensorDataWrapper(processed_dict)
                    else:
                        calibration_data[k] = v

                print("Using custom calibration table function")
                custom_write_calibration_table(calibration_data)

            print("Calibration is done. Calibration cache is saved to calibration.json")

            with open(flags.calibration_table, "w") as f:
                json.dump(calibration_table, f)
            print("Calibration is done. Calibration cache is saved to " + calibration_table_name)
            provider_args["migraphx_int8_calibration_table_name"] = calibration_table_name


        model_quants = model_quants + precision

        if flags.ort_quant:
            print(precision + " Quantization Done with Onnxruntime Quantizer")
            mode = QuantizationMode.QLinearOps
            # In TRT, it recommended to add QDQ pair to inputs of Add node followed by ReduceMean node.
            # Mirroring here what TRT does in MIGraphX Quantization to be able to perform an apples to apples comparison
            nodes_to_exclude = get_op_nodes_not_followed_by_specific_op(model, "Add", "ReduceMean")
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
                nodes_to_exclude,
                op_types_to_quantize,
                {'ActivationSymmetric' : True, 'AddQDQPairToWeight' : True, 'OpTypesToExcludeOutputQuantization': op_types_to_quantize, 'DedicatedQDQPair': True, 'QDQOpTypePerChannelSupportToAxis': {'MatMul': 1} }) #extra_options
            quantizer.quantize_model()
            quantizer.model.save_model_to_file(qdq_model_path, False) 
            print("QDQ model is saved to ", qdq_model_path)
        else:
            qdq_model_path = model_path
            print(precision + " Quantization Done with " + cal_ep)
            #Quantize with MIGraphX's INT8/FP8 quantizer instead 
    else:
        qdq_model_path = model_path

    # No fp16 cal needed, MIGraphX will handle that through Onnxruntime & MIGraphX Execution Provider during compile
    if flags.fp16:
        model_quants = model_quants + "_fp16"
        provider_args["migraphx_fp16_enable"] = str(True)

    model_name = ""
    if flags.save_load:
        model_name = str(qdq_model_path) + "_s" + str(flags.seq_len) + "_b" + str(flags.batch) + str(model_quants) + ".mxr"
        print("save load model from " + str(model_name))
        provider_args["migraphx_save_compiled_model"] = flags.save_load
        provider_args["migraphx_load_compiled_model"] = flags.save_load
        provider_args["migraphx_save_model_path"] = model_name
        provider_args["migraphx_load_model_path"] = model_name

    # QDQ model inference and get SQUAD prediction 
    batch_size = flags.batch 
    data_reader = BertDataReader(qdq_model_path, input_dataset, input_tokens, batch_size, sequence_lengths[-1], flags.query_len, doc_stride[-1], end_index=samples)
    sess_options = onnxruntime.SessionOptions()
    if flags.ort_verbose:
        sess_options.log_severity_level = 0
        sess_options.log_verbosity_level = 0

    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(qdq_model_path, sess_options=sess_options, 
                                                providers=[("MIGraphXExecutionProvider", provider_args)])
    
    print("Running Inferences")
    latency = [] #Used for timing information
    all_predictions = inference(data_reader, ort_session, latency, flags.verbose) 

    print("Inference Complete!")

    output_run_config(flags, samples)

    print("Rate = {} QPS ".format(
        format((((flags.batch) / (sum(latency[1:]) / len(latency[1:])))),
                '.2f')))
    print("Average Execution time = {} ms".format(
            format(sum(latency[1:]) * 1000 / len(latency[1:]), '.2f')))


    # Verify output result from run
    if not flags.no_eval:
        print(" Saving predictions")
        prediction_file = "./prediction.json"
        with open(prediction_file, "w") as f:
            f.write(json.dumps(all_predictions, indent=4))
            print("\nOutput dump to {}".format(prediction_file))

        version = flags.version
        if flags.version > 2.0:
            version = 2.0
        print("Evaluate QDQ model for SQUAD v"+ str(version))
        if version == 1.1:
            subprocess.call(['python3', './squad/evaluate-v'+ str(version) + '.py', './squad/dev-v' + str(version) + '.json', './prediction.json', '90'])
        else:
            subprocess.call(['python3', './squad/evaluate-v'+ str(version) + '.py', './squad/dev-v' + str(version) + '.json', './prediction.json'])
