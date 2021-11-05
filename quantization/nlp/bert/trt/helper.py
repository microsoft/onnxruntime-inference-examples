import data_processing as dp
import tokenization
import collections
import subprocess
import json
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

class BertDataReader(CalibrationDataReader):
    def __init__(self,
                 model_path,
                 squad_json,
                 vocab_file,
                 batch_size,
                 max_seq_length,
                 start_index=0,
                 end_index=0):
        self.model_path = model_path
        self.data = dp.read_squad_json(squad_json)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.example_stride = batch_size # number of examples as one example stride. (set to equal to batch size) 
        self.start_index = start_index # squad example index to start with
        self.end_index = len(self.data) if end_index == 0 else end_index 
        self.current_example_index = start_index
        self.current_feature_index = 0 # global feature index (one example can have more than one feature) 
        self.tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.doc_stride = 128
        self.max_query_length = 64
        self.enum_data_dicts = iter([])
        self.features_list = []
        self.token_list = []
        self.example_id_list = []
        self.start_of_new_stride = False # flag to inform that it's a start of new example stride

    def update_load_range(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = len(self.data) if end_index == 0 else end_index 
        self.current_example_index = start_index
        self.enum_data_dicts = iter([])
        self.features_list = []
        self.token_list = []
        self.example_id_list = []
        self.start_of_new_stride = False # flag to inform that it's a start of new example stride

    def get_next(self):
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

        if self.current_example_index % 10 == 0:
            current_batch = int(self.current_feature_index / self.batch_size) 
            print("Reading example index {:}, batch {:}, containing {:} sentences".format(self.current_example_index, current_batch, self.batch_size))

        # example could have more than one feature
        # we collect all the features of examples and process them in one example stride
        features_in_current_stride = []
        for i in range(self.example_stride):
            example = self.data[self.current_example_index+ i]
            features = dp.convert_example_to_features(example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
            self.example_id_list.append(example.id)
            self.features_list.append(features)
            self.token_list.append(example.doc_tokens)
            features_in_current_stride += features
        self.current_example_index += self.example_stride
        self.current_feature_index+= len(features_in_current_stride)


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
                    break
                feature = features_in_current_stride[feature_idx + i]
                if len(input_ids) and len(segment_ids) and len(input_mask):
                    input_ids = np.vstack([input_ids, feature.input_ids])
                    input_mask = np.vstack([input_mask, feature.input_mask])
                    segment_ids = np.vstack([segment_ids, feature.segment_ids])
                else:
                    input_ids = np.expand_dims(feature.input_ids, axis=0)
                    input_mask = np.expand_dims(feature.input_mask, axis=0)
                    segment_ids = np.expand_dims(feature.segment_ids, axis=0)

            data.append({"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids":segment_ids})
            # data.append({"input_ids": input_ids, "input_mask": input_mask, "segment_ids":segment_ids})

        self.enum_data_dicts = iter(data)
        self.start_of_new_stride = True
        return next(self.enum_data_dicts, None)


class BertEvaluater:
    def __init__(self,
                 ort_session,
                 data_reader):
        self.ort_session = ort_session
        self.data_reader = data_reader

    def evaluate(self):
        all_predictions = self.inference(self.data_reader, self.ort_session)

        prediction_file = "./prediction.json"
        with open(prediction_file, "w") as f:
            f.write(json.dumps(all_predictions, indent=4))
            print("\nOutput dump to {}".format(prediction_file))

        print("Evaluate QDQ model for SQUAD v1.1")
        subprocess.call(['python', './bert-squad/evaluate-v1.1.py', './bert-squad/dev-v1.1.json', './prediction.json', '90'])

        # uncomment following code if you want to evaluate on SQUAD v2.0
        # you also need to re-run QDQ model inference to get prediction based on dev-v2.0.json
        #
        # print("Evaluate QDQ model for SQUAD v2.0")
        # subprocess.call(['python', './bert-squad/evaluate-v2.0.py', './bert-squad/dev-v2.0.json', './prediction.json'])

    def inference(self, data_reader, ort_session):

        _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
                "NetworkOutput",
                ["start_logits", "end_logits", "feature_index"])
        all_predictions = collections.OrderedDict()
        
        example_id_in_current_stride = [] 
        features_in_current_stride = []  
        token_list_in_current_stride = []
        outputs = []
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break

            if data_reader.start_of_new_stride:
                self.get_predictions(example_id_in_current_stride, features_in_current_stride, token_list_in_current_stride, data_reader.batch_size, outputs, _NetworkOutput, all_predictions)

                # reset current example stride
                example_id_in_current_stride = data_reader.example_id_list[-data_reader.example_stride:]
                features_in_current_stride = data_reader.features_list[-data_reader.example_stride:] 
                token_list_in_current_stride = data_reader.token_list[-data_reader.example_stride:]
                outputs = []

            output = ort_session.run(["output_start_logits","output_end_logits"], inputs)
            outputs.append(output)


        # handle the last example stride
        self.get_predictions(example_id_in_current_stride, features_in_current_stride, token_list_in_current_stride, data_reader.batch_size, outputs, _NetworkOutput, all_predictions)

        return all_predictions

    def get_predictions(self,
                        example_id_in_current_stride,
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
