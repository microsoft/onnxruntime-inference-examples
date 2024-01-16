# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation
import os
import torch
import logging
import argparse
import numpy as np
import onnxruntime as ort
from datasets import load_dataset
import onnxruntime as ort
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
from transformers import LlamaConfig, LlamaTokenizer
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--model_input',
    type=str,
    help="Folder path of onnx model"
)
parser.add_argument(
    '--benchmark',
    action='store_true', \
    default=False,
    help="whether benchmark the model"
)
parser.add_argument(
    '--quantize',
    action='store_true', \
    default=False,
    help="whether quantize the model"
)
parser.add_argument(
    '--model_output',
    type=str,
    default=None,
    help="Folder path of quantized onnx model "
)
parser.add_argument(
    '--batch_size',
    default=1,
    type=int,
)
parser.add_argument(
    '--workspace',
    type=str,
    help="workspace to save intermediate files",
    default="nc_workspace"
)
parser.add_argument(
    '--quant_format',
    type=str,
    default='QOperator', 
    choices=['QOperator', 'QDQ'],
    help="quantization format"
)
parser.add_argument(
    '--pad_max',
    default=196,
    type=int,
)
parser.add_argument(
    "--tasks",
    nargs='+',
    default=["winogrande", "copa", "piqa", "rte", "hellaswag", "openbookqa", \
             "lambada_openai", "lambada_standard", "wikitext"],
    type=str,
    help="tasks list for accuracy validation"
)
parser.add_argument(
    "--dataset",
    nargs="?",
    default="NeelNanda/pile-10k",
    const="NeelNanda/pile-10k"
)
parser.add_argument(
    "--smooth_quant_alpha",
    type=float,
    default=0.6
)
parser.add_argument(
    "--sampling_size",
    type=int, 
    default=8,
    help="sampling size of calibration"
)
args = parser.parse_args()

# load tokenizer and config
tokenizer = LlamaTokenizer.from_pretrained(args.model_input)
config = LlamaConfig.from_pretrained(args.model_input)

def tokenize_function(examples):
    example = tokenizer(examples['text'])
    return example

def eval_func(model):
    logger.info("start to evaluate onnx model ...")
    results = evaluate(
        model="hf-causal",
        model_args="pretrained=" + model + ",tokenizer="+ args.model_input,
        batch_size=args.batch_size,
        tasks=args.tasks,
        model_format="onnx"
    )
    for task_name in args.tasks:
        if task_name == "wikitext":
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]))
        else:
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]))

class CalibDataloader(CalibrationDataReader):
    def __init__(self, model_path, pad_max=196, batch_size=1, sub_folder='train', sampling_size=8):
        self.pad_max = pad_max
        self.batch_size=batch_size
        dataset = load_dataset(args.dataset, split=sub_folder)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset = dataset.select(range(sampling_size))
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

        session = ort.InferenceSession(model_path)
        inputs_names = [input.name for input in session.get_inputs()]
        self.key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0

        self.processed_data = iter(self.process_data(dataloader))

    def collate_batch(self, batch):
        input_ids_padded = []
        attention_mask_padded = []
        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
        return torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)
    
    def process_data(self, dataloader):
        processed_data = []
        for (input_ids, attention_mask) in dataloader:
            ort_input = {}
            if not self.use_cache:
                ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
                ort_input["attention_mask"] = attention_mask[:, :-1].detach().cpu().numpy().astype('int64')
            else:
                num_attention_heads = config.num_key_value_heads
                embed_size_per_head = config.hidden_size // config.num_attention_heads
                shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                key_or_value = np.zeros(shape, dtype=np.float32)

                for key_value_input_name in self.key_value_input_names:
                    ort_input[key_value_input_name] = key_or_value

                ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
                ort_input["attention_mask"] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')

            input_shape = ort_input["input_ids"].shape
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
            ort_input["position_ids"] = position_ids.numpy()
            processed_data.append(ort_input)
        return processed_data


    def get_next(self) -> dict:
        return next(self.processed_data, None)
        # res = next(self.dataloader, None)
        # if res is not None:
        #     input_ids, attention_mask = res
        #     ort_input = {}
        #     if not self.use_cache:
        #         ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
        #         ort_input["attention_mask"] = attention_mask[:, :-1].detach().cpu().numpy().astype('int64')
        #     else:
        #         num_attention_heads = config.num_key_value_heads
        #         embed_size_per_head = config.hidden_size // config.num_attention_heads
        #         shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
        #         key_or_value = np.zeros(shape, dtype=np.float32)

        #         for key_value_input_name in self.key_value_input_names:
        #             ort_input[key_value_input_name] = key_or_value

        #         ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
        #         ort_input["attention_mask"] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')

        #     input_shape = ort_input["input_ids"].shape
        #     position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        #     ort_input["position_ids"] = position_ids.numpy()
        #     return ort_input
        # else:
        #     return None


if __name__ == "__main__":
    from neural_compressor import set_workspace
    set_workspace(args.workspace)

    if args.benchmark:
        eval_func(args.model_input)

    if args.quantize:
        model_name = "model.onnx"
        model_file = os.path.join(args.model_input, model_name)
        output_model_file = os.path.join(args.model_output, model_name)

        data_reader = CalibDataloader(model_file, pad_max=args.pad_max, batch_size=1)

        quantize_static(model_file,
                        output_model_file, 
                        calibration_data_reader=data_reader,
                        quant_format="QOperator",
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        op_types_to_quantize=["MatMul"],
                        use_external_data_format=True,
                        extra_options={"SmoothQuant": True,
                                       "SmoothQuantAlpha": args.smooth_quant_alpha,
                                       "OpTypesToExcludeOutputQuantization": ["MatMul"]})
        tokenizer.save_pretrained(args.model_output)
        config.save_pretrained(args.model_output)
