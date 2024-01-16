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
import random
import torch
import logging
import argparse
import datasets
import numpy as np
import onnxruntime as ort
from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
from transformers import LlamaConfig, LlamaTokenizer
from onnxruntime.quantization import matmul_4bits_quantizer

logger = logging.getLogger(__name__)
logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.WARN)

parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_input",
    type=str,
    help="dirctory path of onnx model"
)
parser.add_argument(
    "--model_output",
    type=str,
    default=None,
    help="dirctory path of quantized onnx model"
)
parser.add_argument(
    "--benchmark",
    action="store_true",
    default=False,
    help="whether benchmark the model"
)
parser.add_argument(
    "--quantize",
    action="store_true",
    default=False,
    help="whether quantize the model"
)
parser.add_argument(
    "--batch_size",
    default=1,
    type=int,
)
parser.add_argument(
    "--workspace",
    type=str,
    help="workspace to save intermediate files",
    default="nc_workspace"
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="RTN",
    choices=["RTN", "GPTQ"],
    help="weight only algorithm"
)
parser.add_argument(
    "--pad_max",
    default=196,
    type=int,
)
parser.add_argument(
    "--seqlen",
    default=2048,
    type=int,
)
parser.add_argument(
    "--tasks",
    nargs="+",
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
    "--block_size",
    type=int, 
    default=32
)
parser.add_argument(
    "--is_symmetric",
    type=bool, 
    default=False,
    help="is symmetric or not"
)
parser.add_argument(
    "--accuracy_level",
    type=int, 
    default=None,
    help="accuracy level of the 4-bit quantized MatMul computation"
)
parser.add_argument(
    "--sampling_size",
    type=int, 
    default=8,
    help="sampling size of calibration for GPTQ"
)
args = parser.parse_args()

# load tokenizer and config
tokenizer = LlamaTokenizer.from_pretrained(args.model_input)
config = LlamaConfig.from_pretrained(args.model_input)

def tokenize_function(examples):
    example = tokenizer(examples["text"])
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

class GPTQDataloader:
    def __init__(self, model_input, batch_size=1, seqlen=2048, sub_folder="train", sampling_size=8):
        import random
        random.seed(0)
        self.seqlen = seqlen

        self.batch_size=batch_size
        traindata = datasets.load_dataset(args.dataset, split=sub_folder)
        traindata = traindata.map(tokenize_function, batched=True)
        traindata.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.traindata = traindata.select(range(sampling_size))
        self.sampling_size = sampling_size

        session = ort.InferenceSession(model_input)
        inputs_names = [input.name for input in session.get_inputs()]
        self.key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0

    def __iter__(self):
        try:
            for _ in range(self.sampling_size):
                while True:
                    i = random.randint(0, len(self.traindata) - 1)
                    trainenc = self.traindata[i]
                    if trainenc["input_ids"].shape[0] > self.seqlen:
                        break
                i = random.randint(0, trainenc["input_ids"].shape[0] - self.seqlen - 1)
                j = i + self.seqlen
                inp = trainenc["input_ids"][i:j].unsqueeze(0)
                mask = torch.ones(inp.shape)

                ort_input = {}
                if not self.use_cache:
                    ort_input["input_ids"] = inp.detach().cpu().numpy().astype("int64")
                    ort_input["attention_mask"] = mask.detach().cpu().numpy().astype("int64")
                else:
                    num_attention_heads = config.num_key_value_heads
                    embed_size_per_head = config.hidden_size // config.num_attention_heads
                    shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                    key_or_value = np.zeros(shape, dtype=np.float32)

                    for key_value_input_name in self.key_value_input_names:
                        ort_input[key_value_input_name] = key_or_value

                    ort_input["input_ids"] = inp[:, -1].unsqueeze(0).detach().cpu().numpy().astype("int64")
                    ort_input["attention_mask"] =  np.zeros([self.batch_size, ort_input["past_key_values.0.key"].shape[2]+1], dtype="int64")

                input_shape = ort_input["input_ids"].shape
                position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
                ort_input["position_ids"] = position_ids.numpy()

                yield ort_input
 
        except StopIteration:
            return

if __name__ == "__main__":
    from neural_compressor import set_workspace
    set_workspace(args.workspace)

    if args.benchmark:
        eval_func(args.model_input)

    if args.quantize:
        model_name = "model.onnx"
        model_file = os.path.join(args.model_input, model_name)

        if args.algorithm.upper() == "RTN":
            algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig()
        elif args.algorithm.upper() == "GPTQ":
            data_reader = GPTQDataloader(model_file, seqlen=args.seqlen, batch_size=1, sampling_size=args.sampling_size)
            algo_config = matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig(calibration_data_reader=data_reader)
        
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model_file, 
                                                            block_size=args.block_size, 
                                                            is_symmetric=args.is_symmetric, 
                                                            algo_config=algo_config,
                                                            accuracy_level=args.accuracy_level)
        quant.process()
        quant.model.save_model_to_file(os.path.join(args.model_output, model_name), use_external_data_format=True)
        tokenizer.save_pretrained(args.model_output)
        config.save_pretrained(args.model_output)
