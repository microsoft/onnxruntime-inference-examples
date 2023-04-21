# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import transformers
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
from contextlib import contextmanager



def get_model_from_huggingface(model_name: str = "csarron/mobilebert-uncased-squad-v2"):
    """
    Step 1. Download the model from huggingface and convert to onnx
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    config = transformers.AutoConfig.from_pretrained(model_name)
    model = transformers.MobileBertForQuestionAnswering.from_pretrained(model_name)
    onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(config, "question-answering")

    model_path = Path('app/src/main/res/raw/csarron_mobilebert_uncased_squad_v2.onnx')
    onnx_inputs, onnx_outputs = transformers.onnx.export(tokenizer, model, onnx_config, 16, model_path)
    return model_path


def quantize_model(model_path: Path):
    """
    Step 2. Quantize the model, so that it can be run on mobile devices with smaller memory footprint
    """
    quantized_model_path = model_path.with_name(model_path.stem+"_quant").with_suffix(model_path.suffix)
    quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QInt8)
    model_path.unlink()
    return quantized_model_path


def add_pre_post_process(quantized_model_path: Path, model_name: str = "csarron/mobilebert-uncased-squad-v2"):
    """
    Step 3. Add pre and post processing to the model, for tokenization and post processing
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    @contextmanager
    def temp_vocab_file():
        vocab_file = quantized_model_path.parent / "vocab.txt"
        yield vocab_file
        vocab_file.unlink()

    tokenizer_type = 'BertTokenizer'
    task_type = 'QuestionAnswering'
    output_model_path = quantized_model_path.with_name(
        quantized_model_path.stem+'_with_pre_post_processing').with_suffix(quantized_model_path.suffix)
    with temp_vocab_file() as vocab_file:
        import json
        with open(str(vocab_file), 'w') as f:
            f.write(json.dumps(tokenizer.vocab))
        add_ppp.transformers_and_bert(quantized_model_path, output_model_path, vocab_file, tokenizer_type, task_type)
    quantized_model_path.unlink()
    return output_model_path


if __name__ == "__main__":
    model = get_model_from_huggingface()
    quantized_model = quantize_model(model)
    output_model = add_pre_post_process(quantized_model)
