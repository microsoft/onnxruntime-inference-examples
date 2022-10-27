from dataclasses import dataclass
import numpy as np
import torch
from tqdm.auto import tqdm
import os
from torch.utils.data import DataLoader, SequentialSampler

from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from qat_bert import quantize_bert_model

from qat_utils import remove_qconfig_for_module
def remove_qconfig_before_convert(model):
    # Do not quantize subgraph that not in the bert part
    remove_qconfig_for_module(model, '', ['classifier', 'bert.pooler'], remove_subtree=True)

OUTPUT_PREFIX="../BertModel"
task = "imdb"
model_id = "bert-base-uncased"  # or distilbert-base-uncased
training_output_dir = f"{OUTPUT_PREFIX}/{model_id}-finetuned-{task}-training"
qat_training_output_dir = f"{OUTPUT_PREFIX}/{model_id}-finetuned-{task}-qat-training"
model_save_dir = f"{OUTPUT_PREFIX}/{model_id}-finetuned-{task}-saved"
qat_model_save_dir = f"{OUTPUT_PREFIX}/{model_id}-finetuned-{task}-qat-saved"
onnx_fp32 = f"{OUTPUT_PREFIX}/{model_id}-finetuned-{task}.onnx"
qat_onnx_model = f"{OUTPUT_PREFIX}/{model_id}-finetuned-{task}-qat.onnx"

metric = None
config = None
tokenizer = None
raw_test_dataset = None
train_eval_datasets = None
train_eval_datasets_encoded = None
dataset_test_encoded = None
prepared = False

def preprocess_function_batch(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

def prepare_resources():
    global metric, config, tokenizer, raw_test_dataset, train_eval_datasets, train_eval_datasets_encoded, dataset_test_encoded, prepared

    if not prepared:
        metric = load_metric("accuracy")
        config = AutoConfig.for_model('bert')
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        train_eval_raw = load_dataset(task, split='train').shuffle(86).select(range(1000))
        train_eval_datasets = train_eval_raw.train_test_split(test_size=0.20)
        train_eval_datasets_encoded = train_eval_datasets.map(preprocess_function_batch, batched=False)

        raw_test_dataset = load_dataset(task, split='test').shuffle(68).select(range(200))
        dataset_test_encoded = raw_test_dataset.map(preprocess_function_batch, batched=False)
        prepared = True
    

args = TrainingArguments(
    output_dir=training_output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    learning_rate=3e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Function that will be called at the end of each evaluation phase on the whole
# arrays of predictions/labels to produce metrics.
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = metric.compute(predictions=predictions, references=labels)
    return results

def get_trainer(model):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_eval_datasets_encoded["train"],
        eval_dataset=train_eval_datasets_encoded["test"].shuffle(36),  #.select(range(1600)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer


def export_model(model, onnx_model_path):
    trainer = get_trainer(model)
    dataloader = trainer.get_test_dataloader(dataset_test_encoded)

    model = trainer.model
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = next(iter(dataloader))
        torch.onnx.export(
            model, 
            (inputs['input_ids'].to(device), inputs['attention_mask'].to(device)),
            f=onnx_model_path,
            input_names=['input_ids', 'attention_mask'], 
            output_names=['logits'], 
            dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                        'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                        'logits': {0: 'batch_size', 1: 'sequence'}}, 
            do_constant_folding=True, 
            opset_version=13,
        )
    print(f"===={onnx_model_path} generated, please using:")
    print(f"python -m onnxruntime.transformers.optimizer --input {onnx_model_path} "
           "--output (YOUR_OPT_INT8_MODLE_NAME) --num_heads 12")


def export_saved_model(model_save_dir, onnx_model_path):
    print(f"=================================================================")
    print(f"====Export fine tuned model from dir: {model_save_dir} to {onnx_fp32}")
    print(f"=================================================================")
    model = AutoModelForSequenceClassification.from_pretrained(model_save_dir)
    export_model(model, onnx_model_path)


def fine_tune_fp32():
    print(f"=================================================================")
    print(f"====Fine tuning from pretrained model {model_id}")
    print(f"=================================================================")
    # Sikp warning of not used weights and randomly initialized weights.
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    trainer = get_trainer(model)
    trainer.train()
    print(f"====Saving fine tuned model into dir: {model_save_dir}")
    trainer.save_model(model_save_dir)


def qat_fine_tune():
    print(f"=================================================================")
    print(f"====QAT fine tune model from saved dir: {model_save_dir}.........")
    print(f"=================================================================", "")    
    model = AutoModelForSequenceClassification.from_pretrained(model_save_dir)
    model = quantize_bert_model(model, remove_qconfig_before_convert)
    trainer = get_trainer(model)
    trainer.train()

    print(f"====Saving fine tuned model into dir: {qat_model_save_dir}")
    trainer.save_model(qat_model_save_dir)


def qat_evaluate():
    print(f"=================================================================")
    print(f"====QAT evaluate model from saved dir: {qat_model_save_dir}.........")
    print(f"=================================================================", "")    
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model = quantize_bert_model(model, remove_qconfig_before_convert)
    print(f"====Loading qat model state_dict from {qat_model_save_dir}/pytorch_model.bin .........", "")
    model.load_state_dict(torch.load(os.path.join(qat_model_save_dir, 'pytorch_model.bin')))

    model.apply(torch.quantization.disable_observer)
    trainer = get_trainer(model)
    test_predictions = trainer.predict(dataset_test_encoded)
    test_predictions_argmax = np.argmax(test_predictions[0], axis=1)
    test_references = np.array(raw_test_dataset["label"])
    accuracy = metric.compute(predictions=test_predictions_argmax, references=test_references)
    print(f"====Accuracy is: {accuracy}")


def qat_export():
    print(f"=================================================================")
    print(f"====QAT evaluate model from saved dir: {qat_model_save_dir}.........")
    print(f"=================================================================", "")    
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model = quantize_bert_model(model, remove_qconfig_before_convert)
    print(f"====Loading qat model state_dict from {qat_model_save_dir}/pytorch_model.bin .........", "")
    model.load_state_dict(torch.load(os.path.join(qat_model_save_dir, 'pytorch_model.bin')))

    print(f"====Exporting qat onnx model.........", "")
    model.apply(torch.quantization.disable_observer)
    export_model(model, qat_onnx_model)


def eval_fine_tuned():
    print(f"=================================================================")
    print(f"====Evaluating fine tuned model from saved dir: {model_save_dir}")
    print(f"=================================================================")
    model = AutoModelForSequenceClassification.from_pretrained(model_save_dir)
    trainer = get_trainer(model)
    test_predictions = trainer.predict(dataset_test_encoded)
    test_predictions_argmax = np.argmax(test_predictions[0], axis=1)
    test_references = np.array(raw_test_dataset["label"])
    accuracy = metric.compute(predictions=test_predictions_argmax, references=test_references)
    print(f"====Accuracy is: {accuracy}")


def eval_onnx_model(onnx_model, padding_base = 0, custom_lib = None, providers = ["CUDAExecutionProvider"],):
    print(f"=================================================================")
    print(f"====Evaluating onnx tuned model: {onnx_model}")
    print(f"=================================================================")
    import onnxruntime
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if custom_lib is not None:
        sess_options.register_custom_ops_library(custom_lib)
    session = onnxruntime.InferenceSession(onnx_model, sess_options, providers=providers)

    test_predictions = None
    test_references = None

    def collate_fn(examples):
        return tokenizer.pad(examples)
    tds = dataset_test_encoded.remove_columns(['text'])
    dataloader = DataLoader(tds, sampler=SequentialSampler(tds), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    for inputs in tqdm(dataloader):
        np_labels = np.array(inputs["label"])
        test_references = np_labels if test_references is None else np.concatenate((test_references, np_labels), axis=0)
        input_ids = np.array(inputs['input_ids'], dtype=np.int64)
        attention_mask = np.array(inputs['attention_mask'], dtype=np.int64)
        if (padding_base > 0):
            seq_len = (input_ids.shape[-1] + 15) // 16 * 16
            pad_width = seq_len - input_ids.shape[-1]
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)))
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)))
        onnx_outputs = session.run(['logits'], {'input_ids' : input_ids, 'attention_mask' : attention_mask})
        logits = onnx_outputs[0]
        test_predictions = logits if test_predictions is None else np.concatenate((test_predictions, logits), axis=0)
    test_predictions_argmax = np.argmax(test_predictions, axis=1)
    accuracy = metric.compute(predictions=test_predictions_argmax, references=test_references)
    print(f"====Accuracy is: {accuracy}")


@dataclass
class ActionArgs:
    do_fp32_fine_tune : bool = False
    do_fp32_eval :  bool = False
    do_fp32_export : bool = False
    do_fp32_onnx_eval : bool = False
    do_fp32_all : bool = False
    do_qat_fine_tune : bool = False
    do_qat_export : bool = False
    do_qat_eval : bool = False
    do_qat_all : bool = False
    do_qat_onnx_eval : bool = False
    qat_opt_onnx_model : str = None


def main():
    parser = HfArgumentParser(ActionArgs)
    actions, = parser.parse_args_into_dataclasses()

    prepare_resources()
    if actions.do_fp32_fine_tune or actions.do_fp32_all:
        fine_tune_fp32()
    if actions.do_fp32_eval or actions.do_fp32_all:
        eval_fine_tuned()
    if actions.do_fp32_export or actions.do_fp32_all:
        export_saved_model(model_save_dir, onnx_fp32)
    if actions.do_fp32_onnx_eval or actions.do_fp32_all:
        eval_onnx_model(onnx_fp32)

    # change training output dir for qat
    args.output_dir = qat_training_output_dir

    if actions.do_qat_fine_tune or actions.do_qat_all:
        qat_fine_tune()
    if actions.do_qat_eval or actions.do_qat_all:
        qat_evaluate()
    if actions.do_qat_export or actions.do_qat_all:
        qat_export()
    if actions.do_qat_onnx_eval or actions.do_qat_all:
        eval_onnx_model(qat_onnx_model if actions.qat_opt_onnx_model is None else actions.qat_opt_onnx_model, padding_base=16)

if __name__ == "__main__":
    main()
