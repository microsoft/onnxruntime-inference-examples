import os
import json
import logging
import argparse
from lm_eval import utils, tasks, evaluator, base
import onnxruntime as ort
import gc
import time

from ort_lm import ORTCausalLM

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARN)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_args',
        type=str,
        required=True,
        help="Folder path of pre-trained onnx model and additional arguments. E.g pretrained=llama-2-7b-onnx.py"
    )
    parser.add_argument(
        "--mode",
        default=None,
        required=True,
        choices=["perf", "acc"],
        help="Choose perf to measure performance, choose acc to measure accuracy")
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
    )
    parser.add_argument("--profiling", action="store_true",
                        help="Get a memory and runtime profile for an ORT model evaluation")
    parser.add_argument("--perf_batch", default=10, type=int)
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    return parser.parse_args()


def acc_main(args):
    model = "ort-causal"
    if args.tasks:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    else:
        task_names = tasks.ALL_TASKS

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    additional_config = {"batch_size": args.batch_size, "max_batch_size": args.max_batch_size, "device": args.device}
    model_args = utils.simple_parse_args_string(args.model_args)
    model_args2 = {k: v for k, v in additional_config.items() if v is not None}

    lm_model = ORTCausalLM(
        **model_args,
        **model_args2
    )

    if not args.no_cache:
        lm_model = base.CachingLM(
            lm_model,
            "lm_cache/"
            + (model if isinstance(model, str) else model.model.config._name_or_path)
            + "_"
            + args.model_args.replace("=", "-").replace(",", "_").replace("/", "-")
            + ".db",
        )

    task_dict = tasks.get_task_dict(task_names)

    if args.check_integrity:
        utils.run_task_tests(task_list=tasks)

    results = evaluator.evaluate(
        lm=lm_model,
        task_dict=task_dict,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        bootstrap_iters=10000,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    # add info about the model and few shot config
    model_name = None
    if isinstance(model, str):
        model_name = model
    results["config"] = {
        "model": model_name,
        "model_args": args.model_args,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "batch_sizes": list(lm_model.batch_sizes.values()) if hasattr(lm_model, "batch_sizes") else [],
        "device": args.device,
        "no_cache": args.no_cache,
        "limit": args.limit,
        "bootstrap_iters": 10000,
        "description_dict": description_dict,
    }

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))


def perf_main(args):
    additional_config = {"batch_size": args.batch_size, "max_batch_size": args.max_batch_size, "device": args.device}
    model_args = utils.simple_parse_args_string(args.model_args)
    model_args2 = {k: v for k, v in additional_config.items() if v is not None}

    options = ort.SessionOptions()
    if args.profiling:
        options.enable_profiling = True
    lm_model = ORTCausalLM(
        **model_args,
        **model_args2,
        session_options=options,
    )

    n_batch = args.perf_batch

    prompt_lengths = {64: {},
                      128: {},
                      256: {},
                      512: {},
                      1024: {}}

    new_token_lengths = [1, 129]

    if args.profiling:
        prompt_lengths = {64: {}}
        new_token_lengths = [1]

    for prompt_length in prompt_lengths.keys():
        times = {}
        for new_token_length in new_token_lengths:

            print(f"prompt_numbers = {prompt_length}, new_token_length = {new_token_length}:")

            prompt = "happy " * prompt_length

            ## ONNX Runtime
            inputs = lm_model.tokenizer(
                prompt,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            gen_tokens_length = inputs.input_ids.shape[-1] + new_token_length

            # warmup
            _ = lm_model.model.generate(
                **inputs,
                min_length=gen_tokens_length,
                max_length=gen_tokens_length
            )

            gc.collect()
            gc.disable()
            start = time.time()
            for _ in range(n_batch):
                _ = lm_model.model.generate(
                    **inputs,
                    min_length=gen_tokens_length,
                    max_length=gen_tokens_length
                )
            end = time.time()
            runtime = round((end - start) / n_batch, 3)
            times[new_token_length] = runtime
            gc.enable()
            print(f"ORT: {runtime} s")
        if args.profiling: continue
        prompt_lengths[prompt_length]["per token cost"] = round((times[129] - times[1]) / 128, 8)
        prompt_lengths[prompt_length]["prompt cost"] = round(times[1] - prompt_lengths[prompt_length]["per token cost"],
                                                             8)
    for k, v in prompt_lengths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "acc":
        acc_main(args)
    elif args.mode == "perf":
        perf_main(args)
    else:
        raise ValueError(f"{args.mode} is inaccurate mode selection. Select 'perf' or 'acc'")
