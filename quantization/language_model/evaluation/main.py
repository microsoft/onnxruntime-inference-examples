import os
import json
import logging
import argparse
from lm_eval import utils, tasks, evaluator, base

from ort_lm import ORTCausalLM




logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_args',
        type=str,
        help="Folder path of pre-trained onnx model and additional arguments. E.g pretrained=llama-2-7b-onnx.py"
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
    )
    parser.add_argument(
        "--tasks", help="tasks list for accuracy validation"
    )
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


def main(args):
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


if __name__ == "__main__":
    args = parse_args()
    main(args)

