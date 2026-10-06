"""Independent native GenAI sanity check. CPU here is validation, never serving."""
import json
import os
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=Path)
parser.add_argument("--provider", choices=["cpu", "webgpu"], default="cpu")
args = parser.parse_args()
root = args.model_dir.resolve()
notes = root.parent / (root.name + ".reports")
notes.mkdir(parents=True, exist_ok=True)
os.chdir(notes)
result = {"provider": args.provider, "purpose": "pre-browser export validation", "tests": []}
try:
    config = og.Config(str(root))
    config.clear_providers()
    if args.provider != "cpu":
        config.append_provider(args.provider)
    model = og.Model(config)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    for prompt in ["What is the capital of France?", "What is 12 * 13? Answer with the number only."]:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False,
        )
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(input_ids) + 32)
        start = time.perf_counter()
        generator = og.Generator(model, params)
        generator.append_tokens(np.asarray(input_ids, dtype=np.int32))
        generated = []
        first = None
        while not generator.is_done() and len(generated) < 32:
            generator.generate_next_token()
            generated.append(int(generator.get_next_tokens()[0]))
            if first is None:
                first = time.perf_counter()
        end = time.perf_counter()
        item = {"prompt": prompt, "input_ids": input_ids, "generated_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "first_token_ms": (first - start) * 1000 if first else None,
                "elapsed_ms": (end - start) * 1000}
        result["tests"].append(item)
        print(json.dumps(item, indent=2), flush=True)
        del generator
    result["checks"] = {
        "capital": "paris" in result["tests"][0]["text"].lower(),
        "arithmetic": "156" in result["tests"][1]["text"],
    }
    if not all(result["checks"].values()):
        raise RuntimeError("Native generation completed but failed the answer sanity checks")
    result["status"] = "passed"
except Exception as error:
    result["status"] = "failed"
    result["error"] = f"{type(error).__name__}: {error}"
    print(result["error"], flush=True)
finally:
    (notes / f"native-validation-{args.provider}.json").write_text(json.dumps(result, indent=2) + "\n")
if result["status"] != "passed":
    raise SystemExit(1)
