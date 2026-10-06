"""Compare original/adapted graph logits and every cache after prefill/decode."""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

root = Path(sys.argv[1]).resolve()
notes = root.parent / (root.name + ".reports")
notes.mkdir(parents=True, exist_ok=True)
config = json.loads((root / "genai_config.json").read_text())
manifest = json.loads((root / "manifest.json").read_text())
graph = next(g for g in manifest["graphs"] if g["file"] == config["model"]["decoder"]["filename"])
original = ort.InferenceSession(str(root / graph["file"]), providers=["CPUExecutionProvider"])
adapted = ort.InferenceSession(str(root / "model.browser.onnx"), providers=["CPUExecutionProvider"])
reference = json.loads((notes / "native-validation-cpu.json").read_text())["tests"][0]
symbols = {"batch_size": 1, "past_sequence_length": 0, "kv_cache_dim": config["model"]["decoder"]["head_size"]}
state = {}
for value in graph["inputs"]:
    if value["name"].startswith("past"):
        shape = [symbols.get(d, d) for d in value["dims"]]
        state[value["name"]] = np.zeros(shape, dtype=np.float16)
ids = reference["input_ids"]
past = 0
checks = []
for step in range(2):
    total = past + len(ids)
    feeds = {**state, "input_ids": np.asarray([ids], dtype=np.int64),
             "position_ids": np.tile(np.arange(past, total, dtype=np.int64), (3, 1, 1)),
             "attention_mask": np.ones((1, total), dtype=np.int64)}
    original_outputs = dict(zip([v.name for v in original.get_outputs()], original.run(None, feeds)))
    browser_feeds = {k: v for k, v in feeds.items() if k != "attention_mask"}
    browser_feeds.update(seqlens_k=np.asarray([total - 1], dtype=np.int32),
                         total_sequence_length=np.asarray(total, dtype=np.int32))
    browser_outputs = dict(zip([v.name for v in adapted.get_outputs()], adapted.run(None, browser_feeds)))
    equal = {name: np.array_equal(value, browser_outputs[name]) for name, value in original_outputs.items()}
    assert all(equal.values()), f"Graph adaptation changed outputs: {equal}"
    checks.append({"step": step, "inputTokens": len(ids), "outputsCompared": len(equal), "allBitwiseEqual": True})
    for name in state:
        output = name.replace("past_key_values.", "present.").replace("past.", "present.")
        state[name] = original_outputs[output]
    ids = [int(np.argmax(original_outputs["logits"][0, -1]))]
    past = total
result = {"status": "passed", "provider": "cpu", "purpose": "offline graph-adaptation equivalence check", "checks": checks}
(notes / "adaptation-equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
