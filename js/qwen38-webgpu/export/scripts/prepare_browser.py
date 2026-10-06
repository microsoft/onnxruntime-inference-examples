"""Expose GQA sequence metadata as host inputs; leave model math/weights intact.

This adapter only supports batch size one, contiguous unpadded text. It removes
the six attention-mask metadata nodes that ORT explicitly places on CPU. The
original GenAI export is retained for native validation.
"""
import json
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

root = Path(sys.argv[1]).resolve()
notes = root.parent / (root.name + ".reports")
notes.mkdir(parents=True, exist_ok=True)
config = json.loads((root / "genai_config.json").read_text())
original = config["model"]["decoder"]["filename"]
model = onnx.load(root / original, load_external_data=False)
prefix = "/model/attn_mask_reformat/attn_mask_subgraph/"
nodes = [n for n in model.graph.node if n.name.startswith(prefix)]
expected = {"ReduceSum", "Sub", "Sub/Cast", "Shape", "Gather", "Gather/Cast"}
assert {n.name.removeprefix(prefix) for n in nodes} == expected, "Unknown mask subgraph; inspect before adapting"
replacements = {
    prefix + "Sub/Cast/output_0": "seqlens_k",
    prefix + "Gather/Cast/output_0": "total_sequence_length",
}
removed_outputs = {o for n in nodes for o in n.output}
remaining = [n for n in model.graph.node if n not in nodes]
gqa = [n for n in remaining if n.op_type == "GroupQueryAttention"]
assert gqa and all(n.input[5] in replacements and n.input[6] in replacements for n in gqa)
for node in remaining:
    for index, name in enumerate(node.input):
        if name in replacements:
            node.input[index] = replacements[name]
        else:
            assert name not in removed_outputs, f"Unexpected mask-subgraph consumer: {node.name}"
del model.graph.node[:]
model.graph.node.extend(remaining)
mask = config["model"]["decoder"]["inputs"]["attention_mask"]
inputs = [v for v in model.graph.input if v.name != mask]
inputs.extend([
    helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, ["batch_size"]),
    helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []),
])
del model.graph.input[:]
model.graph.input.extend(inputs)
values = [v for v in model.graph.value_info if v.name not in removed_outputs]
del model.graph.value_info[:]
model.graph.value_info.extend(values)
browser = "model.browser.onnx"
onnx.save_model(model, root / browser)
# ORT's exporter uses an ai.onnx SimplifiedLayerNormalization extension that the
# generic ONNX checker may not recognize. Require the same failure on the untouched
# export; semantic validation is performed by check_adaptation.py through ORT.
checker_warning = None
try:
    onnx.checker.check_model(str(root / browser))
except onnx.checker.ValidationError as error:
    try:
        onnx.checker.check_model(str(root / original))
    except onnx.checker.ValidationError as baseline:
        if str(error).split("\n")[0] != str(baseline).split("\n")[0]:
            raise
        checker_warning = str(error).split("\n")[0]
    else:
        raise
adaptation = {
    "originalGraph": original, "browserGraph": browser,
    "removedNodes": [{"name": n.name, "op": n.op_type} for n in nodes],
    "newInputs": {"seqlens_k": "int32[1] = total tokens processed including this step - 1",
                  "total_sequence_length": "int32 scalar = total tokens processed including this step"},
    "restriction": "batch size 1, contiguous unpadded text only",
    "weightsModified": False,
    "onnxCheckerWarningAlsoPresentInOriginal": checker_warning,
}
(root / "browser-adaptation.json").write_text(json.dumps(adaptation, indent=2) + "\n")
(notes / "browser-adaptation.json").write_text(json.dumps(adaptation, indent=2) + "\n")
print(json.dumps(adaptation, indent=2))
