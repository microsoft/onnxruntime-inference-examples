"""Read graph metadata without loading external weights; emit a browser manifest."""
import json
import sys
from collections import Counter
from pathlib import Path

import onnx

root = Path(sys.argv[1]).resolve()
notes = root.parent / (root.name + ".reports")
notes.mkdir(parents=True, exist_ok=True)

def tensor_info(value):
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "type": onnx.TensorProto.DataType.Name(tensor.elem_type).lower(),
        "dims": [d.dim_param if d.dim_param else d.dim_value for d in tensor.shape.dim],
    }

graphs = []
for path in sorted(root.glob("*.onnx")):
    model = onnx.load(path, load_external_data=False)
    data_files = set()
    for tensor in model.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                data_files.add(item.value)
    graphs.append({
        "file": path.name,
        "inputs": [tensor_info(v) for v in model.graph.input],
        "outputs": [tensor_info(v) for v in model.graph.output],
        "externalData": sorted(data_files),
        "operators": dict(sorted(Counter(f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in model.graph.node).items())),
        "opsets": {v.domain or "ai.onnx": v.version for v in model.opset_import},
        "int4": [{"name": n.name, "bits": onnx.helper.get_attribute_value(next(a for a in n.attribute if a.name == "bits")),
                  "blockSize": onnx.helper.get_attribute_value(next(a for a in n.attribute if a.name == "block_size"))}
                 for n in model.graph.node if n.op_type == "MatMulNBits"],
        "largestInitializers": sorted(
            [{"name": t.name, "dims": list(t.dims), "type": onnx.TensorProto.DataType.Name(t.data_type).lower(),
              "externalBytes": int(next((v.value for v in t.external_data if v.key == "length"), "0"))}
             for t in model.graph.initializer], key=lambda t: t["externalBytes"], reverse=True)[:10],
    })
files = [{"name": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"]
manifest = {"files": files, "totalBytes": sum(f["bytes"] for f in files), "graphs": graphs}
if (root / "browser-adaptation.json").exists():
    manifest["browserAdaptation"] = json.loads((root / "browser-adaptation.json").read_text())
    manifest["browserGraph"] = manifest["browserAdaptation"]["browserGraph"]
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(notes / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
for file in files:
    print(f"{file['bytes']:>14,} bytes  {file['name']}")
print(f"Total: {manifest['totalBytes']:,} bytes")
for graph in graphs:
    print(json.dumps(graph, indent=2))
