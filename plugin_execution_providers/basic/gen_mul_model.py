from pathlib import Path
import sys

import onnx
from onnxscript import script, FLOAT, opset15 as op

def gen_mul_model(model_path: Path):
    @script(default_opset=op)
    def model(x: FLOAT[2, 3], y: FLOAT[2, 3]) -> FLOAT[2, 3]:
        return x * y

    model_proto = model.to_model_proto()
    onnx.save(model_proto, model_path)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: gen_mul_model.py OUTPUT_DIR"
    output_dir = Path(sys.argv[1])
    gen_mul_model(output_dir / "mul.onnx")
