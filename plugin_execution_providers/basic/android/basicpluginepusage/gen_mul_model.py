from pathlib import Path
import onnx
from onnxscript import script, FLOAT, opset15 as op

@script(default_opset=op)
def model(x: FLOAT[2, 3], y: FLOAT[2, 3]) -> FLOAT[2, 3]:
    return x * y

model_proto = model.to_model_proto()
script_dir = Path(__file__).parent
onnx.save(model_proto, f"{script_dir}/src/main/res/raw/mul.onnx")
