from pathlib import Path
from onnx import hub
import onnx

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_path = script_dir / "model.onnx"
    model_name = "MobileNet v2-1.0-fp32"
    
    model = hub.load(model_name)
    onnx.save(model, output_path)
