# GroundingDINO with ONNX Runtime

This example runs GroundingDINO zero-shot object detection with ONNX Runtime.
It uses Hugging Face Transformers for preprocessing and post-processing, and
ONNX Runtime for model execution.

## Model

The default model is
[`onnx-community/grounding-dino-tiny-ONNX`](https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX).
The example downloads only the requested ONNX file from the Hugging Face Hub.

The default ONNX file is `onnx/model_quantized.onnx`. It is smaller than the
full precision model and is suitable for CPU execution.

## Setup

Install the dependencies:

```bash
pip install onnxruntime
pip install -r requirements.txt
```

For GPU execution, install the ONNX Runtime GPU package instead of the CPU
package:

```bash
pip install onnxruntime-gpu
pip install -r requirements.txt
```

## Run

```bash
python infer_grounding_dino_onnxruntime.py \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --text "a cat. a remote control." \
  --output output.jpg
```

Run with a specific execution provider:

```bash
python infer_grounding_dino_onnxruntime.py \
  --provider CPUExecutionProvider \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --text "a cat. a remote control."
```

Run with a local ONNX file:

```bash
python infer_grounding_dino_onnxruntime.py \
  --model-path path/to/model.onnx \
  --model-repo onnx-community/grounding-dino-tiny-ONNX \
  --image path/to/image.jpg \
  --text "a cat. a remote control."
```

## Output

The script prints:

- ONNX Runtime execution provider.
- Model input and output names.
- Input and output types and shapes.
- Detected text labels, confidence scores, and bounding boxes.

It also writes an annotated image to `--output`.
