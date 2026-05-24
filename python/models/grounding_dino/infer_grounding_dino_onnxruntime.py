# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoObjectDetectionOutput,
)


ORT_DTYPE_TO_NUMPY = {
    "tensor(bool)": np.bool_,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO ONNX inference with ONNX Runtime."
    )
    parser.add_argument(
        "--model-repo",
        default="onnx-community/grounding-dino-tiny-ONNX",
        help="Hugging Face model repository containing the processor files.",
    )
    parser.add_argument(
        "--onnx-file",
        default="onnx/model_quantized.onnx",
        help="ONNX filename inside --model-repo. Ignored when --model-path is set.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Local ONNX model path. If unset, the file is downloaded from --model-repo.",
    )
    parser.add_argument(
        "--image",
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="Image URL or local image path.",
    )
    parser.add_argument(
        "--text",
        default="a cat. a remote control.",
        help="Grounding text. Separate object names with periods.",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="Minimum object confidence score.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.3,
        help="Minimum token confidence score for text label extraction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.jpg"),
        help="Annotated output image path.",
    )
    return parser.parse_args()


def resolve_model_path(
    model_repo: str, onnx_file: str, model_path: Path | None
) -> Path:
    if model_path is not None:
        return model_path

    downloaded_path = hf_hub_download(repo_id=model_repo, filename=onnx_file)
    return Path(downloaded_path)


def load_image(source: str) -> Image.Image:
    parsed_url = urlparse(source)
    if parsed_url.scheme in {"http", "https"}:
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(source).convert("RGB")

    print(f"Loaded image: Type: PIL.Image.Image, Shape: {image.size[::-1] + (3,)}")
    return image


def create_session(model_path: Path, provider: str) -> ort.InferenceSession:
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise ValueError(
            f"Requested provider '{provider}' is not available. "
            f"Available providers: {available_providers}"
        )

    session = ort.InferenceSession(str(model_path), providers=[provider])
    print(f"ONNX model path: {model_path}")
    print(f"ONNX Runtime providers: {session.get_providers()}")
    return session


def cast_for_ort(name: str, value: np.ndarray, ort_type: str) -> np.ndarray:
    expected_dtype = ORT_DTYPE_TO_NUMPY.get(ort_type)
    if expected_dtype is None:
        raise TypeError(f"Unsupported ONNX Runtime input type for '{name}': {ort_type}")

    array = np.asarray(value)
    if array.dtype != expected_dtype:
        array = array.astype(expected_dtype)
    return array


def build_ort_inputs(
    session: ort.InferenceSession, encoded_inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    ort_inputs = {}
    for input_meta in session.get_inputs():
        if input_meta.name not in encoded_inputs:
            raise KeyError(f"Processor output does not contain '{input_meta.name}'")

        value = cast_for_ort(
            input_meta.name, encoded_inputs[input_meta.name], input_meta.type
        )
        ort_inputs[input_meta.name] = value
        print(
            f"Prepared input {input_meta.name}: "
            f"Type: np.ndarray[{value.dtype}], Shape: {value.shape}"
        )

    return ort_inputs


def run_ort(
    session: ort.InferenceSession, ort_inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    output_names = [output_meta.name for output_meta in session.get_outputs()]
    print(f"Model output names: {output_names}")

    raw_outputs = session.run(output_names, ort_inputs)
    output_map = dict(zip(output_names, raw_outputs, strict=True))

    for name, value in output_map.items():
        print(
            f"Raw output {name}: Type: np.ndarray[{value.dtype}], Shape: {value.shape}"
        )

    return output_map


def post_process(
    processor,
    output_map: dict[str, np.ndarray],
    input_ids: np.ndarray,
    image: Image.Image,
    box_threshold: float,
    text_threshold: float,
):
    required_outputs = {"logits", "pred_boxes"}
    missing_outputs = required_outputs.difference(output_map)
    if missing_outputs:
        raise KeyError(f"Missing required model outputs: {sorted(missing_outputs)}")

    outputs = GroundingDinoObjectDetectionOutput(
        logits=torch.from_numpy(output_map["logits"]),
        pred_boxes=torch.from_numpy(output_map["pred_boxes"]),
        input_ids=torch.from_numpy(input_ids),
    )
    target_sizes = [image.size[::-1]]
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=outputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )
    result = results[0]
    print(
        f"Postprocessed boxes: Type: torch.Tensor, Shape: {tuple(result['boxes'].shape)}"
    )
    print(
        f"Postprocessed scores: Type: torch.Tensor, Shape: {tuple(result['scores'].shape)}"
    )
    return result


def annotate_image(image: Image.Image, result, output_path: Path) -> None:
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for box, score, label in zip(
        result["boxes"], result["scores"], result["text_labels"], strict=True
    ):
        x0, y0, x1, y1 = [float(value) for value in box.tolist()]
        caption = f"{label}: {float(score):.3f}"
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
        text_box = draw.textbbox((x0, y0), caption)
        draw.rectangle(text_box, fill="red")
        draw.text((x0, y0), caption, fill="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_image.save(output_path)
    print(f"Saved annotated image: {output_path}")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_repo, args.onnx_file, args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_repo)
    image = load_image(args.image)
    session = create_session(model_path, args.provider)

    encoded_inputs = processor(images=image, text=args.text, return_tensors="np")
    ort_inputs = build_ort_inputs(session, encoded_inputs)
    output_map = run_ort(session, ort_inputs)
    result = post_process(
        processor,
        output_map,
        ort_inputs["input_ids"],
        image,
        args.box_threshold,
        args.text_threshold,
    )

    for index, (box, score, label) in enumerate(
        zip(result["boxes"], result["scores"], result["text_labels"], strict=True),
        start=1,
    ):
        rounded_box = [round(float(value), 2) for value in box.tolist()]
        print(
            f"Detection {index}: Label: {label}, "
            f"Score: {float(score):.3f}, Box: {rounded_box}"
        )

    annotate_image(image, result, args.output)


if __name__ == "__main__":
    main()
