import argparse
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

import gpt2_input_reader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model",
        default="gpt2_medium_fp32_preprocessed.onnx",
        help="Path to float 32 gpt-2 model.",
    )
    parser.add_argument(
        "--output_model", required=False, help="Path to quantized model",
        default="gpt2_medium_fp32_quant.onnx"
    )
    parser.add_argument(
        "--calibrate_dataset",
        default="./test_input",
        help="Specify the destination folder of input data sets.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    if not output_model_path:
        output_model_path = (
            input_model_path[: -len(".onnx")]
            if input_model_path.endswith(".onnx")
            else input_model_path
        )
        output_model_path += "_qdq.onnx"

    calibration_dataset_path = args.calibrate_dataset
    input_reader = gpt2_input_reader.Gpt2InputReader(calibration_dataset_path)
    quantize_static(
        input_model_path,
        output_model_path,
        input_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")


if __name__ == "__main__":
    main()
