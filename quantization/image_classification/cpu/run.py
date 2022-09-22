import argparse
import numpy as np
import onnxruntime
import time
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

import resnet50_data_reader


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 224, 224), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset", default="./test_images", help="calibration data set"
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    dr = resnet50_data_reader.ResNet50DataReader(
        calibration_dataset_path, input_model_path
    )

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        optimize_model=False,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()
