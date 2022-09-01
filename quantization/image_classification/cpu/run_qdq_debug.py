import argparse
import onnx
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_activation_error, compute_weight_error,
    create_activation_matching, create_weight_matching,
    modify_model_output_intermediate_tensors)

import resnet50_data_reader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--float_model", required=True, help="Path to original floating point model"
    )
    parser.add_argument("--qdq_model", required=True, help="Path to qdq model")
    parser.add_argument(
        "--calibrate_dataset", default="./test_images", help="calibration data set"
    )
    args = parser.parse_args()
    return args


def _generate_aug_model_path(model_path: str) -> str:
    aug_model_path = (
        model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
    )
    return aug_model_path + ".save_tensors.onnx"


def main():
    # Process input parameters and setup model input data reader
    args = get_args()
    float_model_path = args.float_model
    qdq_model_path = args.qdq_model
    calibration_dataset_path = args.calibrate_dataset

    print("------------------------------------------------\n")
    print("Comparing weights of float model vs qdq model.....")

    matched_weights = create_weight_matching(float_model_path, qdq_model_path)
    weights_error = compute_weight_error(matched_weights)
    for weight_name, err in weights_error.items():
        print(f"Cross model error of '{weight_name}': {err}\n")

    print("------------------------------------------------\n")
    print("Augmenting models to save intermediate activations......")

    aug_float_model = modify_model_output_intermediate_tensors(float_model_path)
    aug_float_model_path = _generate_aug_model_path(float_model_path)
    onnx.save(
        aug_float_model,
        aug_float_model_path,
        save_as_external_data=False,
    )
    del aug_float_model

    aug_qdq_model = modify_model_output_intermediate_tensors(qdq_model_path)
    aug_qdq_model_path = _generate_aug_model_path(qdq_model_path)
    onnx.save(
        aug_qdq_model,
        aug_qdq_model_path,
        save_as_external_data=False,
    )
    del aug_qdq_model

    print("------------------------------------------------\n")
    print("Running the augmented floating point model to collect activations......")
    input_data_reader = resnet50_data_reader.ResNet50DataReader(
        calibration_dataset_path, float_model_path
    )
    float_activations = collect_activations(aug_float_model_path, input_data_reader)

    print("------------------------------------------------\n")
    print("Running the augmented qdq model to collect activations......")
    input_data_reader.rewind()
    qdq_activations = collect_activations(aug_qdq_model_path, input_data_reader)

    print("------------------------------------------------\n")
    print("Comparing activations of float model vs qdq model......")

    act_matching = create_activation_matching(qdq_activations, float_activations)
    act_error = compute_activation_error(act_matching)
    for act_name, err in act_error.items():
        print(f"Cross model error of '{act_name}': {err['xmodel_err']} \n")
        print(f"QDQ error of '{act_name}': {err['qdq_err']} \n")


if __name__ == "__main__":
    main()
