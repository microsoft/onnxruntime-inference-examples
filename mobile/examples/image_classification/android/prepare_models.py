#!/usr/bin/env python3

import argparse
import onnx.hub
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description="Prepares model files used by the image_classification example.")
    default_format = "onnx"
    parser.add_argument(
        "--format",
        choices=["onnx", "ort"],
        default=default_format,
        help=f"Model format to generate. Default: {default_format}",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory.",
    )
    return parser.parse_args()


def download_file(url: str, output_path: pathlib.Path):
    with urllib.request.urlopen(url) as response:
        assert response.status == 200
        with open(output_path, mode="wb") as file:
            file.write(response.read())


def download_onnx_model(model_name: str, output_path: pathlib.Path):
    model = onnx.hub.load(model_name)
    onnx.save(model, str(output_path))


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # download classes file
    download_file(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        args.output_dir / "imagenet_classes.txt",
    )

    # prepare models in temporary directory, then copy them to output directory
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        for model_name, model_path_stem in [
            ("MobileNet v2-1.0-int8", "mobilenetv2_int8"),
            ("MobileNet v2-1.0-fp32", "mobilenetv2_fp32"),
        ]:
            model_path = temp_dir / f"{model_path_stem}.onnx"

            # download from ONNX model zoo
            download_onnx_model(model_name, model_path)

            # make batch_size dim fixed
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "onnxruntime.tools.make_dynamic_shape_fixed",
                    "--dim_param=batch_size",
                    "--dim_value=1",
                    str(model_path),
                    str(model_path),
                ],
                check=True,
            )

            # convert to ORT format if needed
            if args.format == "ort":
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "onnxruntime.tools.convert_onnx_models_to_ort",
                        "--optimization_style=Fixed",
                        str(model_path),
                    ],
                    check=True,
                )

                # update to converted model
                model_path = model_path.with_suffix(".ort")

            dest_model_path = args.output_dir / model_path.name
            shutil.copyfile(model_path, dest_model_path)


if __name__ == "__main__":
    main()
