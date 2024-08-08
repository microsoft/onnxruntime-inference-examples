#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ============================================================
set -e -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOWNLOAD_DIR="${SCRIPT_DIR}/ModelsAndData"

mkdir -p "${DOWNLOAD_DIR}/tflite_model"

# Download source tflite model file
curl -L -o "${DOWNLOAD_DIR}/tflite_model/model.tar.gz" \
  https://www.kaggle.com/api/v1/models/tensorflow/ssd-mobilenet-v1/tfLite/metadata/1/download

# Extract model
tar -xzf "${DOWNLOAD_DIR}/tflite_model/model.tar.gz" -C "${DOWNLOAD_DIR}/tflite_model"
# Extract the labelmap.txt file within the tflite file
unzip "${DOWNLOAD_DIR}/tflite_model/1.tflite" labelmap.txt -d "${DOWNLOAD_DIR}"

# Convert tflite model to onnx model
python -m tf2onnx.convert --tflite "${DOWNLOAD_DIR}/tflite_model/1.tflite" --opset 13 --output "${DOWNLOAD_DIR}/ssd_mobilenet_v1.onnx"

# Convert onnx model to ort format
# This step generates a .ort format ssd mobilenet model in /ModelsAndData.
# The .ort format model is the one gets executed in this sample application
python -m onnxruntime.tools.convert_onnx_models_to_ort "${DOWNLOAD_DIR}/ssd_mobilenet_v1.onnx"

# Remove the tflite model
rm -r "${DOWNLOAD_DIR}/tflite_model"
