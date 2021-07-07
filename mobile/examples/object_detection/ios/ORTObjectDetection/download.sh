#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ============================================================

MODELS_URL="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

mkdir -p "${SCRIPT_DIR}/ModelsAndData"

DOWNLOAD_DIR="${SCRIPT_DIR}/ModelsAndData"

#Download source tflite model file
curl -L ${MODELS_URL} >${DOWNLOAD_DIR}/ssd_mobilenet_v1_1_metadata_1.tflite

#Unzip and get model metadata
curl -L ${MODELS_URL} >${DOWNLOAD_DIR}/ssd_mobilenet_v1_1_metadata_1.zip
unzip ${DOWNLOAD_DIR}/ssd_mobilenet_v1_1_metadata_1.zip -d ${DOWNLOAD_DIR}

#Convert tflite model to onnx model
python3 -m tf2onnx.convert --tflite ${DOWNLOAD_DIR}/ssd_mobilenet_v1_1_metadata_1.tflite --opset 13  --output ${DOWNLOAD_DIR}/ssd_mobilenet_v1.onnx

#Convert onnx model to ort format
python3 -m onnxruntime.tools.convert_onnx_models_to_ort ${DOWNLOAD_DIR}
