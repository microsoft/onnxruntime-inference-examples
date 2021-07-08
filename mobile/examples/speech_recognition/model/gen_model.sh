#!/bin/bash

set -e

OUTPUT_DIR=${1:?"Please specify an output directory."}

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

python3 ${DIR}/wav2vec2_gen.py
python3 -m onnxruntime.tools.convert_onnx_models_to_ort .
