#!/bin/bash

# helper script to download model files for CI build
# see mobile/examples/image_classification/android/README.md

set -e

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

RESOURCES_DIR="${DIR}/app/src/main/res/raw"

curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
  --output "${RESOURCES_DIR}/imagenet_classes.txt"

curl https://onnxruntimeexamplesdata.z13.web.core.windows.net/mobilenet_v2_ort_models.zip \
  --output "${RESOURCES_DIR}/models.zip" \
  --retry 3

unzip "${RESOURCES_DIR}/models.zip" -d "${RESOURCES_DIR}"

rm "${RESOURCES_DIR}/models.zip"
