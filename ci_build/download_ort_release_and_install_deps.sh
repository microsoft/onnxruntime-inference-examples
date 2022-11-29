#!/bin/bash
set -e -x

ORT_VER="1.13.1"

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