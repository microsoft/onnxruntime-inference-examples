#!/bin/bash
set -x

while getopts o:w:p: parameter
do case "${parameter}"
in
o) ONNXRUNTIME_ROOTDIR=${OPTARG};;
p) ORT_PACKAGE=${OPTARG};;
w) WORKSPACE=${OPTARG};;
esac
done

ONNX_MODEL_URL="https://media.githubusercontent.com/media/onnx/models/main/archive/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
ONNX_MODEL="squeezenet.onnx"

CUR_PWD=$(pwd)
cd "$(dirname ${ORT_PACKAGE})"

tar zxvf ${ORT_PACKAGE}
ORT_LIB="${ORT_PACKAGE%.*}/lib"
ls -al ${ORT_LIB}

export LD_LIBRARY_PATH=${ORT_LIB}
export LIBRARY_PATH=${ORT_LIB}

cd ${WORKSPACE}
mkdir -p build
cd build

cmake .. -DONNXRUNTIME_ROOTDIR=${ONNXRUNTIME_ROOTDIR}
make -j4

if [ -f /data/models/opset8/test_squeezenet/model.onnx ]; then
   cp /data/models/opset8/test_squeezenet/model.onnx "${ONNX_MODEL}"
else
   curl ${ONNX_MODEL_URL} --output ${ONNX_MODEL}
fi
./capi_test

if [ $? -ne 0 ]
then
    echo "capi test application failed."
    cd ${CUR_PWD}
    exit 1
fi

cd ${CUR_PWD}
