# Image classification using an augmented ONNX model

## Pre-requisites

* Install cmake
* Install gcc for your platform
* Install conda
* Create a conda environment
* Install onnxruntime-extension package from source: 

## Augment the MobileNetV2 model

Base model: https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx

Do a netron screen grab

### Preprocessing

The pre processing for the MobileNet model consists of the following steps:
- resize
- center
- normalize

### Postprocessing

The post processing of the MobileNet model consists of taking the scores for each of the classes and transforming them into the top 10 class ids and corresponding top 10 probabilities.

## Notes

/pnp/_onnx_ops.py:71: UserWarning: The maximum opset needed by this model is only 9.
  warnings.warn('The maximum opset needed by this model is only %d.' % op_version)