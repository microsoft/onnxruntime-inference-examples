# Object detection with tinyYOLOv2 in Python using OpenVINO™ Execution Provider:

1. The Object detection sample uses a tinyYOLOv2 Deep Learning ONNX Model from the ONNX Model Zoo.

2. The sample involves presenting a frame-by-frame video to ONNX Runtime (RT), which uses the OpenVINO™ Execution Provider to run inference on various Intel hardware devices as mentioned before and perform object detection to detect up to 20 different objects like birds, buses, cars, people and much more.

The source code for this sample is available [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/tiny_yolo_v2_object_detection).

# How to build

## Prerequisites
1. Download the latest tinyYOLOv2 model from the ONNX Model Zoo.
   This model was adapted from [ONNX Model Zoo](https://github.com/onnx/models).Download the latest version of the [tinyYOLOv2](https://github.com/onnx/models/tree/master/validated/vision/object_detection_segmentation/tiny-yolov2) model from here.

## Install ONNX Runtime for OpenVINO™ Execution Provider
Please install the onnxruntime-openvino python package from [here](https://pypi.org/project/onnxruntime-openvino). The package for Linux contains prebuilt OpenVINO Libs with ABI 0.
```
pip3 install onnxruntime-openvino openvino
```

## Optional Build steps for ONNX Runtime
[build instructions](https://onnxruntime.ai/docs/build/eps.html#openvino)

Note: Make sure to install [OpenVINO™ Runtime using an installer](https://docs.openvino.ai/latest/openvino_docs_install_guides_install_runtime.html) to build the python wheels from source.

## Reference Documentation
[Documentation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

## Requirements
* ONNX Runtime 1.6+
* numpy version 1.21.6+
* opencv 4.5.5+
* python 3+
* Use any sample video with objects as test input to this sample [Download Sample videos](https://github.com/intel-iot-devkit/sample-videos)
* Download the tinyYOLOv2 model from the [ONNX Model Zoo](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2) 

Note: For all the python package dependencies requirements, check 'requirements.txt' file in the sample directory. You may also install these dependencies with:
```bash
pip3 install -r requirements.txt
```

### How to run the sample
```bash
python3 tiny_yolov2_obj_detection_sample.py --h
```
## Running the ONNXRuntime OpenVINO™ Execution Provider sample
```bash
python3 tiny_yolov2_obj_detection_sample.py --video face-demographics-walking-and-pause.mp4 --model tinyyolov2.onnx --device CPU
```

## To stop the sample from running
```bash
Just press the letter 'q' or Ctrl+C if on Windows
```

## References:

[Download OpenVINO™ Eexecution Provider Latest pip wheels from here](https://pypi.org/project/onnxruntime-openvino/)

[OpenVINO™ Execution Provider](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/faster-inferencing-with-one-line-of-code.html)

[Docker Containers](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-docker-container.html)

[Python Pip Wheel Packages](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-for-onnx-runtime.html)

[Get started with ORT for Python](https://onnxruntime.ai/docs/get-started/with-python.html)

