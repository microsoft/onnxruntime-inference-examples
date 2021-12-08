# Object detection with YOLOv4 in Python using OpenVINO Execution Provider:

1. The Object detection sample uses a YOLOv4 Deep Learning ONNX Model from the ONNX Model Zoo.

2. The sample involves presenting a frame-by-frame video to ONNX Runtime (RT), which uses the OpenVINO Execution Provider to run inference on various Intel hardware devices as mentioned before and perform object detection to detect up to 80 different objects like person, bicycle, car, motorbike and much more from the coco dataset.

3. The sample can work with multiple types of inputs
* Image
* Video
* Live camera video input

The source code for this sample is available [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/yolov4_object_detection).

4. Once the inferencing is done on the sample, the recording of the same also gets downloaded on the disk.

# How to build

## Prerequisites
1. [The Intel<sup>®</sup> Distribution of OpenVINO toolkit](https://docs.openvinotoolkit.org/latest/index.html)
2. Download the latest tinyYOLOv2 model from the ONNX Model Zoo.
   This model was adapted from [ONNX Model Zoo](https://github.com/onnx/models).Download the latest version of the [YOLOv4](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4) model from here.

## Install ONNX Runtime for OpenVINO Execution Provider

## Build steps
[build instructions](https://onnxruntime.ai/docs/build/eps.html#openvino)

## Reference Documentation
[Documentation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

## Requirements
* ONNX Runtime 1.6+
* numpy version 1.19.5+
* opencv 4.5.1+
* python 3+
* use any sample video with objects as test input to this sample
* Download the tinyYOLOv2 model from the ONNX Model Zoo
[Download Sample videos](https://github.com/intel-iot-devkit/sample-videos)

Note: For all the python package dependencies requirements, check 'requirements.txt' file in the sample directory. You may also install these dependencies with:
```bash
pip3 install -r requirements.txt
```

## Running the ONNXRuntime OpenVINO Execution Provider sample

### How to run the sample
```bash
python3 python3 yolov4.py --h
```

### Run the sample on OpenVINO EP
```bash
python3 yolov4.py --device CPU_FP32 --video bottle-detection.mp4
```
Note: You can pick different device options to run on OpenVINO EP like GPU_FP32, GPU_FP16 and MYRIAD_FP16.

### Run the sample on default CPU EP (MLAS)
```bash
python3 yolov4.py --device cpu --video bottle-detection.mp4
```

### Run the sample with video as Input
```bash
python3 yolov4.py --device CPU_FP32 --video bottle-detection.mp4
```

### Run the sample with Image as Input
```bash
python3 yolov4.py --device CPU_FP32 --image cat.jpg
```

### Run the sample with Live Input stream Like webcam
```bash
python3 yolov4.py --device CPU_FP32
```

## To stop the sample from running
```bash
Just press the letter 'q' or Ctrl+C if on Windows
```

## References:

[Download OpenVINO EP Latest pip wheels from here](https://github.com/intel/onnxruntime/releases/)

[OpenVINO Execution Provider](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/faster-inferencing-with-one-line-of-code.html)

[Docker Containers](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-docker-container.html)

[Python Pip Wheel Packages](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-for-onnx-runtime.html)

[Get started with ORT for Python](https://onnxruntime.ai/docs/get-started/with-python.html)

