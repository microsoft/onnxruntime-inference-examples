# Object detection with yolov8 in Python using OpenVINO™ Execution Provider:

1. The Object detection sample uses a yolov8 Deep Learning ONNX Model from ultralytics.

2. The sample involves presenting a image to ONNX Runtime (RT), which uses the OpenVINO™ Execution Provider to run inference on various Intel hardware devices and perform object detection to detect up to 80 different objects like birds, bench, dogs, person and much more.

## Requirements
For all the python package dependencies requirements, check 'requirements.txt' file in the sample directory. You may also install these dependencies with in a virtual environment:
```bash
pip3 install -r requirements.txt
```

# How to build
## Prerequisites
1. Download and export yolov8 model from ultralytics
    Download pytorch model: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt

    Convert pytorch model to onnx: yolo mode=export model=yolov8n.pt format=onnx dynamic=True 

## Install ONNX Runtime for OpenVINO™ Execution Provider
Please install the onnxruntime-openvino python package from [here](https://pypi.org/project/onnxruntime-openvino). The package for Linux contains prebuilt OpenVINO Libs with ABI 0.
```
pip3 install onnxruntime-openvino openvino
```

## Reference Documentation
[Documentation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)



### How to run the sample
```bash
python3 yolov8.py.py --h
```
## Running the ONNXRuntime OpenVINO™ Execution Provider sample
```bash
python3 yolov8.py --model <path_to_the_yolov8_onnx_model> --device "OVEP" 
```

## References:

[Download OpenVINO™ Eexecution Provider Latest pip wheels from here](https://pypi.org/project/onnxruntime-openvino/)

[Python Pip Wheel Packages](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-for-onnx-runtime.html)

[Get started with ORT for Python](https://onnxruntime.ai/docs/get-started/with-python.html)