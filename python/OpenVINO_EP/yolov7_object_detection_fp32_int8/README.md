# Object detection with YOLOv7 in Python using OpenVINO™ Execution Provider
## Virtual Environement
Create a virtual environment either using a the `conda` virtual environment or `python venv`.

### Python venv
```
$ python3 -m venv yv7-pyenv
$ source yv7-pyenv/bin/activate
$ pip install --upgrade pip
```
### Conda
If system python3 version is < 3.8.13 it's recommended to either upgrade to python3.8 or use create virtual environment python3.8 using conda.
```
$ conda create -n yolov7 python=3.8
$ conda activate yolov7
```
## Install required packages
After creating and activating virtual environment, install the required packages:

### NNCF Experimental

```
$ git clone https://github.com/openvinotoolkit/nncf.git
$ cd nncf && python setup.py develop --onnx
$ pip install --no-cache-dir onnxconverter_common onnxruntime-openvino==1.11.0
$ cd .. && rm -fR nncf 
```

### YoloV7 requirements

```
$ cd notebooks
$ git clone https://github.com/WongKinYiu/yolov7
$ cp -R yolov7/* .
$ pip install --no-cache-dir -r requirements.txt
$ rm -fR yolov7
```

Launch the notebook from here.
## Dataset
The dataset used in this sample is the coco-validation2017 dataset. It will be downloaded automatically in the desired location while running the notebook.

## Outputs

- `yolov7-tiny.onnx` 
- `yolov7-tiny-quantized.onnx`
- `cat-yolov7-detected.jpg`

## References
### [ONNX Rruntime](https://onnxruntime.ai/docs/install/)
### [NNCF](https://github.com/openvinotoolkit/nncf/tree/develop)