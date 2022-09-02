# FP32/INT8 YOLOv7 Sample

## Virtual Environement
Create a virtual environment either using a the `conda` virtual environment or `python venv`.

### Conda
```bash
$ conda create -n yolov7 python=3.8
$ conda activate yolov7
```
### Python venv
```bash
$ python3 -m venv yolov7
$ source yolov7/bin/activate
```

## Install required packages
After creating and activating virtual environment, install the required packages:

1. Jupyter Notebook
```
$ pip install jupyter
```
2. NNCF Experimental
```
$ git clone https://github.com/openvinotoolkit/nncf.git
$ cd nncf && python setup.py install --onnx
$ pip install -qr onnxruntime-openvino==1.12.0
$ rm -fR nncf 
```
## YoloV7

The YoloV7 repository, has a `requirements.txt` file which will install all the necessary packages.

**Please follow the instructions mentioned within the notebook. You might have to restart the notebook once the packages are installed.**

## Dataset
The dataset used in this sample is the coco-validation2017 dataset. It will be downloaded automatically in the desired location while running the notebook.

## Outputs

- `yolov7-tiny.onnx` 
- `yolov7-tiny-quantized.onnx`
- `cat-yolov7-detected.jpg`

## References
### [ONNX Rruntime](https://onnxruntime.ai/docs/install/)
### [NNCF](https://github.com/openvinotoolkit/nncf/tree/develop)