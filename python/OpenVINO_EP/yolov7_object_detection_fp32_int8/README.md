# FP32/INT8 YOLOv7 Sample

## Virtual Environement
Create a virtual environment either using a the `conda` virtual environment or `python venv`.

### Conda
```
$ conda create -n yolov7 python=3.8
$ conda activate yolov7
```
### Python venv
```
$ python3 -m venv yolov7
$ source yolov7/bin/activate
```

## Install required packages
After creating and activating virtual environment, install the required packages:

1. Jupyter Notebook
```
$ pip install --no-cache-dir jupyter
```
2. NNCF Experimental
```
$ git clone https://github.com/openvinotoolkit/nncf.git
$ cd nncf && python setup.py install --onnx
$ pip install --no-cache-dir onnxruntime-openvino==1.11.0
$ pip install --no-cache-dir onnxconverter_common
$ cd .. && rm -fR nncf 
```
3. YoloV7 requirements

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