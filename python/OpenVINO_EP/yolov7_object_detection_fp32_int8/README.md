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
If system python3 version is < 3.8 it's recommended to either upgrade to python3.8 or use create virtual environment python3.8 using conda.
```
$ conda create -n yolov7 python=3.8
$ conda activate yolov7
```

## Install required packages
After creating and activating virtual environment, install the required packages:

### NNCF Experimental
```
$ git clone https://github.com/openvinotoolkit/nncf.git
$ cd nncf && python setup.py install --onnx
$ pip install --no-cache-dir onnxconverter_common onnxruntime-openvino==1.11.0
$ cd .. && rm -fR nncf 
```
### YoloV7 requirements

```
$ cd notebooks && git clone https://github.com/WongKinYiu/yolov7
$ cp -R yolov7/* .
$ pip install --no-cache-dir -r requirements.txt
$ rm -fR yolov7
```

# Steps to perform NNCF Quantization for YoloV7

## Export Model to onnx
The models are to be exported using the export script present in the yolov7 repo. Download desired yolov7 model from [here](https://github.com/WongKinYiu/yolov7/releases).
```
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```
Export Model using the below command. (As mentioned in the official public [repo](https://github.com/WongKinYiu/yolov7#export).)
**Note: Adjust the values of `iou-thres` and `conf-thres` while exporting to get better bounding boxes.**
```
$ python export.py --weights yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

## Download Data
```
$ cd .. && wget -O tmp.zip 'https://ultralytics.com/assets/coco2017val.zip'
$ unzip -q tmp.zip -d datasets && rm tmp.zip
```


## Quantization

The quantization script is specific to yolov7 model. All the dependencies (helper functions & loader objects, taken from the public [repo](https://github.com/WongKinYiu/yolov7/)) are present within the `ptq_yolov7.py` script. 

```
python ptq_yolov7.py --data datasets/coco/images/val2017 \
                     --input-model notebooks/yolov7-tiny.onnx \
                     --sample-size 100
```

## Run Inference

### Run the sample on default CPU Execution Provider (MLAS)
```
$ python3 yolov7.py --device cpu --video classroom.mp4 --model <path to model>
```
### Run the sample with video as Input
```
$ python3 yolov7.py --device CPU_FP32 --video classroom.mp4 --model <path to model>
```
### Run the sample with Image as Input
```
$ python3 yolov7.py --device CPU_FP32 --image cat.jpg --model <path to model>
```

### Run inference using notebook
Install jupyter notebook in the corresponding env and run the notebook present in the notebooks directory.