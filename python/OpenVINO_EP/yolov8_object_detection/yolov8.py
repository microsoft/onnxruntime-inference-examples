import os
from datetime import datetime
import requests
import shutil
import torch
from zipfile import ZipFile
import platform
import subprocess
import argparse

# Libraries for pre and post processsing
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.plotting import Annotator, colors
CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']

# import onnx_runtime related package
import onnxruntime as rt
import onnx
import numpy as np
import cv2
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType

        
def parse_arguments():
  parser = argparse.ArgumentParser(description='Object Detection using YOLOv8 using OpenVINO Execution Provider for ONNXRuntime')

  parser.add_argument('--device', default='OVEP', help="Device to perform inference on 'cpu (MLAS)' or on  OpenVINO-Execution provider.")

  parser.add_argument('--model', required=True, help='Path to model.')

  parser.add_argument('--image_url', default='https://storage.openvinotoolkit.org/data/test_data/images/cat.jpg', help='url for image to download for object detection \
                                                other options to download images are \
                                                https://storage.openvinotoolkit.org/data/test_data/images/dog.jpg\
                                                https://storage.openvinotoolkit.org/data/test_data/images/banana.jpg\
                                                https://storage.openvinotoolkit.org/data/test_data/images/apple.jpg\
                                                https://storage.openvinotoolkit.org/data/test_data/images/car.png')
  
  parser.add_argument('--niter', default=30, type=int, help='total number of iterations')

  parser.add_argument('--warmup_iter', default=10, type=int, help='total number of iterations')

  parser.add_argument("--quantize",help="Optional. Quantize yolov8 and run inference.",action='store_true')

  parser.add_argument("--show_image",help="Optional. Show image with object detection.",action='store_true')

  args = parser.parse_args()
  return args


# Process arguments
args = parse_arguments()
no_of_iterations = args.niter
warmup_iter = args.warmup_iter
device = args.device
original_model_path = args.model

if warmup_iter >= no_of_iterations:
    sys.exit("Warmup iterations are more than no of iterations(niter)!!")

# Parameters for pre-processing
imgsz = (640,640) # default value for this usecase. 
stride = 32 # default value for this usecase( differs based on the model selected )

# Parameters for post-processing
conf = 0.25
iou = 0.45
max_det = 300
classes = None
agnostic = False
labels = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 
                12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 
                48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
                60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
                72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

path = os.getcwd()


def initialize(quantize=False, device='OVEP'):
    "Initialize the model also getting model output and input names"

    initialized = True
    model_dir = os.getcwd()
    ov_model = None; mlas_model = None

    so = rt.SessionOptions()
    if quantize == True:
        print("Inferencing through OVEP")
        ov_model = rt.InferenceSession(quantized_model_path, so,
                                    providers=['OpenVINOExecutionProvider'],
                                    provider_options=[{'device_type' : 'CPU_FP32'}])
    else:
        ov_model = rt.InferenceSession(original_model_path, so,
                                    providers=['OpenVINOExecutionProvider'],
                                    provider_options=[{'device_type' : 'CPU_FP32'}])
    if quantize == True:
        mlas_model = rt.InferenceSession(quantized_model_path, so, providers=['CPUExecutionProvider'])
    else:
        mlas_model = rt.InferenceSession(original_model_path, so, providers=['CPUExecutionProvider'])

    if device == 'OVEP':
      input_names = ov_model.get_inputs()[0].name
      outputs = ov_model.get_outputs()
    else:
      input_names = mlas_model.get_inputs()[0].name
      outputs = mlas_model.get_outputs()      
    output_names = list(map(lambda output:output.name, outputs))
    return input_names, output_names, mlas_model, ov_model


print("device : ", device)
input_names, output_names, mlas_model, ov_model = initialize(device=device)

def preprocess(image_url):
    ## Set up the image URL and filename
    path = os.getcwd()
    image_path=os.path.join(path, image_url.split("/")[-1])

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(image_path,'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully downloaded: ',path)
    else:
        print('Image couldn\'t be retreived')
        return
    
    image_abs_path = os.path.abspath(image_path)
    if os.path.isfile(image_abs_path) and image_abs_path.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:

        # Load Image
        img0 = cv2.imread(image_abs_path)

        # Padded resize
        #Letterbox: Resize image and padding for detection, instance segmentation, pose
        img = LetterBox(imgsz, stride=stride)(image=img0.copy())

        # Convert
        img =  img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = img.astype(np.float32)  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        return img0, img
    else:
        print("Invalid image format.")
        return

def inference(input_names, output_names, device, mlas_model, ovep_model, model_input):
    inf_lst = []
    if device == 'CPUEP':
        print("Performing ONNX Runtime Inference with default CPU EP.")
        for i in range(no_of_iterations):
            start_time = datetime.now()
            prediction = mlas_model.run(output_names, {input_names: model_input})
            end_time = datetime.now()
            if i > warmup_iter:
                inf_lst.append((end_time - start_time).total_seconds())
            # print((end_time - start_time).total_seconds())
    elif device == 'OVEP':
        print("Performing ONNX Runtime Inference with OpenVINO EP.")
        for i in range(no_of_iterations):
            start_time = datetime.now()
            prediction = ovep_model.run(output_names, {input_names: model_input})
            end_time = datetime.now()
            if i > warmup_iter:
                inf_lst.append((end_time - start_time).total_seconds())
            # print((end_time - start_time).total_seconds())
    else:
        print("Invalid Device Option. Supported device options are 'cpu', 'CPU_FP32'.")
        return None
    
    average_inference_time = np.average(inf_lst)
    print(f'Average inference time is for {i+1 - args.warmup_iter} iterations is {average_inference_time}')
    return prediction, (end_time - start_time).total_seconds()

def postprocess( img0, img, inference_output):
    if inference_output is not None:
        prediction = inference_output[0]
        inference_time = inference_output[1]
        
        prediction = [torch.from_numpy(pred) for pred in prediction]
        preds = ops.non_max_suppression(prediction,
                                                0.25,
                                                0.45,
                                                agnostic=agnostic,
                                                max_det=max_det,
                                                classes=classes)
        log_string = ''
        results = []
        for _, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()
            results.append(Results(img0, path, labels, boxes=pred))

        det = results[0].boxes
        
        if len(det) == 0:
            return log_string+'No detection found.'
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {labels[int(c)]}{'s' * (n > 1)}, "

        raw_output = ''
        annotator = Annotator(img0, pil=False)
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            c = int(cls)  # integer class
            name = f'id:{int(d.id.item())} {labels[c]}' if d.id is not None else labels[c]
            label = f'{name} {conf:.2f}'
            box = d.xyxy.squeeze().tolist()
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            raw_output+=f"name: {name}, confidence: {conf:.2f}, start_point: {p1}, end_point:{p2}\n"
            annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            # annotator.box_label(box, label, color=colors(c, True))

        result_img = annotator.result()
        if args.show_image:
            cv2.imshow('image', org_input)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return [f"inference_time: {inference_time}s\nInference_summary: {log_string}\nraw_output:\n{raw_output}"]
    return None

org_input, model_input = preprocess(args.image_url)

input_names, output_names, mlas_model, ov_model = initialize()
# inference_output = inference(input_names, output_names, 'CPU_FP32', mlas_model, ov_model, model_input)
inference_output = inference(input_names, output_names, args.device, mlas_model, ov_model, model_input)
pred, time_required = inference_output

result = postprocess(org_input, model_input, inference_output)

#yolov8 dynamic quantization
if args.quantize:
    print("Quantizing yolov8 model.")
    model_fp32 = original_model_path
    model_quant = os.path.join(os.getcwd(), 'yolov8m_quantized.onnx')
    quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
    print(f'Quantized yolov8 model at {model_quant}')

