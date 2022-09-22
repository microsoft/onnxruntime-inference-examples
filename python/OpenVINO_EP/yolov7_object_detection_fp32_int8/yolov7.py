import argparse
import logging
import os
import random
import sys
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
from PIL import Image

warnings.filterwarnings(action="ignore")


# Inference for ONNX model
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


random.seed(42)
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = {name: [random.randint(0, 255) for _ in range(3)]
          for i, name in enumerate(names)}


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


set_logging(0)  # run before defining LOGGER
LOGGER = logging.getLogger("yolov7")


def preProcess_image(image):
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255

    return im, ratio, dwdh


def create_session(model_path, device='CPU_FP32'):

    if device == 'CPU_FP32':
        providers = ['OpenVINOExecutionProvider']
    elif device == 'cpu':
        providers = ['CPUExecutionProvider']
    else:
        LOGGER.info(f'No provider passed, using default CPU EP ...')
        providers = ['CPUExecutionProvider']

    LOGGER.info(f'Use ORT providers: {providers}')
    sess = onnxruntime.InferenceSession(model_path,
                                        providers=providers,
                                        provider_options=[{'device_type': device}])

    outname = [i.name for i in sess.get_outputs()]
    inname = [i.name for i in sess.get_inputs()]

    return sess, outname, inname


def main(args):

    model_name = args.model.split('.')[0]

    if (args.image):
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)
        output_file = args.image[:-4]+f'_{model_name}_{args.device}.jpg'
    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4]+f'_{model_name}_{args.device}.avi'
    else:
        # Webcam input
        cap = cv2.VideoCapture(0)

    # Get the video writer initialized to save the output video
    if (not args.image):
        vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    sess, outputs, inputs = create_session(args.model, args.device)

    while cv2.waitKey(1) < 0:
        # get frame from the video
        has_frame, frame = cap.read()
        # Stop the program if reached end of video
        if not has_frame:
            print("Done processing !!!")
            print("Output file is stored as ", output_file)
            has_frame = False
            # cv2.waitKey(3000)
            # Release device
            cap.release()
            break

        original_image = frame
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # original_image_size = original_image.shape[:2]

        image_data, proces_ratio, dwdh = preProcess_image(
            original_image.copy())

        inp = {inputs[0]: image_data}
        start = time.time()
        pred = sess.run(outputs, inp)[0]
        end = time.time()
        inference_time = end - start

        original_image = [original_image]
        image = original_image[0]
        if pred.any():
            for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(pred):
                image = original_image[int(batch_id)]
                box = np.array([x0, y0, x1, y1])
                box -= np.array(dwdh*2)
                box /= proces_ratio
                box = box.round().astype(np.int32).tolist()
                cls_id = int(cls_id)

                score = round(float(score), 3)
                name = names[cls_id]
                color = colors[name]
                name += ' '+str(score)
                cv2.rectangle(image, box[:2], box[2:], color, 2)
                cv2.putText(image, name, (box[0], box[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, [225, 255, 255], thickness=1)
                cv2.putText(image, 'FPS: {:.8f}'.format(1.0 / inference_time),
                            (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 255, 255), 0)

        # Write the frame with the detection boxes
        if (args.image):
            cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print(f"saving image..{output_file}")
        else:
            vid_writer.write(image.astype(np.uint8))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if args.view_img:
            win_name = "test_yolov7"
            cv2.imshow(win_name, image)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Object Detection using YOLOv7 in OPENCV using OpenVINO Execution Provider for ONNXRuntime')
    parser.add_argument('--device', default='CPU_FP32',
                        help="Device to perform inference on 'cpu (MLAS)' or on devices supported by OpenVINO-EP [CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16].")
    parser.add_argument('--image', 
                        help='Path to image file.')
    parser.add_argument('--video', default="person-bicycle-car-detection.mp4", help='Path to video file.')
    parser.add_argument('--model', default="notebooks/yolov7-tiny.onnx",
                        help='Path to model.')

    parser.add_argument('--view-img', action="store_true",
                        help='Path to model.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
