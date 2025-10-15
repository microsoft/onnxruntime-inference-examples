'''
Copyright (C) 2021-2022, Intel Corporation
SPDX-License-Identifier: Apache-2.0
Major Portions of this code are copyright of their respective authors and released under the Apache License Version 2.0:
- onnx, Copyright 2021-2022. For licensing see https://github.com/onnx/models/blob/master/LICENSE
'''

import cv2
import numpy as np
from onnx import numpy_helper
import onnx
import onnxruntime as rt
import os
from PIL import Image
from scipy import special
import colorsys
import random
import argparse
import sys
import time
import platform
import json

if platform.system() == "Windows":
    import onnxruntime.tools.add_openvino_win_libs as utils
    utils.add_openvino_libs_to_path()

def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes

def postprocess_bbbox(pred_bbox):
    '''define anchor boxes'''
    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

        xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(float)

        pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)
    return pred_bbox

def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    '''remove boundary boxs with a low detection probability'''
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes that are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def bboxes_iou(boxes1, boxes2):
    '''calculate the Intersection Over Union value'''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def draw_bbox(image, bboxes, classes=read_class_names("coco.names"), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image

def get_anchors(anchors_path, tiny=False):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)

#Specify the path to anchors file on your machine
ANCHORS = "./yolov4_anchors.txt"    
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]
ANCHORS = get_anchors(ANCHORS)
STRIDES = np.array(STRIDES)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Object Detection using YOLOv4 in OPENCV using OpenVINO Execution Provider for ONNXRuntime')
    parser.add_argument('--device', default='CPU', help="Device to perform inference on 'cpu (MLAS)' or on devices supported by OpenVINO-EP [CPU, GPU, NPU].")
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    parser.add_argument('--model', help='Path to model.')
    args = parser.parse_args()
    return args
    
def check_model_extension(fp):
  # Split the extension from the path and normalise it to lowercase.
  ext = os.path.splitext(fp)[-1].lower()

  # Now we can simply use != to check for inequality, no need for wildcards.
  if(ext != ".onnx"):
    raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")
  
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the onnx model file is Invalid")

def main(): 

    # Process arguments
    args = parse_arguments()

    # Validate model file path
    check_model_extension(args.model)

    # Process inputs
    win_name = 'Object detection using ONNXRuntime OpenVINO Execution Provider using YoloV4 model'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    output_file = "yolo_out_py.avi"
    if (args.image):
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)
        output_file = args.image[:-4]+'_yolo_out_py.jpg'
    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4]+'_yolo_out_py.avi'
    else:
        # Webcam input
        cap = cv2.VideoCapture(0)

    # Get the video writer initialized to save the output video
    if (not args.image):
        vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # Check the device information and create a session
    device = args.device
    so = rt.SessionOptions()
    so.log_severity_level = 3
    if(args.device == 'cpu'):
        print("Device type selected is 'cpu' which is the default CPU Execution Provider (MLAS)")
        #Specify the path to the ONNX model on your machine and register the CPU EP
        sess = rt.InferenceSession(args.model, so, providers=['CPUExecutionProvider'])
    else:
        #Specify the path to the ONNX model on your machine and register the OpenVINO EP
        
        config_dict = {
            "CPU": {
                "INFERENCE_NUM_THREADS": "5",
                "CACHE_DIR": "C:\\"
            }
        }

        config_json = json.dumps(config_dict)

        sess = rt.InferenceSession(args.model, so, providers=['OpenVINOExecutionProvider'], provider_options=[{'device_type' : device, 'load_config' : config_json}])
        print("Device type selected is: " + device + " using the OpenVINO Execution Provider")
        '''
        other 'device_type' options are: (Any hardware target can be assigned if you have the access to it)
        'CPU', 'GPU', 'NPU'
        '''

    input_name = sess.get_inputs()[0].name
    
    while cv2.waitKey(1) < 0:
        # get frame from the video
        has_frame, frame = cap.read()
        # Stop the program if reached end of video
        if not has_frame:
            print("Done processing !!!")
            print("Output file is stored as ", output_file)
            has_frame=False
            cv2.waitKey(3000)
            # Release device
            cap.release()
            break
            
        input_size = 416
        original_image = frame
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
    
        outputs = sess.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))

        start = time.time()
        detections = sess.run(output_names, {input_name: image_data})
        end = time.time()
        inference_time = end - start

        pred_bbox = postprocess_bbbox(detections)
        bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
        bboxes = nms(bboxes, 0.213, method='nms')
        image = draw_bbox(original_image, bboxes)

        cv2.putText(image,device,(10,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        cv2.putText(image,'FPS: {}'.format(1.0/inference_time),(10,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

        # Write the frame with the detection boxes
        if (args.image):
            cv2.imwrite(output_file, image.astype(np.uint8))
        else:
            vid_writer.write(image.astype(np.uint8))
 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow(win_name, image)

if __name__ == "__main__":
  main()
