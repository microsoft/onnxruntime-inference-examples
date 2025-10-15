'''
Copyright (C) 2021-2022, Intel Corporation
SPDX-License-Identifier: Apache-2.0
'''

import numpy as np
import onnxruntime as rt
import cv2
import time
import os
import argparse
import platform

if platform.system() == "Windows":
    import onnxruntime.tools.add_openvino_win_libs as utils
    utils.add_openvino_libs_to_path()

# color look up table for different classes for object detection sample
clut = [(0,0,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,128),
        (128,255,0),(128,128,0),(0,128,255),(128,0,128),
        (255,0,128),(128,0,255),(255,128,128),(128,255,128),(255,255,0),
        (255,128,128),(128,128,255),(255,128,128),(128,255,128),(128,255,128)]

# 20 labels that the tiny-yolov2 model can do the object_detection on
label = ["aeroplane","bicycle","bird","boat","bottle",
         "bus","car","cat","chair","cow","diningtable",
         "dog","horse","motorbike","person","pottedplant",
          "sheep","sofa","train","tvmonitor"]

def parse_arguments():
  parser = argparse.ArgumentParser(description='Object Detection using YOLOv2 in OPENCV using OpenVINO Execution Provider for ONNXRuntime')
  parser.add_argument('--device', default='CPU', help="Device to perform inference on 'cpu (MLAS)' or on devices supported by OpenVINO-EP [CPU, GPU, NPU].")
  parser.add_argument('--video', help='Path to video file.')
  parser.add_argument('--model', help='Path to model.')
  args = parser.parse_args()
  return args

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

def softmax(x):
  score_mat_exp = np.exp(np.asarray(x))
  return score_mat_exp / score_mat_exp.sum(0)

def check_model_extension(fp):
  # Split the extension from the path and normalise it to lowercase.
  ext = os.path.splitext(fp)[-1].lower()

  # Now we can simply use != to check for inequality, no need for wildcards.
  if(ext != ".onnx"):
    raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")
  
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the onnx model file is Invalid")

def check_video_file_extension(fp):
  # Split the extension from the path and normalise it to lowercase.
  ext = os.path.splitext(fp)[-1].lower()
  # Now we can simply use != to check for inequality, no need for wildcards.
  
  if(ext == ".mp4" or ext == ".avi" or ext == ".mov"):
    pass
  else:
    raise Exception(fp, "is an unknown file format. Use the video file ending with .mp4 or .avi or .mov formats")
  
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the video file is Invalid")

def image_preprocess(frame):
  in_frame = cv2.resize(frame, (416, 416))
  preprocessed_image = np.asarray(in_frame)
  preprocessed_image = preprocessed_image.astype(np.float32)
  preprocessed_image = preprocessed_image.transpose(2,0,1)
  #Reshaping the input array to align with the input shape of the model
  preprocessed_image = preprocessed_image.reshape(1,3,416,416)
  return preprocessed_image

def postprocess_output(out, frame, x_scale, y_scale, i):
  out = out[0][0]
  num_classes = 20
  anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
  existing_labels = {l: [] for l in label}

  #Inside this loop we compute the bounding box b for grid cell (cy, cx)
  for cy in range(0,13):
    for cx in range(0,13):
      for b in range(0,5):
      # First we read the tx, ty, width(tw), and height(th) for the bounding box from the out array, as well as the confidence score
        channel = b*(num_classes+5)
        tx = out[channel  ][cy][cx]
        ty = out[channel+1][cy][cx]
        tw = out[channel+2][cy][cx]
        th = out[channel+3][cy][cx]
        tc = out[channel+4][cy][cx]

        x = (float(cx) + sigmoid(tx))*32
        y = (float(cy) + sigmoid(ty))*32
        w = np.exp(tw) * 32 * anchors[2*b]
        h = np.exp(th) * 32 * anchors[2*b+1] 

        #calculating the confidence score
        confidence = sigmoid(tc) # The confidence value for the bounding box is given by tc
        classes = np.zeros(num_classes)
        for c in range(0,num_classes):
          classes[c] = out[channel + 5 +c][cy][cx]
          # we take the softmax to turn the array into a probability distribution. And then we pick the class with the largest score as the winner.
          classes = softmax(classes)
          detected_class = classes.argmax()
          # Now we can compute the final score for this bounding box and we only want to keep the ones whose combined score is over a certain threshold
          if 0.60 < classes[detected_class]*confidence:
            color =clut[detected_class]
            x = (x - w/2)*x_scale
            y = (y - h/2)*y_scale
            w *= x_scale
            h *= y_scale
               
            labelX = int((x+x+w)/2)
            labelY = int((y+y+h)/2)
            addLabel = True
            lab_threshold = 100
            for point in existing_labels[label[detected_class]]:
              if labelX < point[0] + lab_threshold and labelX > point[0] - lab_threshold and \
                 labelY < point[1] + lab_threshold and labelY > point[1] - lab_threshold:
                  addLabel = False
              #Adding class labels to the output of the frame and also drawing a rectangular bounding box around the object detected.
            if addLabel:
              cv2.rectangle(frame, (int(x),int(y)),(int(x+w),int(y+h)),color,2)
              cv2.rectangle(frame, (int(x),int(y-13)),(int(x)+9*len(label[detected_class]),int(y)),color,-1)
              cv2.putText(frame,label[detected_class],(int(x)+2,int(y)-3),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
              existing_labels[label[detected_class]].append((labelX,labelY))
            print('{} detected in frame {}'.format(label[detected_class],i))
  

def show_bbox(device, frame, inference_time):
  cv2.putText(frame,device,(10,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
  cv2.putText(frame,'FPS: {}'.format(1.0/inference_time),(10,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  cv2.imshow('frame',frame)

def main():
  
  # Process arguments
  args = parse_arguments()

  # Validate model file path
  check_model_extension(args.model)
  so = rt.SessionOptions()
  so.log_severity_level = 3
  if (args.device == 'cpu'):
    print("Device type selected is 'cpu' which is the default CPU Execution Provider (MLAS)")
    #Specify the path to the ONNX model on your machine and register the CPU EP
    sess = rt.InferenceSession(args.model, so, providers=['CPUExecutionProvider'])
  elif (args.device == 'CPU', args.device == 'GPU' or args.device == 'NPU'):
    #Specify the path to the ONNX model on your machine and register the OpenVINO EP
    sess = rt.InferenceSession(args.model, so, providers=['OpenVINOExecutionProvider'], provider_options=[{'device_type' : args.device}])
    print("Device type selected is: " + args.device + " using the OpenVINO Execution Provider")
    '''
    other 'device_type' options are: (Any hardware target can be assigned if you have the access to it)
    'CPU', 'GPU', 'NPU'
    '''
  else:
    raise Exception("Device type selected is not [CPU, GPU, NPU]")

  # Get the input name of the model
  input_name = sess.get_inputs()[0].name

  #validate video file input path
  check_video_file_extension(args.video)

  #Path to video file has to be provided
  cap = cv2.VideoCapture(args.video)

  # capturing different metrics of the image from the video
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  x_scale = float(width)/416.0  #In the document of tino-yolo-v2, input shape of this network is (1,3,416,416).
  y_scale = float(height)/416.0      
 
  # writing the inferencing output as a video to the local disk
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  output_video_name = args.device + "_output.avi"
  output_video = cv2.VideoWriter(output_video_name,fourcc, float(17.0), (640,360))

  # capturing one frame at a time from the video feed and performing the inference
  i = 0
  while cv2.waitKey(1) < 0:
    l_start = time.time()
    ret, frame = cap.read()
    if not ret:
      break
    initial_w = cap.get(3)
    initial_h = cap.get(4)
        
    # preprocessing the input frame and reshaping it.
    #In the document of tino-yolo-v2, input shape of this network is (1,3,416,416). so we resize the model frame w.r.t that size.
    preprocessed_image =  image_preprocess(frame)

    start = time.time()
    #Running the session by passing in the input data of the model
    out = sess.run(None, {input_name: preprocessed_image})
    end = time.time()
    inference_time = end - start

    #Get the output
    postprocess_output(out, frame, x_scale, y_scale, i)
   
    #Show the Output
    output_video.write(frame)
    show_bbox(args.device, frame, inference_time)
        
    #Press 'q' to quit the process
    print('Processed Frame {}'.format(i))
    i += 1
    l_end = time.time()
    print('Loop Time = {}'.format(l_end - l_start))

  output_video.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()