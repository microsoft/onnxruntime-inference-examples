#!/usr/bin/env python3
import cv2
import os
import sys
import numpy as np
from PIL import Image
from onnxruntime.quantization import quantize, quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
import onnxruntime as ort
import onnx
from onnx import shape_inference


from onnx import numpy_helper

def get_image(path, show=False):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    if show:
        plt.imshow(img)
        plt.axis('off')
    return img

def preprocess(image):
    # resize so that the shorter side is 256, maintaining aspect ratio
    def image_resize(image, min_len):
        image = Image.fromarray(image)
        ratio = float(min_len) / min(image.size[0], image.size[1])
        if image.size[0] > image.size[1]:
            new_size = (int(round(ratio * image.size[0])), min_len)
        else:
            new_size = (min_len, int(round(ratio * image.size[1])))
        image = image.resize(new_size, Image.BILINEAR)
        return np.array(image)
    image = image_resize(image, 256)

    # Crop centered window 224x224
    def crop_center(image, crop_w, crop_h):
        h, w, c = image.shape
        start_x = w//2 - crop_w//2
        start_y = h//2 - crop_h//2
        return image[start_y:start_y+crop_h, start_x:start_x+crop_w, :]
    image = crop_center(image, 224, 224)

    # transpose
    image = image.transpose(2, 0, 1)

    # convert the input data into the float32 input
    img_data = image.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def predict(image_path, model_path):
    img = get_image(image_path)
    img = preprocess(img)
    img.tofile('kitten_input.raw')
    img.transpose(0, 2, 3, 1).tofile('kitten_input_nhwc.raw')
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    with open('./synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    print('class=%s; probability=%f' %(labels[a[0]],preds[a[0]]))

# The more image files the better to improve quantzation accuracy
def GetImages(data_dir):
    list_ = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            img = get_image(os.path.join(data_dir, filename))
            img = preprocess(img)
            list_.append(img)
    return list_

# Generate a new model which changes dynamic shape to static shape
def DynamicShapeToStaticShape(model_file, static_shape_model_file):
    mp = onnx.load(model_file)
    mp.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1    
    mp.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1    
    mp = shape_inference.infer_shapes(mp)
    onnx.save(mp, static_shape_model_file)

class GenerateRandomCalibrationData(CalibrationDataReader):
    def __init__(self, data_dir):
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.data_dir = data_dir

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            data_list = GetImages(self.data_dir)
            self.enum_data_dicts = iter([{"input": data} for data in data_list])
        return next(self.enum_data_dicts, None)

def main():
    input_model_file = 'mobilenetv2-12.onnx'
    qdq_model_file = input_model_file.replace(".onnx", "_quant.onnx")
    static_shape_qdq_model_file = qdq_model_file.replace(".onnx", "_shape.onnx")
    dr = GenerateRandomCalibrationData('./images')

    qnn_config = get_qnn_qdq_config(input_model_file,
                                    dr,
                                    activation_type=QuantType.QUInt8,
                                    weight_type=QuantType.QUInt8)
    quantize(input_model_file, qdq_model_file, qnn_config)    
    print('Calibrated and quantized model saved.')

    print('Run with quantized model.')
    predict('./images/kitten.jpg', qdq_model_file)

    # Change the fp32 model to static shape
    DynamicShapeToStaticShape(input_model_file, input_model_file.replace(".onnx", "_shape.onnx"))
    # Change the QDQ model to static shape
    DynamicShapeToStaticShape(qdq_model_file, static_shape_qdq_model_file)

if __name__ == '__main__':
    main()
