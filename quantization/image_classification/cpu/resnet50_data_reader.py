import numpy
import onnxruntime
import os
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = numpy.float32(pillow_img) - numpy.array(
            [123.68, 116.78, 103.94], dtype=numpy.float32
        )
        nhwc_data = numpy.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.image_folder = calibration_image_folder
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.model_path, None)
            (_, _, height, width) = session.get_inputs()[0].shape
            nhwc_data_list = _preprocess_images(
                self.image_folder, height, width, size_limit=0
            )
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter(
                [{input_name: nhwc_data} for nhwc_data in nhwc_data_list]
            )
        return next(self.enum_data_dicts, None)
