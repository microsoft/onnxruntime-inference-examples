#!/usr/bin/env python3
import os
import sys
import argparse
import onnx
from onnxconverter_common import float16

def main(args):
    input_model_path = args.input_model
    output_model_path = os.path.join(os.path.dirname(input_model_path), os.path.basename(input_model_path).replace(".onnx", "_fp16.onnx"))
    
    model = onnx.load(input_model_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_model_path)
    print('fp16 model saved.')

def parse_args(args_list):
    parser = argparse.ArgumentParser(description="Generate onnx from TF1")
    parser.add_argument('-m', "--input_model",help="path to pb model", default='mobilenetv2-12_shape.onnx')
    
    args = parser.parse_args(args_list)
    return args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
