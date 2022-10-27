import onnx
import os
from onnx import helper
from onnx import shape_inference


model = onnx.load("vgg16.onnx")
model_output_name = model.graph.output[0].name
softmax_input_name = model_output_name + '_softmax'

for node in model.graph.node:
    if node.output[0] == model_output_name:
        if node.op_type == 'Softmax':
            print('Already has Softmax inserted to then end! No need to do it again')
            exit()
        node.output[0] = softmax_input_name

output_transpose_node = helper.make_node('Softmax',
                                        name= softmax_input_name,
                                        inputs=[softmax_input_name],
                                        outputs=[model_output_name])

model.graph.node.extend([output_transpose_node])

os.rename("vgg16.onnx", "vgg16_original.onnx")

onnx.save(model, "vgg16.onnx")