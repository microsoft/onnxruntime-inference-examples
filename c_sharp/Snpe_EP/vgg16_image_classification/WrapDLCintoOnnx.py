import onnx
from onnx import helper
from onnx import TensorProto

with open('./vgg16_q.dlc','rb') as file:
    file_content = file.read()

input1 = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 224, 224, 3])
output1 = helper.make_tensor_value_info('vgg0_dense2_fwd', TensorProto.FLOAT, [1, 1000])

snpe_node = helper.make_node('Snpe', name='snpe concat', inputs=['data'], outputs=['vgg0_dense2_fwd'], DLC=file_content, snpe_version='1.61.46', target_device='CPU', notes='VGG16 dlc model.', domain='com.microsoft')

graph_def = helper.make_graph([snpe_node], 'snpe dlc', [input1], [output1])
model_def = helper.make_model(graph_def, producer_name='onnx', opset_imports=[helper.make_opsetid('', 13)])
onnx.save(model_def, 'vgg16_dlc_q.onnx')