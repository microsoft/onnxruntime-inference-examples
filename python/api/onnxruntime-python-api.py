# A set of code samples showing different usage of the ONNX Runtime Python API
import numpy as np
import torch
import onnxruntime

model_name = '.model.onnx'

def model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x):
            return torch.std_mean(x)

    return Model()

def create_model():
    x = torch.rand(3)
    torch.onnx.export(model(), x, model_name, input_names=["x"], output_names=["s", "m"], dynamic_axes={"x": {0 : "array_length"}})

def run_default(x):
    sess = onnxruntime.InferenceSession(model_name)
    y = sess.run(["s", "m"], {"x": x})
    print(y)    

def run_with_data_on_cpu(x):
   ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x, 'cpu')
   ortvalue.device_name()  # 'cpu'
   ortvalue.shape()        # shape of the numpy array X
   ortvalue.data_type()    # 'tensor(float)'
   ortvalue.is_tensor()    # 'True'
   np.array_equal(ortvalue.numpy(), x)  # 'True'

   print(f'device_name: {ortvalue.device_name()}')
   print(f'shape: {ortvalue.shape()}')
   print(f'data_type: {ortvalue.data_type()}')
   print(f'is_tensor: {ortvalue.is_tensor()}')
   # ortvalue can be provided as part of the input feed to a model
   sess = onnxruntime.InferenceSession(model_name)
   y = sess.run(["s", "m"], {"x": ortvalue})
   print(y)

def run_with_data_on_device(x):
    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x, 'cpu', 0)
    y = x=np.float32([1.0, 1.0])
    y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(y, 'cpu', 0) 

    print(f'3: y_ortvalue.shape: {y_ortvalue.shape()}')
    session = onnxruntime.InferenceSession(model_name)
    io_binding = session.io_binding()
    io_binding.bind_input(name='x', device_type=y_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=x_ortvalue.shape(), buffer_ptr=x_ortvalue.data_ptr())
    io_binding.bind_output(name='s', device_type=y_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=y_ortvalue.shape(), buffer_ptr=y_ortvalue.data_ptr())
    session.run_with_iobinding(io_binding)

    print(x_ortvalue)
    print(y_ortvalue)


def main():
    create_model()

    run_default(x=np.float32([1.0, 2.0, 3.0]))

    run_with_data_on_cpu(x=np.float32([1.0, 2.0, 3.0, 4.0]))

    run_with_data_on_device(x=np.float32([5.0, 10.0]))

    

if __name__ == "__main__":
    main()   