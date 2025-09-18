# A set of code samples showing different usage of the ONNX Runtime Python API
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import os
import re
import onnxruntime

MODEL_FILE = '.model.onnx'
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_INDEX = 0     # Replace this with the index of the device you want to run on
DEVICE=f'{DEVICE_NAME}:{DEVICE_INDEX}'
LIB_EXT = 'so' if os.name != 'nt' else 'dll'

# A simple model to calculate addition of two tensors
def model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x, y):
            return x.add(y)

    return Model()

# Create an instance of the model and export it to ONNX graph format, with dynamic size for the data
def create_model(type: torch.dtype = torch.float32):
    sample_x = torch.ones(3, dtype=type)
    sample_y = torch.zeros(3, dtype=type)

    torch.onnx.export(model(), (sample_x, sample_y), MODEL_FILE, input_names=["x", "y"], output_names=["z"],
                               dynamic_axes={"x": {0 : "array_length_x"}, "y": {0: "array_length_y"}})
 
# Create an ONNX Runtime session with the provided model
def create_session(model: str) -> onnxruntime.InferenceSession:
    available_providers = {device.ep_name for device in  onnxruntime.get_ep_devices()}
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        if 'CUDAExecutionProvider' in available_providers:
            providers.insert(0, 'CUDAExecutionProvider')
        if 'NvTensorRTRTXExecutionProvider' in available_providers:
            providers.insert(0, 'NvTensorRTRTXExecutionProvider')
    return onnxruntime.InferenceSession(model, providers=providers)


# Run the model on device consuming and producing ORTValues
def run_with_data_on_device(x: np.array, y: np.array) -> onnxruntime.OrtValue:
    session = create_session(MODEL_FILE)
    mem_info = session.get_input_memory_infos()[0]

    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x, 'gpu', device_id=mem_info.device_id, vendor_id=mem_info.device_vendor_id)
    y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(y, 'gpu', device_id=mem_info.device_id, vendor_id=mem_info.device_vendor_id)

    io_binding = session.io_binding()
    io_binding.bind_input(name='x', device_type=x_ortvalue.device_name(), device_id=mem_info.device_id, element_type=x.dtype, shape=x_ortvalue.shape(), buffer_ptr=x_ortvalue.data_ptr())
    io_binding.bind_input(name='y', device_type=y_ortvalue.device_name(), device_id=mem_info.device_id, element_type=y.dtype, shape=y_ortvalue.shape(), buffer_ptr=y_ortvalue.data_ptr())
    io_binding.bind_output(name='z', device_type=x_ortvalue.device_name(), device_id=mem_info.device_id, element_type=x.dtype, shape=x_ortvalue.shape())
    session.run_with_iobinding(io_binding)

    z = io_binding.get_outputs()

    return z[0]

# Run the model on device consuming and producing native PyTorch tensors
def run_with_torch_tensors_on_device(x: torch.Tensor, y: torch.Tensor, np_type: np.dtype = np.float32, torch_type: torch.dtype = torch.float32, dlpack=False) -> torch.Tensor:
    session = create_session(MODEL_FILE)
    mem_info = session.get_input_memory_infos()[0]

    binding = session.io_binding()

    x_tensor = x.contiguous()
    y_tensor = y.contiguous()

    binding.bind_input(
        name='x',
        device_type="gpu",
        device_id=mem_info.device_id,
        element_type=np_type,
        shape=tuple(x_tensor.shape),
        buffer_ptr=x_tensor.data_ptr(),
        )

    binding.bind_input(
        name='y',
        device_type="gpu",
        device_id=mem_info.device_id,
        element_type=np_type,
        shape=tuple(y_tensor.shape),
        buffer_ptr=y_tensor.data_ptr(),
        )
    if dlpack:
        binding.bind_output(
            name='z',
            device_type="gpu",
        )
    else:
        ## Allocate the PyTorch tensor for the model output
        z_tensor = torch.empty(x_tensor.shape, dtype=torch_type, device=DEVICE).contiguous()
        binding.bind_output(
            name='z',
            device_type="gpu",
            device_id=mem_info.device_id,
            element_type=np_type,
            shape=tuple(z_tensor.shape),
            buffer_ptr=z_tensor.data_ptr(),
        )

    session.run_with_iobinding(binding)
    if dlpack:
        from onnxruntime.capi import _pybind_state as C
        outputs = binding.get_outputs()
        return torch.tensor(C.OrtValue.from_dlpack(outputs[0]._ortvalue.to_dlpack(), False))
    else:
        return z_tensor


def main():
    # check if plugin based providers are available and register them
    ort_capi_dir = os.path.dirname(onnxruntime.capi.__file__)
    for p in  os.listdir(ort_capi_dir):
        match = re.match(r".*onnxruntime_providers_(.*)\."+LIB_EXT, p)
        if match is not None:
            ep_name = match.group(1)
            if ep_name == 'shared': continue
            onnxruntime.register_execution_provider_library(ep_name, os.path.join(ort_capi_dir, p))
            print(f"Registered execution provider {ep_name} with library: {p}")

    create_model()

    print(run_with_data_on_device(x=np.float32([1.0, 2.0, 3.0, 4.0, 5.0]), y=np.float32([1.0, 2.0, 3.0, 4.0, 5.0])).numpy())
    # [ 2.  4.  6.  8. 10.]

    x = torch.rand(5).to(DEVICE)
    y = torch.rand(5).to(DEVICE)
    print(run_with_torch_tensors_on_device(x, y, dlpack=True))
    # tensor([0.7023, 1.3127, 1.7289, 0.3982, 0.8386])

    print(run_with_torch_tensors_on_device(x, y, dlpack=False))
    # tensor([0.7023, 1.3127, 1.7289, 0.3982, 0.8386])

    create_model(torch.int64)
 
    print(run_with_torch_tensors_on_device(torch.ones(5, dtype=torch.int64).to(DEVICE), torch.zeros(5, dtype=torch.int64).to(DEVICE), np_type=np.int64, torch_type=torch.int64))
    # tensor([1, 1, 1, 1, 1])


if __name__ == "__main__":
    main()   
