# A set of code samples showing different usage of the ONNX Runtime Python API
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import onnxruntime

MODEL_FILE = '.model.onnx'
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    return MODEL_FILE

# Create an ONNX Runtime session with the provided model
def create_session(model: str) -> onnxruntime.InferenceSession:
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    return onnxruntime.InferenceSession(model, providers=providers)

# Run the model on CPU consuming and producing numpy arrays
def run(x: np.array, y: np.array) -> np.array:
    session = create_session(MODEL_FILE)

    z = session.run(["z"], {"x": x, "y": y})

    return z[0]

def main():
    create_model()

    print(run(x=np.float32([1.0, 2.0, 3.0]),y=np.float32([4.0, 5.0, 6.0])))
    # [array([5., 7., 9.], dtype=float32)]

if __name__ == "__main__":
    main()
