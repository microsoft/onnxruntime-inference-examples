# Basic Usage Example Model

The basic usage example uses a simple model that adds two floats.

The inputs are named `A` and `B` and the output is named `C` (A + B = C).
All inputs and outputs are float tensors with shape [1].

## How to generate the model

### Install the Python requirements

It is a good idea to use a separate Python environment instead of the system Python.
E.g., a new Conda environment.

Run:

```bash
python3 -m pip install -r <this directory>/requirements.txt
```

### Run the model generation script

Run:

```bash
<this directory>/gen_model.sh <output directory>
```

The model will be generated in the given output directory.

In particular, .onnx and .ort model files will be generated.
The .ort model file can be used by ONNX Runtime Mobile.
