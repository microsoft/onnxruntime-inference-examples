# How to generate the model

## Install the Python requirements

It is a good idea to use a clean Python environment, e.g., a new Conda environment.

Run:

```bash
python -m pip install -r <this directory>/requirements.txt
```

## Run the model generation script

Run:

```bash
<this directory>/gen_model.sh <output directory>
```

The model will be generated in the given output directory.

In particular, .onnx and .ort model files will be generated.
The .ort model file can be used by ONNX Runtime Mobile.
