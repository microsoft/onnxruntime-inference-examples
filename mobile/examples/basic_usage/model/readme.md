# How to generate the model

## Install the Python requirements

It is a good idea to use a clean Python environment, e.g., a new Conda environment.

From this directory, run:

```bash
python -m pip install -r requirements.txt
```

## Run the model generation script

From this directory, run:

```bash
gen_model.sh <output directory>
```

The model will be generated in the given output directory.

In particular, .onnx and .ort model files will be generated.
The .ort model file can be used by ONNX Runtime Mobile.

