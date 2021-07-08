# Wav2Vec 2.0

This example uses the [Wav2Vec 2.0](https://huggingface.co/transformers/model_doc/wav2vec2.html) model for speech recognition.

The model generation script was adapted from [this PyTorch example script](https://github.com/pytorch/ios-demo-app/blob/f2b9aa196821c136d3299b99c5dd592de1fa1776/SpeechRecognition/create_wav2vec2.py).

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
