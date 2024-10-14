# this script was adapted from here:
# https://github.com/pytorch/ios-demo-app/blob/f2b9aa196821c136d3299b99c5dd592de1fa1776/SpeechRecognition/create_wav2vec2.py

import torch
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from transformers import Wav2Vec2ForCTC

# Load Wav2Vec2 pretrained model from Hugging Face Hub
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Convert the model to torchaudio format
model = import_huggingface_model(model)

model = model.eval()

input = torch.zeros(1, 1024)

torch.onnx.export(model, input, "wav2vec2-base-960h.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": [1], "output": [1]},
                  opset_version=14)
