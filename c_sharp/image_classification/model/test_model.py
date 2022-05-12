import torch
import onnxruntime

# Read an image
image = torch.ones(500, 400, 3).to(torch.uint8).numpy()
image_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(image)

# Load the model
session = onnxruntime.InferenceSession('mobilenetv2-7-aug.onnx')

# Run the model
results = session.run(["top_classes", "top_probs"], {"image": image_ortvalue})

print(results)