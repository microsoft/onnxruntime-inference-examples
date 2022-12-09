from PIL import Image
from numpy import asarray
import onnxruntime

# load the image
image = asarray(Image.open('kimono.jpg'))

image_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(image)

# Load the model
session = onnxruntime.InferenceSession('mobilenetv2-12-aug.onnx')

# Run the model
results = session.run(["top_classes", "top_probs"], {"image": image_ortvalue})

print(results)