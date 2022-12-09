import torch
import torchvision.transforms as transforms
import numpy
from PIL import Image
import onnxruntime

# Pre-processing function for ImageNet models
# (N x 3 x H x W)
def preprocess(img):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]) 

    preprocessing = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       normalize])

    return preprocessing(img).numpy()[numpy.newaxis, ...]

# Output of the model is a list of raw scores              
def postprocess(scores):
    probabilities = torch.softmax(scores, dim=1)
    top10_prob, top10_ids = probabilities.topk(k=10, dim=1, largest=True, sorted=True)
    return top10_ids, top10_prob


# Load the model
session = onnxruntime.InferenceSession('mobilenetv2-12.onnx')

# load the image
image = Image.open('kimono.jpg')

# Run the model
image_data = preprocess(image)
scores = session.run(["output"], {"input": image_data})
results = postprocess(torch.Tensor(scores[0]))

print(results)    