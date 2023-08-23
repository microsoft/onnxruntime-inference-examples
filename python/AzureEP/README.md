Since onnxruntime 1.16, AzureEP supports the running of a "hybrid" model combining two compoments:

- An edge model that runs locally.
- A proxy model talks to a remote endpoint hosted hosting a model as a service.

There are three ways to create such a "hybrid" model:

- Run either one.
- Run both.
- Run the first model, then the second if need to.

Also, there are two examples for usage:

- Create hybrid models over a local tiny-yolo model and a proxy model talks to yolov2-coco hosted on [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-with-triton?view=azureml-api-2&tabs=azure-cli%2Cendpoint).
- Create hybrid models over a local whisper tiny model and a proxy model talks to [OpenAI audio service](https://api.openai.com/v1/audio/transcriptions).
