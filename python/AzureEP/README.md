Since onnxruntime 1.16, AzureEP supports the running of a "hybrid" model combining two components:

- An edge model that runs locally.
- A proxy model talks to an online service endpoint hosting a model.

In create_hybrid_model.py, there are three ways suggested to create such a "hybrid" model:

- Run either one.
- Run both.
- Run the first model, then the second if need to.

Also, the script provides two examples for usage:

- Create hybrid models over a local tiny-yolo model and a proxy model talks to yolov2-coco hosted on [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-with-triton?view=azureml-api-2&tabs=azure-cli%2Cendpoint).
- Create hybrid models over a local whisper tiny model and a proxy model talks to [OpenAI Audio Service](https://platform.openai.com/docs/api-reference/audio).
