Since onnxruntime 1.16, AzureEP support the running of a "hybrid" model combining two compoments:

- An edge model that runs locally.
- A proxy model talks to a remote endpoint hosted on Azure.

The file implement three ways to create such a "hybrid" model:

- Run either one.
- Run both.
- Run the first model, then the second if need to.

In the end, there are demos of usage over a local tiny-yolo model and a proxy model talks to yolov2-coco hosted on Azure.
For how to deploy a model to Azure Machine Learning, pls refer to the [link](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-with-triton?view=azureml-api-2&tabs=azure-cli%2Cendpoint).