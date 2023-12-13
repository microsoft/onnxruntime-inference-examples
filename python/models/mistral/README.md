# Mistral 7B v0.1 Inference Benchmarking

This demo will show how to run Inference benchmark for comparing ONNX Runtime with Torch Eager mode and Torch compile.

## Background

[Mistral 7B v0.1](https://mistral.ai/news/announcing-mistral-7b/) Large Language Model (LLM) is a pre-trained generative text model with 7 billion parameters.

## Set up

### AzureML
The easiest option to run the demo will be using AzureML as the environment details are already included, there is another option to run directly on the machine which is provided later. For AzureML, please complete the following prerequisites:

#### Local environment
Set up your local environment with az-cli and azureml dependency for script submission:

```
az-cli && az login
pip install azure-ai-ml azure-identity
```

#### AzureML Workspace
- An AzureML workspace is required to run this demo. Download the config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) for your workspace. Make sure to put this config file in this folder and name it ws_config.json.
- The workspace should have a gpu cluster. This demo was tested with GPU cluster of SKU [Standard_ND40rs_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python). We do not recommend running this demo on `NC` series VMs which uses old architecture (K80).
- Additionally, you'll need to create a [Custom Curated Environment ACPT](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) with PyTorch >=2.0.1 and the requirements file in the environment folder:

## Run Experiments
The demo is ready to be run.

#### `aml_submit.py` submits an training job to AML for both Pytorch+DeepSpeed+LoRA and ORT+DeepSpeed+LoRA. This job builds the training environment and runs the fine-tuning script in it.

```bash
python aml_submit.py
```

The above script will generate a URL showing the prompt processing and token generation time for each case..


### Run directly on your compute

If you are using CLI by directly logging into your machine then you can follow the below instructions. The below steps assume you have the required packages like Pytorch, ORT Nightly GPU, Transformers and more already installed in your system. For easier setup, you can look at the environment folder.

```bash
cd inference-code
bash inference_setup.sh
```

