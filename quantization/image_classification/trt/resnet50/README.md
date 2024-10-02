# ONNX PTQ for using TensorRT EP
Following is the end-to-end example using ORT quantization tool to quantize ONNX model and run/evaluate the quantized model with TRT EP.  

## Note
We suggest to use ImageNet 2012 classification dataset to do the model calibration and evaluation. In addition to the sample code we provide below, TensorRT model optimizer which leverages torchvision.datasets already provides
the ability to work with ImageNet dataset.
