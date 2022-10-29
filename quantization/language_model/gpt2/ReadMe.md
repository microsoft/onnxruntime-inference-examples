# Example of GPT-2-medium Quantization Example

This folder contains example code for quantizing GPT2-medium model. This is by an large similar to
[this example](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu).

## Obtaining the 32-bit floating point model

ONNX Runtime provides tools for converting GPT2 models to ONNX, run:

```console
python -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m gpt2-medium --output gpt2_medium_fp32.onnx -o -p fp32
```


## Preparing the floating point model for quantization

Here we pre-process the model, essentially run shape inferences and model optimization, both of
which may improve the performance of quantization.

```console
python -m onnxruntime.quantization.preprocess --input gpt2_medium_fp32.onnx --output gpt2_medium_fp32_preprocessed.onnx
```

## Quantize 

We use static quantization here, for which a calibration data set is required. You can run
`generate_inputs.py` to generate random dummy input for gpt-2 medium. See the python source
code for finer control options


With calibration data set, run the following command to invoke the quantization tool, which
will run the model with provided data set, compute quantization parameters for each
weight and activation tensors, and output the quantized model:

```console
python run_qdq.py --input_model gpt2_medium_fp32_preprocessed.onnx --output_model gpt2_medium_quant.onnx --calibrate_dataset ./test_input
```

## Quantization Debugging

Python file `run_qdq_debug.py` showcase how to use our quantization debugging API to match up
corresponding weight/activation tensors between floating point and quantized models. Run

```console
python run_qdq_debug.py --float_model gpt2_medium_fp32_preprocessed.onnx --qdq_model gpt2_medium_quant.onnx  --calibrate_dataset ./test_input
```

