# ONNX Runtime Quantization Example

This folder contains example code for quantizing Resnet50 or MobilenetV2 models. The example has
three parts:

1. Pre-processing
2. Quantization
3. Debugging

## Pre-processing

Pre-processing prepares a float32 model for quantization. Run the following command to pre-process
model `mobilenetv2-7.onnx`.

Model `resnet50-v1-12.onnx` can be downloaded from [ONNX repo](https://github.com/onnx/models/tree/main/vision/classification/resnet/model).

```console
python -m onnxruntime.quantization.preprocess --input mobilenetv2-7.onnx --output mobilenetv2-7-infer.onnx
```

The pre-processing consists of the following optional steps
- Symbolic Shape Inference. It works best with transformer models.
- ONNX Runtime Model Optimization.
- ONNX Shape Inference.

Quantization requires tensor shape information to perform its best. Model optimization
also improve the performance of quantization. For instance, a Convolution node followed
by a BatchNormalization node can be merged into a single node during optimization.
Currently we can not quantize BatchNormalization by itself, but we can quantize the
merged Convolution + BatchNormalization node.

It is highly recommended to run model optimization in pre-processing instead of in quantization.
To learn more about each of these steps and finer controls, run:
```console
python -m onnxruntime.quantization.preprocess --help
```

## Quantization

Quantization tool takes the pre-processed float32 model and produce a quantized model.
It's recommended to use Tensor-oriented quantization (QDQ; Quantize and DeQuantize).

```console
python run.py --input_model mobilenetv2-7-infer.onnx --output_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/
```
This will generate quantized model mobilenetv2-7.quant.onnx

The code in `run.py` creates an input data reader for the model, uses these input data to run
the model to calibrate quantization parameters for each tensor, and then produces quantized
model. Last, it runs the quantized model. Of these step, the only part that is specific to
the model is the input data reader, as each model requires different shapes of input data.
All other code can be easily generalized for other models.

For historical reasons, the quantization API performs model optimization by default.
It's highly recommended to turn off model optimization using parameter
`optimize_model=False`. This way, it is easier for the quantization debugger to match
tensors of the float32 model and its quantized model, facilitating the triaging of quantization
loss.

## Debugging

Quantization is not a loss-less process. Sometime it results in significant loss in accuracy.
To help locate the source of these losses, our quantization debugging tool matches up
weight tensors of the float32 model vs those of the quantized model.  If a input data reader
is provided, our debugger can also run both models with the same input and compare their
corresponding tensors:

```console
python run_qdq_debug.py --float_model mobilenetv2-7-infer.onnx --qdq_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/
```

If you have quantized a model with optimization turned on, and found the debugging tool can not
match certain float32 model tensors with their quantized counterparts, you can try to run the
pre-processor to produce the optimized model, then compare the optimized model with the quantized model.

For instance, you have a model `abc_float32_model.onnx`, and a quantized model
`abc_quantized.onnx`. During quantization process, you had optimization turned on
by default. You can run the following code to produce an optimized float32 model:

```console
python -m onnxruntime.quantization.preprocess --input abc_float32_model.onnx --output abc_optimized.onnx --skip_symbolic_shape True
```

Then run the debugger comparing `abc_optimized.onnx` with `abc_quantized.onnx`.
