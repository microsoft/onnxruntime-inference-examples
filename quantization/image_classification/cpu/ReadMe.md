# ONNX Runtime Quantization Example

This folder contains example code for quantizing Resnet50 or mobilenetv2 models, which consists of 3 steps:

- Pre-processing
- Quantization
- Debugging


## Pre-processing

Quantization works best with shape inferencing, as not knowing a tensor's shape makes
it harder to quantize it. On the other hand, ONNX shape inferencing works best with
optimized models. So, it is recommended to pre-process the original 32 bit floating
point model with optimization and shape inferencing, before quantization.

```console
python -m onnxruntime.quantization.shape_inference --input mobilenetv2-7.onnx --output mobilenetv2-7-infer.onnx
```

The pre-processing consists of 3 optional sub-steps
- Symbolic Shape Inference. It works best with transformer models.
- ONNX Runtime Model Optimization.
- ONNX Shape Inference

To learn more about these pre-processing steps and how to skip some of them, run:
```console
python -m onnxruntime.quantization.shape_inference --help
```

## Quantization

Quantization tool takes the pre-processed float32 model and produce a quantized model.
It's recommended to use Tensor-oriented quantization (QDQ; Quantize and DeQuantize).

```console
python run.py --input_model mobilenetv2-7-infer.onnx --output_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/
```
This will generate quantized model mobilenetv2-7.quant.onnx

The code in ```run.py``` creates an input data reader for the model, uses these input data to run
the model to calibrate quantization parameters for each tensor, and then produces quantized
model. Last, it runs the quantized model. Of these step, the only part that is specific to
the model is the input data reader, as each model requires different shapes of input data.
All other code can be easily generalized for other models.

For historical reasons, the quantization API performs model optimization by default.
It's highly recommended to turn off model optimization using parameter
```optimize_model=False```. This way, it is easier for the quantization debugger to match
tensors of the float32 model and its quantized model, facilitating the triaging of quantization
loss.

## Debugging

Quantization is not a loss-less process. Sometime it results in significiant loss in accuracy.
To help locate the source of these losses, our quantization debugging tool matches and
compare weights of the float32 model vs those of the quantized model.  If a input data reader
is provided, our debugger can also run both models with the same input and compare their
corresponding tensors:

'''console
python run_qdq_debug.py --float_model mobilenetv2-7-infer.onnx --qdq_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/
'''

For historical reasons, the quantization API performs model optimization by default. If you
have a quantized model with optimization turned on, and found the debugging tool can not match
certain float32 model tensors with their quantized counterparts, you can try running the
debugger again, comparing the optimized float32 model with the quantized model.

For instance, you have a model ```abc_float32_model.onnx```, and a quantized model
```abc_quantized.onnx```. During quantization process, you had optimization turned on
by default. You can run the following code to produce an optimized float32 model:

```console
python -m onnxruntime.quantization.shape_inference --input abc_float32_model.onnx --output abc_optimized.onnx --skip_symbolic_shape True
```

Then run the debugger comparing ```abc_optimized.onnx``` with ```abc_quantized.onnx```.
