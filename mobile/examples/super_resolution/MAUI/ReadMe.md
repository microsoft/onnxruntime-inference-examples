# Super Resolution MAUI example.

## Original model
https://github.com/ai-forever/Real-ESRGAN

**WARNING**
This is an advanced super resolution model that takes time to run.
If performance is critical please select a simpler model.

## ONNX model
The ONNX model was created using ./real_esrgan_ort_e2e.py.
- PyTorch model was converted to ONNX.
- Pre/post processing was added to the ONNX model using onnxruntime-extensions for the image decode/encode custom ops.

You will need to pip install the dependencies and Real-ESRGAN

```
pip install pillow numpy onnx onnxruntime onnxruntime_extensions torch torchvision
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```

And download the weights to ./weights/REalESRGAN_x4.pth as per https://github.com/ai-forever/Real-ESRGAN/blob/main/weights/README.md

### Notes
- In this simple example we resize the input image to 240x240 which is the size the model supports.
- In the RealESRGAN python model the input image gets padded and split into tiles of size 240x240 so any input size
  can be handled and no initial resize of the input is required.
  - See https://github.com/ai-forever/Real-ESRGAN/blob/362a0316878f41dbdfbb23657b450c3353de5acf/RealESRGAN/model.py#L65-L85
- This padding/tiling/batching is not currently supported in the pre-defined ONNX Runtime pre/post processing steps
  but could be added in the future.
