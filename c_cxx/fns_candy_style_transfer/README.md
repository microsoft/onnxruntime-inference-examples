# FNS Candy

FNS Candy is a style transfer model. In this sample application, we use the ONNX Runtime C API to process an image using the FNS (Fast Neural Style) Candy model in ONNX format. It adds a candy looking style to any image.

![Heart shaped image made of read and white hearts](before-candy-image.png)

![Candy style heart shape](after-candy-image.png)

## Build Instructions

See [../README.md](../README.md)

## Prepare data

First, download the FNS Candy ONNX model from [here](https://raw.githubusercontent.com/microsoft/Windows-Machine-Learning/master/Samples/FNSCandyStyleTransfer/UWP/cs/Assets/candy.onnx).

Then, prepare an image:

1. PNG format
2. Dimension of 720x720

## Run

Copy `onnxruntime.dll` into the same folder as `fns_candy_style_transfer.exe`.

Command to run the application:

```bat
fns_candy_style_transfer.exe <model_path> <input_image_path> <output_image_path> [cpu|cuda|dml]
```
