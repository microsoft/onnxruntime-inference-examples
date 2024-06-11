# Local Chatbot in the browser using Phi3, ONNX Runtime Web and WebGPU

This repository contains an example of running [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) in your browser using [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) with WebGPU.

You can try out the live demo [here](https://guschmue.github.io/ort-webgpu/chat/index.html).

We keep this example simple and use the onnxruntime-web api directly. ONNX Runtime Web has been powering 
higher level frameworks like [transformers.js](https://github.com/xenova/transformers.js).

## Getting Started

### Prerequisites

Ensure that you have [Node.js](https://nodejs.org/) installed on your machine.

### Installation

Install the required dependencies:

```sh
npm install
```

### Building the project

Build the project:

```sh
npm run build
```

The output can be found in the ***dist*** directory.

### Building for developent

```sh
npm run dev
```

This will build the project and start a dev server.
Point your browser to http://localhost:8080/.

### The Phi3 ONNX Model

The model used in this example is hosted on [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web). It is an optimized ONNX version specific to Web and slightly different than the ONNX model for CUDA or CPU:
1. The model output 'logits' is kept as float32 (even for float16 models) since Javascript does not support float16.
2. Our WebGPU implementation uses the custom Multiheaded Attention operator instread of Group Query Attention.
3. Phi3 is larger then 2GB and we need to use external data files. To keep them cacheable in the browser,
 both model.onnx and model.onnx.data are kept under 2GB.

If you like to optimize your fine-tuned pytorch Phi-3-min model, you can use [Olive](https://github.com/microsoft/Olive/) which supports float data type conversion and [ONNX genai model builder toolkit](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models).
An example how to optimize Phi-3-min model for ONNX Runtime Web with Olive can be found [here](https://github.com/microsoft/Olive/tree/main/examples/phi3).
