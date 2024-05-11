# Local Chat using Phi3, ONNX Runtime Web and WebGPU

This repository contains an example of running [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) your browser using [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) with WebGPU.

You can try out the live demo [here](https://guschmue.github.io/ort-webgpu/chat/index.html).

We keep this example simple and use the onnxruntime-web api directly without a 
more advanced framework like [transformers.js](https://github.com/xenova/transformers.js).

## Getting Started

### Prerequisites

Ensure that you have [Node.js](https://nodejs.org/) installed on your machine.

### Installation

Install the required dependencies:

```sh
npm install
```

### Building the project

Build the project using vite:

```sh
npm run build
```

The output can be found in the ***dist*** directory.

### Building for developent
For development you can use vite.
You must run ```npm run build``` once to setup the dist directory.

```sh
npm run dev
```

Point your browser to  http://localhost:5173/.

### The ONNX Model

The model used in this project is hosted on [Hugging Face](https://huggingface.co/schmuell/phi3-int4). It was created using the [onnx model builder](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models).

You can create the model with 

```sh
python builder.py -m microsoft/Phi-3-mini-4k-instruct -o $your_output -p int4 -e web
```

