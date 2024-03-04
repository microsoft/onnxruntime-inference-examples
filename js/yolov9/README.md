# yolov9: Browser based realtime object detection with WebGPU and ONNX Runtime Web

This repository contains an example of running [yolov9](https://huggingface.co/Xenova/yolov9-c) in a browser using [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) with WebGPU.

You can try out the live demo [here](https://guschmue.github.io/ort-webgpu/yolov9/index.html).

## Model Overview

See [here](https://arxiv.org/abs/2402.13616) for the research paper and [here](https://github.com/WongKinYiu/yolov9) for the implementation.

## Getting Started

### Prerequisites

Ensure that you have [Node.js](https://nodejs.org/) installed on your machine.

### Installation

1. Install the required dependencies:

```sh
npm install
```

### Building the Project

1. Bundle the code using webpack:

```sh
npm run build
```

This command generates the bundle file `./dist/index.js`.

### The ONNX Model

The model used in this project is hosted on [Hugging Face](https://huggingface.co/Xenova/yolov9-c). 

### Running the Project

Start a web server to serve the current folder at http://localhost:8888/. To start the server, run:

```sh
npm run dev
```
