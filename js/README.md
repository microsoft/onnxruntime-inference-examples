# ONNX Runtime JavaScript examples

## Summary

This folder contains several JavaScript examples. Most of the examples, unless remarked explicitly, are available in all NPM packages as described below:

- [onnxruntime-node](https://github.com/microsoft/onnxruntime/tree/master/js/node): Node.js binding for ONNXRuntime. Can be used in Node.js applications and Node.js compatible environment (eg. Electron.js).
- [onnxruntime-web](https://github.com/microsoft/onnxruntime/tree/master/js/web): ONNXRuntime on browsers.
- [onnxruntime-react-native](https://github.com/microsoft/onnxruntime/tree/master/js/react_native): ONNXRuntime for React Native applications on Android and iOS.

## Usage

Click links for README of each examples.

### Quick Start

* [Quick Start - Nodejs Binding](quick-start_onnxruntime-node) - a demonstration of basic usage of ONNX Runtime Node.js binding.

* [Quick Start - Nodejs Binding Bundle](quick-start_onnxruntime-node-bundler) - a demonstration of basic usage of ONNX Runtime Node.js binding using bundler.

* [Quick Start - Web (using script tag)](quick-start_onnxruntime-web-script-tag) - a demonstration of basic usage of ONNX Runtime Web using script tag.

* [Quick Start - Web (using bundler)](quick-start_onnxruntime-web-bundler) - a demonstration of basic usage of ONNX Runtime Web using a bundler.

### Importing

* [Importing - Nodejs Binding](importing_onnxruntime-node) - a demonstration of how to import ONNX Runtime Node.js binding.

* [Importing - Web](importing_onnxruntime-web) - a demonstration of how to import ONNX Runtime Web.

* [Importing - React Native](importing_onnxruntime-react-native) - a demonstration of how to import ONNX Runtime React Native.

### API usage

* [API usage - Tensor](api-usage_tensor) - a demonstration of basic usage of `Tensor`.

* [API usage - Tensor <--> Image conversion](api-usage-tensor-image) - a demonstration of conversions from Image elements to and from `Tensor`.

* [API usage - InferenceSession](api-usage_inference-session) - a demonstration of basic usage of `InferenceSession`.

* [API usage - SessionOptions](api-usage_session-options) - a demonstration of how to configure creation of an `InferenceSession` instance.

* [API usage - `ort.env` flags](api-usage_ort-env-flags) - a demonstration of how to configure a set of global flags.

### Simple Applications

* [OpenAI Whisper](ort-whisper) - demonstrates how to run [whisper tiny.en](https://github.com/openai/whisper) in your browser using [onnxruntime-web](https://github.com/microsoft/onnxruntime) and the browser's audio interfaces.