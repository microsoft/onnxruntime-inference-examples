# Quick Start - Web (using script tag)

This example is a demonstration of basic usage of ONNX Runtime Web, using script tag in HTML.

Using a `<script>` tag is a simple way to consume a published JavaScript library. See also [Quick Start - Web (using bundler)](../quick-start_onnxruntime-web-bundler) for an example of using bundler.

In this example, we load onnxruntime, create an inference session with a simple model, feed input, get output as result and write it to the HTML page. All functions are called in their basic form.

## Usage

1. use NPM package `light-server` to serve the current folder at http://localhost:8080/
   ```sh
   npx light-server -s . -p 8080
   ```

2. open your browser and navigate to the URL.