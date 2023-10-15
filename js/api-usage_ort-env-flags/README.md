# API usage - `ort.env` flags

## Summary

This example is a demonstration of how to configure global flags by `ort.env`.

Following are some example code snippets:

```js
// enable DEBUG flag
ort.env.debug = true;

// set global logging level
ort.env.logLevel = 'info';
```

See also [`Env` interface](https://onnxruntime.ai/docs/api/js/interfaces/Env.html) in API reference document.

### WebAssembly flags (ONNX Runtime Web)

WebAssembly flags are used to customize behaviors of WebAssembly execution provider.

Following are some example code snippets:

```js
// set up-to-2-threads for multi-thread execution for WebAssembly
// may fallback to single-thread if multi-thread is not available in the current browser
ort.env.wasm.numThreads = 2;

// force single-thread for WebAssembly
ort.env.wasm.numThreads = 1;

// enable worker-proxy feature for WebAssembly
// this feature allows model inferencing to run in a web worker asynchronously.
ort.env.wasm.proxy = true;

// override path of wasm files - using a prefix
// in this example, ONNX Runtime Web will try to load file from https://example.com/my-example/ort-wasm*.wasm 
ort.env.wasm.wasmPaths = 'https://example.com/my-example/';

// override path of wasm files - for each file
ort.env.wasm.wasmPaths = {
    'ort-wasm.wasm': 'https://example.com/my-example/ort-wasm.wasm',
    'ort-wasm-simd.wasm': 'https://example.com/my-example/ort-wasm-simd.wasm',
    'ort-wasm-threaded.wasm': 'https://example.com/my-example/ort-wasm-threaded.wasm',
    'ort-wasm-simd-threaded.wasm': 'https://example.com/my-example/ort-wasm-simd-threaded.wasm'
};
```

See also [WebAssembly flags](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html) in API reference document.

### WebGL flags (ONNX Runtime Web)

WebGL flags are used to customize behaviors of WebGL execution provider.

Following are some example code snippets:

```js
// enable packed texture. This helps to improve inference performance for some models
ort.env.webgl.pack = true;
```

See also [WebGL flags](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGLFlags.html) in API reference document.

### WebGPU flags (ONNX Runtime Web)
WebGPU flags are used to customize behaviors of WebGPU execution provider.

Following are some example code snippets:

```js
// enable WebGPU profiling.
ort.env.webgpu.profilingMode = 'default';

// get the gpu device object.
const device = ort.env.webgpu.device;
```

See also [WebGPU flags](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGpuFlags.html) in API reference document.

### SessionOptions vs. ort.env

Both `SessionOptions` and `ort.env` allow to specify configurations for inferencing behaviors. The biggest difference of them is: `SessionOptions` is set for one inference session instance, while `ort.env` is set global.

See also [API usage - `SessionOptions`](../api-usage_session-options) for an example of using `SessionOptions`.

## Usage

The code snippets demonstrated above cannot run standalone. Put the code into one of the "Quick Start" examples and try it out.
