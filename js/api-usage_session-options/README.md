# API usage - SessionOptions

## Summary

This example is a demonstration of how to configure an `InferenceSession` instance using `SessionOptions`.

A `SessionOptions` is an object with properties to instruct the creation of an `InferenceSession` instance. See [type declaration](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/inference-session.ts) for schema definition. `SessionOptions` is passed to `InferenceSession.create()` as the last parameter:

```js
const mySession = await InferenceSession.create(..., mySessionOptions);
```

### Execution providers

An [execution provider](https://onnxruntime.ai/docs/reference/execution-providers/) (EP) defines how operators get resolved to specific kernel implementation. Following is a table of supported EP in different environments:

| EP name | Hardware          | available in                      |
| ------- | ----------------- | --------------------------------- |
| `cpu`   | CPU (default CPU) | onnxruntime-node                  |
| `cuda`  | GPU (NVIDIA CUDA) | onnxruntime-node                  |
| `dml`   | GPU (Direct ML)   | onnxruntime-node (Windows)        |
| `wasm`  | CPU (WebAssembly) | onnxruntime-web, onnxruntime-node |
| `webgl` | GPU (WebGL)       | onnxruntime-web                   |
| `webgpu`| GPU (WebGPU)      | onnxruntime-web                   |

Execution provider is specified by `sessionOptions.executionProviders`. Multiple EPs can be specified and the first available one will be used.

Following are some example code snippets:

```js
// [Node.js binding example] Use CPU EP.
const sessionOption = { executionProviders: ['cpu'] };
```

```js
// [Node.js binding example] Use CUDA EP.
const sessionOption = { executionProviders: ['cuda'] };
```

```js
// [Node.js binding example] Use CUDA EP, specifying device ID.
const sessionOption = {
  executionProviders: [
    {
      name: 'cuda',
      deviceId: 0
    }
  ]
};
```

```js
// [Node.js binding example] Try multiple EPs using an execution provider list.
// The first successfully initialized one will be used. Use CUDA EP if it is available, otherwise fallback to CPU EP.
const sessionOption = { executionProviders: ['cuda', 'cpu'] };
```

```js
// [ONNX Runtime Web example] Use WebAssembly (CPU) EP.
const sessionOption = { executionProviders: ['wasm'] };
```

```js
// [ONNX Runtime Web example] Use WebGL EP.
const sessionOption = { executionProviders: ['webgl'] };
```

```js
// [ONNX Runtime Web example] Use WebGPU EP.
const sessionOption = { executionProviders: ['webgpu'] };

// [ONNX Runtime Web example] Use WebGPU EP with extra config.
const sessionOption2 = { executionProviders: [{
  name: 'webgpu',
  preferredLayout: 'NCHW'
}] }
```

### other common options

There are also some other options available for all EPs.

Following are some example code snippets:

```js
// [Node.js binding example] Use CPU EP with single-thread and enable verbose logging.
const sessionOption = {
  executionProviders: ['cpu'],
  interOpNumThreads: 1,
  intraOpNumThreads: 1,
  logSeverityLevel: 0
};
```

```js
// [ONNX Runtime Web example] Use WebAssembly EP and enable profiling.
const sessionOptions = {
  executionProviders: ['wasm'],
  enableProfiling: true
};
```

See also [`SessionOptions` interface](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html) in API reference document.

### SessionOptions vs. ort.env

Both `SessionOptions` and `ort.env` allow to specify configurations for inferencing behaviors. The biggest difference of them is: `SessionOptions` is set for one inference session instance, while `ort.env` is set global.

See also [API usage - `ort.env` flags](../api-usage_ort-env-flags) for an example of using `ort.env`.

## Usage

The code snippets demonstrated above cannot run standalone. Put the code into one of the "Quick Start" examples and try it out.
