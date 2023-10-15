# Importing ONNX Runtime Web

## Summary

This example is a demonstration of how to import ONNX Runtime Web in your project.

ONNX Runtime Web can be consumed by either using a script tag in HTML or using a modern web app framework with bundler.

## Usage - Using Script tag in HTML

Please use the following HTML snippet to import ONNX Runtime Web:

```html
<!-- import ONNXRuntime Web from CDN (IIFE) -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```

```html
<!-- import ONNXRuntime Web from CDN (ESM) -->
<script type="module">
    import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js";

    // use "ort"
    // ...
</script>
```

See also [Quick Start - Web (using script tag)](../quick-start_onnxruntime-web-script-tag) for an example of using script tag.

## Usage - Using a bundler

Please use the following code snippet to import ONNX Runtime Web:

```js
// Common.js import syntax
const ort = require('onnxruntime-web');
```

```js
// ES Module import syntax
import * as ort from 'onnxruntime-web';
```

See also [Quick Start - Web (using bundler)](../quick-start_onnxruntime-web-bundler) for an example of using bundler.

### Conditional Importing

ONNX Runtime Web supports conditional importing. Please refer to the following table:

| Description | IIFE Filename | Common.js / ES Module import path |
|--------------|-----------|------------|
| Default import. Includes all official released features | ort.min.js | `onnxruntime-web` |
| Experimental. Includes all features | ort.all.min.js | `onnxruntime-web/experimental` |
| Wasm. Includes WebAssembly backend only | ort.wasm.min.js | `onnxruntime-web/wasm` |
| Wasm-core. Includes WebAssembly backend with core features only. Proxy support and Multi-thread support are excluded | ort.wasm-core.min.js | `onnxruntime-web/wasm-core` |
| Webgl. Includes WebGL backend only | ort.webgl.min.js | `onnxruntime-web/webgl` |
| Webgpu. Includes WebGPU backend only | ort.webgpu.min.js | `onnxruntime-web/webgpu` |
| Training. Includes WebAssembly single-threaded only, with training support | ort.training.wasm.min.js | `onnxruntime-web/training` |

Use the following syntax to import different target:
* for script tag usage, replace the URL's file name from the table ("IIFE Filename" column) above:
  ```html
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/<file-name>"></script>
  ```
  ```html
  <script type="module">
    import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/<file-name>";

    // use "ort"
    // ...
  </script>
  ```

* for Common.js module usage, use
  ```js
  const ort = require('<path>');
  ```
  with the path from the table ("Common.js / ES Module import path" column) above
* for ES Module usage, use
  ```js
  import * as ort from '<path>';
  ```
  with the path from the table ("Common.js / ES Module import path" column) above
