# API usage - InferenceSession

## Summary

This example is a demonstration of basic usage of `InferenceSession`.

- `inference-session-create.js`: In this example, we create `InferenceSession` in different ways.
- `inference-session-properties.js`: In this example, we get input/output names from an `InferenceSession` object.
- `inference-session-run.js`: In this example, we run the model inferencing in different ways.

For more information about `SessionOptions` and `RunOptions`, please refer to other examples.

See also [`InferenceSession.create`](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSessionFactory.html#create) and [`InferenceSession` interface](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.html) in API reference document.

## Usage

```sh
npm install
node ./inference-session-create.js
node ./inference-session-properties.js
node ./inference-session-run.js
```
