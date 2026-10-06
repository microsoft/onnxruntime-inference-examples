# Qwen3.8-27B in the browser with ONNX Runtime WebGPU

A minimal text-generation example using a public **Qwen3.8-27B INT4** model pack, ONNX Runtime Web's **JSPI/Blob loader**, and hardware WebGPU. It runs independently of Clinstruct. All model operators execute locally on GPU, with CPU inference fallback disabled; JavaScript handles tokenization, sequence metadata and greedy sampling.

**Weights: approximately 15.26 GB. Tested on a 48 GiB Apple Silicon Mac, Chrome 154, Metal 3 WebGPU.** Earlier integrated tests measured about 5–6 tokens/s; the standalone 32-token smoke test measured about 10 tokens/s. See [validation.json](validation.json) for exact timings. Other hardware and long contexts are unvalidated. This is a runnable engineering example, not a quality benchmark or an official Microsoft/Alibaba release.

## Run

Install Node.js ≥20.19. Use current Chrome or Edge with hardware WebGPU, `shader-f16` and WebAssembly JSPI, in an ordinary browser profile.

```sh
git clone https://github.com/RuurdKuiper/qwen38-27b-onnx-webgpu.git
cd qwen38-27b-onnx-webgpu
npm ci
npm run download:model
npm run dev -- --port 5174
```

Open `http://127.0.0.1:5174`, click **Load model**, then generate. The download script gets the [public Hugging Face pack](https://huggingface.co/Ruurd/Qwen3.8-27B-ONNX-WebGPU-INT4) at the immutable commit in `model.json` and verifies file sizes/SHA-256 checksums. Re-running it preserves verified completed shards. The local server only serves static assets; it runs no inference. Prompts remain in the browser.

If the model is already installed elsewhere:

```sh
MODEL_DIR=/absolute/path/to/model-pack npm run dev -- --port 5174
```

`npm run build` bundles the application and pinned runtime, excluding weights. To deploy that build, serve the model pack separately at `/models/onnx/Qwen3.8-27B-GenAI-WebGPU-INT4/` on the same secure origin. Never bundle the weights into the application ZIP.

## Model and generation contract

This export requires the complete graph/state contract, not only `input_ids → logits`. The worker inspects `manifest.json` and `genai_config.json` and retains **128 state tensors on GPU**: 32 KV, 48 convolution and 48 recurrent. Each prompt starts fresh state; decoding retains state between tokens. Generation is batch-one contiguous unpadded text, 32 new tokens maximum in this UI, temperature zero, and thinking disabled. Qwen's three-axis position IDs are supplied explicitly.

The native graph is preserved. `model.browser.onnx` replaces six known attention-mask metadata operations with host `seqlens_k` and `total_sequence_length` inputs to avoid CPU placement; weights and model computation are unchanged relative to the native export. INT4 block size is 32; both MatMul and Gather (untied embeddings) are quantized. Vision, MTP, paged attention, speculative decoding and CUDA-specific GatedDeltaNet are excluded.

The example reports model bytes, download/load time, first-token latency, decoding speed, available JS heap statistics, GPU adapter/features and runtime warnings. Heap statistics exclude GPU allocations. The loader uses `onnxruntime-web@1.31.0-dev.20260914-8d85527a0`; Transformers.js 4.3.0 supplies tokenization only. The tested 48 GiB machine is not a measured minimum requirement. Large Blob loading failed in an incognito test.

## Reproduce and test

`export/` contains the pinned Python exporter environment and the official-builder wrapper. The memory-bounded serializer is an optional adapter around the official DEFAULT quantizer; it handles large intermediate weights and emits ≤1 GiB shards without changing quantization arithmetic. See `export/README.md` for commands and architecture limitations.

With Vite running, `npm run test:browser` launches installed Chrome in a fresh ordinary profile, blocks external requests, checks arithmetic, ≥20 consecutive generated tokens, strict GPU execution and all 128 retained state tensors, and writes `results/browser.json`. For another origin set `DEMO_URL`.

## Attribution

Qwen3.8-27B is copyright 2026 Alibaba Cloud under Apache 2.0. The model repository includes its `LICENSE` and `NOTICE`. This browser example and conversion tooling are copyright 2026 Ruurd Kuiper under Apache 2.0; dependencies retain their own licenses. The worker originates from the tested Clinstruct integration and is reproduced here so other applications can use the same generation path.

Upstream projects: [ONNX Runtime](https://github.com/microsoft/onnxruntime), [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai), [Transformers.js](https://github.com/huggingface/transformers.js), [original checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B).
