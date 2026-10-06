# Reproduce the conversion

Run from this repository's root. Requires `uv` and space for the ~55.56 GB original checkpoint, ~15.26 GB output and temporary files. The recorded export used a 48 GiB Mac. These dependencies pin the official builder/runtime tested on 6 October 2026.

```sh
bash export/setup.sh
export/.venv/bin/python export/export.py \
  --model Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --bounded \
  --output .models/Qwen3.8-27B-GenAI-WebGPU-INT4
```

The command downloads the pinned checkpoint, exports INT4 with WebGPU, runs offline native GenAI sanity checks, adapts the known attention-mask metadata subgraph and emits the browser manifest. The native CPU check does not serve inference to the browser. `--skip-native` skips it.

`--bounded` retains the official builder and DEFAULT weight-only quantizer, but serializes/quantizes one weight at a time to avoid retaining the whole FP16 intermediate model during serialization. The original baseline was killed while saving. The adapter copies independent protobuf descriptors, handles >2 GiB intermediate weights through temporary external files, quantizes both MatMul/Gather, and emits ≤1 GiB tensor-aligned shards. It matched all 491 initializer shapes/dtypes/hashes and graph nodes in a 2B cross-check against the ordinary exporter. This internal adapter is tied to the pinned builder version.

Defaults are INT4 block size 32, decoder only, `exclude_mtp=true`, `state_window=0`, ordinary non-paged linear attention and MatMul/Gather quantization. No CUDA kernels, vision, MTP, speculative decoding or paged attention. Output must be empty; use a new directory for a retry. Reports are beside the output in `model-name.reports/`.

The original/native graph is retained. The browser adapter removes exactly six known mask metadata nodes and rejects unfamiliar subgraphs. Different architectures need their own input/output/state and position-ID inspection and WebGPU operator validation; the current worker implements the Qwen graph contract. Optional offline graph/state equivalence check:

```sh
export/.venv/bin/python export/scripts/check_adaptation.py \
  .models/Qwen3.8-27B-GenAI-WebGPU-INT4
```

Serve the example and run `npm run test:browser` to test the result. The model's configured context length is not a measured browser capacity. Reproducible numerical serialization does not establish BF16 quality parity or compatibility with arbitrary GPUs.
