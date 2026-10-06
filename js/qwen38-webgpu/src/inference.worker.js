import * as ort from 'onnxruntime-web/jspi';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jspi.wasm?url';
import wasmModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jspi.mjs?url';
import { AutoTokenizer, env } from '@huggingface/transformers';

// Batch-one text generation for the official GenAI export. All model operators
// run on WebGPU; only tokenization, sequence metadata and sampling run in JS.
let session, tokenizer, chatTemplate, config, graph, bindings, device, modelRoot;
let activeRequestId = 0;
const report = { runtimeVersion: ort.env.versions.web, executionProviders: ['webgpu'], cpuFallbackDisabled: true, messages: [] };
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
function log(level, ...args) {
  const text = args.map(String).join(' ');
  report.messages.push({ level, text });
  if (level === 'info') self.postMessage({ type: 'progress', requestId: activeRequestId, message: text });
  else console[level](...args);
}
function heap() { return performance.memory ? { usedJSHeapSize: performance.memory.usedJSHeapSize } : null; }
async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}
const modelUrl = file => `${modelRoot}/${file}`;
// Match layer-specific names using the generated config, rather than guessing
// that every decoder layer has an ordinary key/value cache.
function cacheBindings() {
  const pairs = [
    ['past_key_names', 'present_key_names'], ['past_value_names', 'present_value_names'],
    ['past_conv_names', 'present_conv_names'], ['past_recurrent_names', 'present_recurrent_names'],
  ];
  const result = [];
  const decoder = config.model.decoder;
  for (const [past, present] of pairs) {
    if (!decoder.inputs[past]) continue;
    for (let layer = 0; layer < decoder.num_hidden_layers; layer++) {
      const input = decoder.inputs[past].replace('%d', layer);
      if (!session.inputNames.includes(input)) continue;
      const output = decoder.outputs[present].replace('%d', layer);
      if (!session.outputNames.includes(output)) throw new Error(`Missing output for ${input}: ${output}`);
      result.push({ input, output, metadata: graph.inputs.find((v) => v.name === input) });
    }
  }
  const supported = new Set([...result.map((v) => v.input), decoder.inputs.input_ids,
    decoder.inputs.attention_mask, decoder.inputs.position_ids, 'seqlens_k', 'total_sequence_length']);
  for (const name of session.inputNames) {
    if (!supported.has(name)) throw new Error(`Unimplemented graph input: ${name}`);
  }
  return result;
}

async function loadModel(modelId) {
  modelRoot = `/models/onnx/${encodeURIComponent(modelId)}`;
  if (session) return;
  const loadStart = performance.now();
  if (typeof WebAssembly.Suspending !== 'function') throw new Error('This large model requires WebAssembly JSPI. Use current Chrome or Edge.');
  report.status = 'loading';
  log('info', 'Preparing WebGPU device…');
  if (!navigator.gpu) throw new Error('WebGPU unavailable. Use current Chrome or Edge on localhost.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.info.isFallbackAdapter) throw new Error('A hardware WebGPU adapter is required.');
  if (!adapter.features.has('shader-f16')) throw new Error('This FP16/INT4 export requires shader-f16.');
  const requiredFeatures = ['shader-f16'];
  if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups');
  const requiredLimits = {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
  device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  device.addEventListener('uncapturederror', (e) => log('error', 'WebGPU:', e.error.message));
  device.lost.then((info) => log('error', 'WebGPU device lost:', info.reason, info.message));
  report.gpu = { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    device: adapter.info.device, description: adapter.info.description,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
    adapterFeatures: [...adapter.features], deviceFeatures: [...device.features],
    deviceLimits: requiredLimits };
  ort.env.webgpu.device = device;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModuleUrl };
  ort.env.logLevel = 'warning';

  const manifest = await json(modelUrl('manifest.json'));
  config = await json(modelUrl('genai_config.json'));
  const graphFile = manifest.browserGraph;
  if (!graphFile) throw new Error("Browser-adapted graph missing. Run prepare_browser.py and inspect_model.py.");
  graph = manifest.graphs.find((g) => g.file === graphFile);
  if (!graph) throw new Error('Decoder graph missing from manifest. Run scripts/inspect_model.py.');
  if (manifest.graphs.some((g) => g.file.includes('mtp')) || config.model.mtp) throw new Error('Export only the decoder with exclude_mtp=true.');
  report.graphFile = graphFile;
  report.browserAdaptation = graphFile !== config.model.decoder.filename ? manifest.browserAdaptation : null;
  report.exportBytes = manifest.totalBytes;
  report.files = manifest.files;
  const downloadStart = performance.now();
  const externalData = [];
  let payloadBytes = 0;
  // Fetch sequentially; retain exact external locations from the ONNX graph.
  for (const path of graph.externalData) {
    log('info', 'Downloading', path);
    const response = await fetch(modelUrl(path));
    if (!response.ok) throw new Error(`Weight download failed: ${response.status}`);
    const data = await response.blob();
    const expectedBytes = manifest.files.find(file => file.name === path)?.bytes;
    if (expectedBytes !== undefined && data.size !== expectedBytes) throw new Error(`Incomplete weight file: ${path}`);
    payloadBytes += data.size;
    externalData.push({ path, data });
  }
  const response = await fetch(modelUrl(graph.file));
  if (!response.ok) throw new Error(`Graph download failed: ${response.status}`);
  const modelBytes = new Uint8Array(await response.arrayBuffer());
  payloadBytes += modelBytes.byteLength;
  report.graphAndWeightDownloadBytes = payloadBytes;
  report.downloadMs = performance.now() - downloadStart;
  const preferredOutputLocation = Object.fromEntries(graph.outputs.map((v) =>
    [v.name, v.name === config.model.decoder.outputs.logits ? 'cpu' : 'gpu-buffer']));
  log('info', 'Creating WebGPU inference session…');
  const sessionStart = performance.now();
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ['webgpu'], externalData, preferredOutputLocation,
    freeDimensionOverrides: { batch_size: 1 },
    // Explicitly reject fallback to WASM CPU inference.
    extra: { session: { disable_cpu_ep_fallback: '1' } },
    logSeverityLevel: 2,
  });
  report.sessionCreateMs = performance.now() - sessionStart;
  bindings = cacheBindings();
  report.cacheTensorCount = bindings.length;
  report.graphInputs = session.inputNames;
  report.graphOutputs = session.outputNames;

  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.useBrowserCache = false;
  env.localModelPath = '/models/onnx/';
  tokenizer = await AutoTokenizer.from_pretrained(modelId, { local_files_only: true });
  const templateResponse = await fetch(modelUrl('chat_template.jinja'));
  if (!templateResponse.ok) throw new Error('Exported chat_template.jinja is missing.');
  chatTemplate = await templateResponse.text();
  report.loadMs = performance.now() - loadStart;
  report.modelResources = performance.getEntriesByType('resource')
    .filter((r) => new URL(r.name).pathname.startsWith(modelRoot + '/'))
    .map((r) => ({ name: new URL(r.name).pathname, decodedBodySize: r.decodedBodySize, transferSize: r.transferSize }));
  report.downloadedModelResourceBytes = report.modelResources.reduce((n, r) => n + r.decodedBodySize, 0);
  report.heapAfterLoad = heap();
  report.status = 'ready';
}

function typedTensor(type, values, dims) {
  const constructors = { int64: BigInt64Array, int32: Int32Array, float16: Uint16Array,
    float: Float32Array, float32: Float32Array };
  const Constructor = constructors[type];
  if (!Constructor) throw new Error(`Unsupported graph tensor type: ${type}`);
  const data = typeof values === 'number' ? new Constructor(values) :
    Constructor.from(values, type === 'int64' ? BigInt : (v) => v);
  return new ort.Tensor(type === 'float' ? 'float32' : type, data, dims);
}
function initialState(metadata) {
  const symbols = { batch_size: 1, past_sequence_length: 0, kv_cache_dim: config.model.decoder.head_size };
  const dims = metadata.dims.map((v) => {
    if (typeof v === 'number') return v;
    if (v in symbols) return symbols[v];
    throw new Error(`Unresolved state dimension: ${v}`);
  });
  return typedTensor(metadata.type, dims.reduce((a, b) => a * b, 1), dims);
}
function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (exponent ? 2 ** (exponent - 15) * (1 + fraction / 1024) : 2 ** -14 * fraction / 1024);
}
async function sample(logits, temperature = 0) {
  const data = await logits.getData();
  const vocab = logits.dims.at(-1);
  const offset = data.length - vocab;
  let best = -Infinity, token = -1;
  const scores = new Float32Array(vocab);
  for (let i = 0; i < vocab; i++) {
    const value = logits.type === 'float16' && data instanceof Uint16Array ? halfToFloat(data[offset + i]) : data[offset + i];
    scores[i] = value;
    if (Number.isNaN(value)) throw new Error('NaN in logits.');
    if (value > best) { best = value; token = i; }
  }
  if (token < 0 || !Number.isFinite(best)) throw new Error('Invalid logits.');
  if (temperature > 0) {
    let total = 0;
    for (let i = 0; i < vocab; i++) total += scores[i] = Math.exp((scores[i] - best) / temperature);
    let target = Math.random() * total;
    for (let i = 0; i < vocab; i++) if ((target -= scores[i]) <= 0) return i;
  }
  return token;
}

async function generate({ messages, maxTokens, temperature, enableThinking }) {
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 1024) throw new Error("maxTokens must be between 1 and 1024.");
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) throw new Error("Invalid temperature.");
  if (!session || !tokenizer) throw new Error('Load the model first.');
  const inputIds = tokenizer.apply_chat_template(messages, {
    tokenize: true, return_tensor: false, return_dict: false,
    add_generation_prompt: true, enable_thinking: enableThinking, chat_template: chatTemplate,
  });
  if (!Array.isArray(inputIds) || !inputIds.every(Number.isInteger)) throw new Error('Unexpected tokenizer output.');
  if (inputIds.length + maxTokens > config.model.context_length) throw new Error("Prompt exceeds model context length.");
  const decoder = config.model.decoder;
  const byName = Object.fromEntries(graph.inputs.map((v) => [v.name, v]));
  let state = Object.fromEntries(bindings.map((v) => [v.input, initialState(v.metadata)]));
  let tokens = inputIds, pastLength = 0, firstTokenAt;
  const generated = [], steps = [];
  const start = performance.now();
  const eos = new Set([config.model.eos_token_id].flat());
  try {
    for (let i = 0; i < maxTokens; i++) {
      const feeds = { ...state };
      const ephemeral = [];
      const put = (name, values, dims) => {
        if (!name || !session.inputNames.includes(name)) return;
        const tensor = typedTensor(byName[name].type, values, dims);
        feeds[name] = tensor; ephemeral.push(tensor);
      };
      put(decoder.inputs.input_ids, tokens, [1, tokens.length]);
      put(decoder.inputs.attention_mask, Array(pastLength + tokens.length).fill(1), [1, pastLength + tokens.length]);
      put('seqlens_k', [pastLength + tokens.length - 1], [1]);
      put('total_sequence_length', [pastLength + tokens.length], []);
      const positions = tokens.map((_, j) => pastLength + j);
      put(decoder.inputs.position_ids, [...positions, ...positions, ...positions], [3, 1, tokens.length]);
      let outputs;
      try { outputs = await session.run(feeds); }
      finally { for (const tensor of ephemeral) tensor.dispose(); }
      try {
        const token = await sample(outputs[decoder.outputs.logits], temperature);
        if (firstTokenAt === undefined) firstTokenAt = performance.now();
        generated.push(token);
        const nextState = {};
        for (const binding of bindings) {
          const value = outputs[binding.output];
          if (!value) throw new Error(`Missing state output: ${binding.output}`);
          if (value.location !== 'gpu-buffer') throw new Error(`State left GPU: ${binding.output} (${value.location})`);
          nextState[binding.input] = value;
        }
        steps.push({ step: i, inputTokens: tokens.length, pastLength, stateTensorsRetained: bindings.length,
          allStateOnGPU: true, kvLength: nextState[bindings.find((v) => v.input.startsWith('past_key_values.')).input].dims[2] });
        for (const tensor of Object.values(state)) tensor.dispose();
        state = nextState;
        pastLength += tokens.length;
        tokens = [token];
        if (eos.has(token)) break;
      } finally {
        const retained = new Set(Object.values(state));
        for (const tensor of Object.values(outputs)) if (!retained.has(tensor)) tensor.dispose();
      }
      await tick();
    }
    const end = performance.now();
    const result = { inputIds, generatedIds: generated,
      promptTokens: inputIds.length, completionTokens: generated.length,
      text: tokenizer.decode(generated, { skip_special_tokens: true }),
      generatedTokens: generated.length, firstTokenMs: firstTokenAt - start,
      tokensPerSecondAfterFirst: generated.length > 1 ? (generated.length - 1) * 1000 / (end - firstTokenAt) : null,
      elapsedMs: end - start, heap: heap(), steps };
    result.decodeTokensPerSecond = result.tokensPerSecondAfterFirst;
    report.lastGeneration = result;
    return result;
  } finally { for (const tensor of Object.values(state)) tensor.dispose(); }
}

// Serialize requests so cache/session ownership cannot overlap.
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    activeRequestId = data.requestId;
    try {
      let value;
      if (data.type === 'load') {
        await loadModel(data.modelId);
        value = report;
      } else if (data.type === 'generate') {
        value = await generate(data);
      } else throw new Error('Unknown GenAI worker request.');
      self.postMessage({ type: 'result', requestId: data.requestId, value });
    } catch (error) {
      self.postMessage({ type: 'error', requestId: data.requestId, error: String(error) });
    }
  });
};
