const modelId = 'Qwen3.8-27B-GenAI-WebGPU-INT4';
let worker, ready = false, busy = false, requestId = 0;
const pending = new Map();
const el = id => document.getElementById(id);
function controls() { el('load').disabled = busy || ready; el('generate').disabled = busy || !ready; el('unload').disabled = !worker; }
function unload() {
  worker?.terminate(); worker = null; ready = false; busy = false;
  for (const { reject } of pending.values()) reject(new Error('Model unloaded.'));
  pending.clear(); controls(); el('status').textContent = 'Model unloaded.';
}
function createWorker() {
  worker = new Worker(new URL('./inference.worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') { el('status').textContent = data.message; return; }
    const task = pending.get(data.requestId); if (!task) return;
    pending.delete(data.requestId);
    data.type === 'error' ? task.reject(new Error(data.error)) : task.resolve(data.value);
  };
  worker.onerror = event => {
    for (const { reject } of pending.values()) reject(new Error(event.message));
    pending.clear();
  };
}
function request(message) {
  return new Promise((resolve, reject) => {
    const id = ++requestId; pending.set(id, { resolve, reject }); worker.postMessage({ ...message, requestId: id });
  });
}
async function load() {
  if (busy || ready) return;
  busy = true; controls(); createWorker();
  try {
    const report = await request({ type: 'load', modelId });
    el('metrics').textContent = JSON.stringify(report, null, 2); ready = true;
    el('status').textContent = 'Model ready. All model operators run on WebGPU; CPU inference fallback is disabled.';
    return report;
  } catch (error) { unload(); el('status').textContent = String(error); throw error; }
  finally { busy = false; controls(); }
}
async function generate(prompt = el('prompt').value) {
  if (busy || !ready) throw new Error('Load the model first.');
  busy = true; controls(); el('status').textContent = 'Generating locally…';
  try {
    const result = await request({ type: 'generate', messages: [{ role: 'user', content: prompt }], maxTokens: 32, temperature: 0, enableThinking: false });
    el('output').textContent = result.text; el('metrics').textContent = JSON.stringify(result, null, 2);
    el('status').textContent = 'Generation complete.'; return result;
  } catch (error) { el('status').textContent = String(error); throw error; }
  finally { busy = false; controls(); }
}
el('load').onclick = () => void load().catch(() => {});
el('generate').onclick = () => void generate().catch(() => {});
el('unload').onclick = unload;
window.qwenDemo = { load, generate, unload };
controls();
