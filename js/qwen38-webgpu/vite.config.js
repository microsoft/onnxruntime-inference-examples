import { createReadStream, statSync } from 'node:fs';
import { resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
const id = 'Qwen3.8-27B-GenAI-WebGPU-INT4';
const modelDir = resolve(process.env.MODEL_DIR ?? fileURLToPath(new URL(`./.models/${id}/`, import.meta.url)));
export default defineConfig({
  plugins: [{ name: 'local-model-files', configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const prefix = `/models/onnx/${id}/`, path = new URL(req.url, 'http://localhost').pathname;
      if (!path.startsWith(prefix) || !['GET', 'HEAD'].includes(req.method)) return next();
      let file;
      try { file = resolve(modelDir, decodeURIComponent(path.slice(prefix.length))); }
      catch { res.writeHead(400).end(); return; }
      if (!file.startsWith(modelDir + sep)) { res.writeHead(403).end(); return; }
      let size;
      try { const stat = statSync(file); if (!stat.isFile()) throw new Error(); size = stat.size; }
      catch { res.writeHead(404).end('Model file missing; see README.md'); return; }
      res.writeHead(200, { 'Content-Length': size, 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store' });
      if (req.method === 'HEAD') { res.end(); return; }
      const stream = createReadStream(file, { highWaterMark: 4 * 1024 * 1024 });
      stream.on('error', () => res.destroy()); res.on('close', () => stream.destroy()); stream.pipe(res);
    });
  } }],
  build: { target: 'es2022' },
});
