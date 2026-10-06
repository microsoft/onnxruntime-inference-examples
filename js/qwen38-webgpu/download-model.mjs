// Download immutable public assets; the browser performs all inference locally.
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
const config = JSON.parse(await readFile(new URL('./model.json', import.meta.url), 'utf8'));
const root = resolve(process.env.MODEL_DIR ?? `.models/${config.id}`);
await mkdir(root, { recursive: true });
async function checksum(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
for (const file of config.files) {
  if (!/^[A-Za-z0-9_.-]+$/.test(file.name) || ['.', '..'].includes(file.name)) throw new Error('Unexpected model filename.');
  const destination = resolve(root, file.name);
  try {
    if ((await stat(destination)).size === file.bytes && await checksum(destination) === file.sha256) {
      console.log('Verified existing', file.name); continue;
    }
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const partial = destination + '.part';
  console.log('Downloading', file.name, file.bytes, 'bytes');
  const response = await fetch(`https://huggingface.co/${config.repo}/resolve/${config.revision}/${file.name}?download=true`);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${file.name}`);
  const hash = createHash('sha256'); let bytes = 0;
  const verify = new Transform({ transform(chunk, encoding, callback) { bytes += chunk.length; hash.update(chunk); callback(null, chunk); } });
  try {
    await pipeline(Readable.fromWeb(response.body), verify, createWriteStream(partial));
    if (bytes !== file.bytes || hash.digest('hex') !== file.sha256) throw new Error(`Integrity check failed: ${file.name}`);
    await rename(partial, destination);
  } catch (error) { await rm(partial, { force: true }); throw error; }
}
console.log('Verified model pack ready:', root);
