"""Repack an official single-file export into <=1 GiB tensor-aligned shards.

Numerical tensors and graph nodes are unchanged. Verify every copied byte before
publishing the new graph and removing the now-unreferenced original data file.
"""
import hashlib
import json
import sys
import os
from pathlib import Path
import onnx
from onnx import external_data_helper

root = Path(sys.argv[1]).resolve()
notes = root.parent / (root.name + ".reports")
notes.mkdir(parents=True, exist_ok=True)
model_path = root / 'model.onnx'
model = onnx.load(model_path, load_external_data=False)
locations = {entry.value for tensor in model.graph.initializer for entry in tensor.external_data if entry.key == 'location'}
assert locations == {'model.onnx.data'}, 'Expected an unsharded export; do not run this on an already sharded model.'
limit = 1024**3
original_files = set()
shards = []
writer = None
index = -1
checks = []
try:
    for tensor in model.graph.initializer:
        ext = {entry.key: entry.value for entry in tensor.external_data}
        if 'location' not in ext:
            continue
        original_files.add(ext['location'])
        length = int(ext['length'])
        if length > limit:
            raise ValueError(f'Tensor exceeds shard limit: {tensor.name}, {length}')
        padding = (-writer.tell()) % 64 if writer else 0
        if writer is None or writer.tell() + padding + length > limit:
            if writer: writer.close()
            index += 1
            location = f'model.onnx.data_{index:03d}'
            shards.append(location)
            writer = (root / location).open('wb')
            padding = 0
        writer.write(b'\0' * padding)
        offset = writer.tell()
        digest = hashlib.sha256()
        with (root / ext['location']).open('rb') as source:
            source.seek(int(ext.get('offset', 0)))
            remaining = length
            while remaining:
                data = source.read(min(remaining, 8 * 1024 * 1024))
                if not data: raise EOFError(tensor.name)
                writer.write(data); digest.update(data); remaining -= len(data)
        tensor.raw_data = b''
        external_data_helper.set_external_data(tensor, location=location, offset=offset, length=length)
        tensor.ClearField('raw_data')
        tensor.data_location = onnx.TensorProto.EXTERNAL
        checks.append({'tensor': tensor.name, 'file': location, 'offset': offset,
                       'bytes': length, 'sha256': digest.hexdigest()})
finally:
    if writer: writer.close()
for check in checks:
    digest = hashlib.sha256()
    with (root / check['file']).open('rb') as source:
        source.seek(check['offset']); remaining = check['bytes']
        while remaining:
            data = source.read(min(remaining, 8 * 1024 * 1024)); assert data
            digest.update(data); remaining -= len(data)
    assert digest.hexdigest() == check['sha256'], check['tensor']
onnx.save_model(model, root / 'model.sharded.part.onnx')
(root / 'model.sharded.part.onnx').replace(model_path)
for name in original_files:
    if name not in shards: (root / name).unlink()
report = {'tensorBytesVerified': sum(check['bytes'] for check in checks), 'tensors': checks,
          'shards': [{'file': name, 'bytes': (root / name).stat().st_size} for name in shards]}
(notes / 'sharding.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'verifiedTensorCount': len(checks), 'shards': report['shards']}, indent=2))
