"""Reproduce the tested GenAI INT4/WebGPU export into a self-contained pack."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Set before importing huggingface_hub: ordinary HTTP avoids the observed Xet stall.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ['PYTHONUNBUFFERED'] = '1'
from huggingface_hub import model_info, snapshot_download

TOOL = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True, help='Official Hugging Face checkpoint ID')
parser.add_argument('--revision', help='Resolve and record HEAD if omitted')
parser.add_argument('--output', type=Path, required=True, help='A new/empty model-pack directory')
parser.add_argument('--cache', type=Path, default=TOOL / '.work/cache')
parser.add_argument('--bounded', action='store_true', help='Use the validated per-weight serializer for large models')
parser.add_argument('--skip-native', action='store_true', help='Skip offline CPU sanity validation')
parser.add_argument('--extra-option', action='append', default=[], help='Additional builder key=value, e.g. block_size=32')
args = parser.parse_args()
root = args.output.resolve()
if root.exists() and any(root.iterdir()):
    parser.error('Output must be empty. Use a new directory or remove the failed output before retrying.')
root.mkdir(parents=True, exist_ok=True)
notes = root.parent / (root.name + '.reports')
notes.mkdir(parents=True, exist_ok=True)
info = model_info(args.model, revision=args.revision)
snapshot = snapshot_download(args.model, revision=info.sha, cache_dir=str(args.cache.resolve()),
                             allow_patterns=['*.json', '*.jinja', '*.safetensors', '*.txt'], max_workers=8)
metadata = {'model': args.model, 'revision': info.sha, 'precision': 'int4', 'provider': 'webgpu',
            'boundedSerializer': args.bounded, 'extraOptions': args.extra_option}
(root / 'export-info.json').write_text(json.dumps(metadata, indent=2) + '\n')

def run(label, command):
    print(f'\n{label}', flush=True)
    with (notes / (label + '.log')).open('w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=notes)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
    if code: raise SystemExit(f'{label} failed ({code}); see {notes}')

builder = [sys.executable, str(TOOL / 'scripts/streaming_export.py')] if args.bounded else [sys.executable, '-m', 'onnxruntime_genai.models.builder']
# MatMul/Gather also quantizes untied input embeddings, essential for the 27B pack.
run('export', builder + ['-m', args.model, '-i', snapshot, '-o', str(root), '-p', 'int4', '-e', 'webgpu',
    '-c', str(args.cache.resolve()), '--extra_options', 'exclude_mtp=true', 'state_window=0',
    'op_types_to_quantize=MatMul/Gather', *args.extra_option])
if not args.skip_native:
    run('native-validation', [sys.executable, str(TOOL / 'scripts/validate.py'), str(root)])
# Keep small exports' original external-data layout; split oversized baseline files.
if any(p.stat().st_size >= 2**31 for p in root.glob('*.onnx.data')):
    run('sharding', [sys.executable, str(TOOL / 'scripts/shard_model.py'), str(root)])
run('browser-adaptation', [sys.executable, str(TOOL / 'scripts/prepare_browser.py'), str(root)])
run('inspection', [sys.executable, str(TOOL / 'scripts/inspect_model.py'), str(root)])
print(f'\nReady for browser validation: {root}\nReports: {notes}')
