"""Print a catalog entry for a newly exported pack; review it before adding it."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model_dir', type=Path)
parser.add_argument('--id', required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--family', required=True)
parser.add_argument('--parameters', type=float, required=True)
parser.add_argument('--license', required=True)
parser.add_argument('--license-url', required=True)
args = parser.parse_args()
manifest = json.loads((args.model_dir / 'manifest.json').read_text())
info = json.loads((args.model_dir / 'export-info.json').read_text())
assert manifest.get('browserGraph'), 'Prepare/inspect the browser graph first.'
entry = {'id': args.id, 'label': args.label, 'family': args.family, 'parametersB': args.parameters,
         'repo': info['model'], 'revision': info['revision'], 'dtype': 'q4f16', 'runtime': 'genai',
         'loader': 'jspi', 'source': 'local-export', 'downloadMb': round(manifest['totalBytes'] / 1e6),
         'licenseName': args.license, 'licenseUrl': args.license_url,
         'files': [f['name'] for f in manifest['files']] + ['manifest.json']}
print(json.dumps(entry, indent=2))
