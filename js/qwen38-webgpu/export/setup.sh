#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-export.lock
