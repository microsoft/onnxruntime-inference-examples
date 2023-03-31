#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ============================================================
set -e

python3 -m pip install -r prepare_model.requirements.txt

# Install ort-extensions on Windows
# pip3 install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-extensions

# Install ort-extensions on Linux/MacOS
python3 -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git

python3 ./prepare_model.py
