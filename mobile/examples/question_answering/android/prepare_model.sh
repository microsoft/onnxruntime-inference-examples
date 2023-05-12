#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ============================================================
set -e

python -m pip install -r prepare_model.requirements.txt

python ./prepare_model.py
