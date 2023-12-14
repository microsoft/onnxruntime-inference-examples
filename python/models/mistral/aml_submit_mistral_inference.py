#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Microsoft Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
from pathlib import Path
import json
import os

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, BuildContext 
from azure.identity import AzureCliCredential

# run test on automode workspace
ws_config = json.load(open("ws_config.json"))
subscription_id = ws_config["subscription_id"]
resource_group = ws_config["resource_group"]
workspace_name = ws_config["workspace_name"]
compute = ws_config["compute"]
nproc_per_node = ws_config["nproc_per_node"]

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", default="MISTRAL-7B-ORT-CLM-Stage2-Experiment", help="Experiment name for AML Workspace")

    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = get_args(raw_args)

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    root_dir = Path(__file__).resolve().parent
    environment_dir = root_dir / "environment"
    code_dir = root_dir / "inference-code"

    model = "mistralai/Mistral-7B-v0.1"

    # https://huggingface.co/datasets/dair-ai/emotion
    dataset_name = "databricks/databricks-dolly-15k"

    inference_job = command(
        code=code_dir,  # local path where the code is stored
        command=f"bash inference_setup.sh",
        environment=Environment(build=BuildContext(path=environment_dir)),
        experiment_name="MISTRAL-7B-Inference-Experiment",
        compute=compute,
        display_name=model.replace(
            "mistral-ai",
            f"Inference-benchmark"
        ),
        description=f"Mistral AI 7B Inference Benchmark",
        tags={"model": model,
              "dataset_name": dataset_name},
        shm_size="16g"
    )
    
    print("submitting Inference job for " + model)
    inference_returned_job = ml_client.create_or_update(inference_job)
    print("submitted job")

    inference_aml_url = inference_returned_job.studio_url
    print("Inference Benchmark job link:", inference_aml_url)


if __name__ == "__main__":
    main()
