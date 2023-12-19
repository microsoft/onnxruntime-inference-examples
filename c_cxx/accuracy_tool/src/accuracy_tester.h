// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>
#include "cmd_args.h"

bool RunAccuracyTest(Ort::Env& env, const AppArgs& app_args);
