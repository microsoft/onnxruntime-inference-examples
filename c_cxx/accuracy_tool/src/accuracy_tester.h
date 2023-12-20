// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>

#include "cmd_args.h"

/// <summary>
/// The main entry point of the application. Measures the accuracy of a set of models (and input datasets)
/// on a given execution provider. The accuracy is computed by comparing with the expected results, which are either
/// loaded from file or attained by running the model with the CPU execution provider.
/// </summary>
/// <param name="env">The ONNX runtime environment for the entire process</param>
/// <param name="app_args">The application's input arguments</param>
/// <returns></returns>
bool RunAccuracyTest(Ort::Env& env, const AppArgs& app_args);
