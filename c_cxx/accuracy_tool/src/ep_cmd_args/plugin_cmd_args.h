// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>

#include "cmd_args.h"

bool ParseEpPluginArgs(AppArgs& app_args, CmdArgParser& cmd_args, std::string_view prog_name, Ort::Env& env);
