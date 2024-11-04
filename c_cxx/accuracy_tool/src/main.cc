// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <onnxruntime_cxx_api.h>

#include <iostream>

#include "accuracy_tester.h"
#include "cmd_args.h"

int main(int argc, char** argv) {
  try {
    AppArgs args;
    Ort::Env env;

    if (!ParseCmdLineArgs(args, argc, argv, env)) {
      return 1;
    }

    if (!RunAccuracyTest(env, args)) {
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "[EXCEPTION]: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
