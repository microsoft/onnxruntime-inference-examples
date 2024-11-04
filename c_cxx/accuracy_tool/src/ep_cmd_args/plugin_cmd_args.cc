// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "ep_cmd_args/plugin_cmd_args.h"

#include <onnxruntime_c_api_ep.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <unordered_set>

static bool ParseEPPluginOptions(const std::string& ep_config_string, std::vector<std::string>& keys,
                                 std::vector<std::string>& values);

// Expect "plugin <ep_name> <plugin_lib_path> "key1|val1 key2|val2 ..."
bool ParseEpPluginArgs(AppArgs& app_args, CmdArgParser& cmd_args, std::string_view prog_name, Ort::Env& env) {
  if (!cmd_args.HasNext()) {
    std::cerr << "[ERROR]: Must specify the name for the EP plugin." << std::endl;
    return false;
  }

  std::string plugin_ep_name = std::string(cmd_args.GetNext());
  if (!cmd_args.HasNext()) {
    std::cerr << "[ERROR]: Must specify a valid path for the EP plugin's library." << std::endl;
    return false;
  }

  std::filesystem::path plugin_ep_lib_path;
  if (!GetValidPath(prog_name, cmd_args.GetNext(), false, plugin_ep_lib_path)) {
    return false;
  }

  const OrtApi* c_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  Ort::Status status(
      c_api->RegisterPluginExecutionProviderLibrary(plugin_ep_lib_path.native().c_str(), env, plugin_ep_name.c_str()));

  if (!status.IsOK()) {
    std::cerr << "[ERROR]: Failed to register EP plugin library with error code " << status.GetErrorCode() << ": "
              << status.GetErrorMessage() << std::endl;
    return false;
  }

  std::vector<std::string> option_keys;
  std::vector<std::string> option_vals;
  if (cmd_args.HasNext()) {
    if (!ParseEPPluginOptions(std::string(cmd_args.GetNext()), option_keys, option_vals)) {
      return false;
    }
  }

  const size_t num_options = option_keys.size();
  assert(option_vals.size() == num_options);

  std::vector<const char*> c_keys;
  std::vector<const char*> c_vals;

  if (num_options) {
    c_keys.reserve(num_options);
    c_vals.reserve(num_options);
    for (size_t i = 0; i < num_options; i++) {
      c_keys.push_back(option_keys[i].c_str());
      c_vals.push_back(option_vals[i].c_str());
    }
  }

  Ort::Status status_append(c_api->SessionOptionsAppendPluginExecutionProvider(
      app_args.session_options, plugin_ep_name.c_str(), env, c_keys.data(), c_vals.data(), num_options));

  if (!status_append.IsOK()) {
    std::cerr << "[ERROR]: Failed to append EP plugin to session with error code " << status_append.GetErrorCode()
              << ": " << status_append.GetErrorMessage() << std::endl;
    return false;
  }

  return true;
}

static bool ParseEPPluginOptions(const std::string& ep_config_string, std::vector<std::string>& keys,
                                 std::vector<std::string>& values) {
  std::istringstream ss(ep_config_string);
  std::string token;

  while (ss >> token) {
    if (token == "") {
      continue;
    }
    std::string_view token_sv(token);

    auto pos = token_sv.find("|");
    if (pos == std::string_view::npos || pos == 0 || pos == token_sv.length()) {
      std::cerr << "Use a '|' to separate the key and value for the run-time option you are trying to use."
                << std::endl;
      return false;
    }

    std::string_view key(token_sv.substr(0, pos));
    std::string_view value(token_sv.substr(pos + 1));

    keys.push_back(std::string(key));
    values.push_back(std::string(value));
  }

  return true;
}
