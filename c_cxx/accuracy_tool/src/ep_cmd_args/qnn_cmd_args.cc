// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "ep_cmd_args/qnn_cmd_args.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <iostream>
#include <sstream>
#include <unordered_set>

static bool ParseQnnRuntimeOptions(const std::string& ep_config_string,
                                   std::unordered_map<std::string, std::string>& qnn_options) {
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

    if (key == "backend_path") {
      if (value.empty()) {
        std::cerr << "[ERROR]: Please provide the QNN backend path." << std::endl;
        return false;
      }
    } else if (key == "qnn_context_cache_enable") {
      if (value != "1") {
        std::cerr << "[ERROR]: Set to 1 to enable qnn_context_cache_enable." << std::endl;
        return false;
      }
    } else if (key == "qnn_context_cache_path") {
      // no validation
    } else if (key == "profiling_level") {
      std::unordered_set<std::string_view> supported_profiling_level = {"off", "basic", "detailed"};
      if (supported_profiling_level.find(value) == supported_profiling_level.end()) {
        std::cerr << "[ERROR]: Supported profiling_level: off, basic, detailed" << std::endl;
        return false;
      }
    } else if (key == "rpc_control_latency" || key == "vtcm_mb") {
      // no validation
    } else if (key == "htp_performance_mode") {
      std::unordered_set<std::string_view> supported_htp_perf_mode = {
          "burst",        "balanced",        "default",     "high_performance",          "high_power_saver",
          "low_balanced", "low_power_saver", "power_saver", "sustained_high_performance"};
      if (supported_htp_perf_mode.find(value) == supported_htp_perf_mode.end()) {
        std::ostringstream str_stream;
        std::copy(supported_htp_perf_mode.begin(), supported_htp_perf_mode.end(),
                  std::ostream_iterator<std::string_view>(str_stream, ","));
        std::string str = str_stream.str();
        std::cerr << "[ERROR]: Supported htp_performance_mode: " << str << std::endl;
        return false;
      }
    } else if (key == "qnn_saver_path") {
      // no validation
    } else if (key == "htp_graph_finalization_optimization_mode") {
      std::unordered_set<std::string_view> supported_htp_graph_final_opt_modes = {"0", "1", "2", "3"};
      if (supported_htp_graph_final_opt_modes.find(value) == supported_htp_graph_final_opt_modes.end()) {
        std::ostringstream str_stream;
        std::copy(supported_htp_graph_final_opt_modes.begin(), supported_htp_graph_final_opt_modes.end(),
                  std::ostream_iterator<std::string_view>(str_stream, ","));
        std::string str = str_stream.str();
        std::cerr << "[ERROR]: Wrong value for htp_graph_finalization_optimization_mode. select from: " << str
                  << std::endl;
        return false;
      }
    } else if (key == "qnn_context_priority") {
      std::unordered_set<std::string_view> supported_qnn_context_priority = {"low", "normal", "normal_high", "high"};
      if (supported_qnn_context_priority.find(value) == supported_qnn_context_priority.end()) {
        std::cerr << "[ERROR]: Supported qnn_context_priority: low, normal, normal_high, high" << std::endl;
        return false;
      }
    } else {
      std::cerr
          << R"([ERROR]: Wrong key type entered. Choose from options: ['backend_path', 'qnn_context_cache_enable',
'qnn_context_cache_path', 'profiling_level', 'rpc_control_latency', 'vtcm_mb', 'htp_performance_mode',
'qnn_saver_path', 'htp_graph_finalization_optimization_mode', 'qnn_context_priority'])"
          << std::endl;
      return false;
    }

    qnn_options.insert(std::make_pair(std::string(key), std::string(value)));
  }

  return true;
}

bool ParseQnnEpArgs(AppArgs& app_args, CmdArgParser& cmd_args) {
  if (!cmd_args.HasNext()) {
    std::cerr << "[ERROR]: Must specify at least a QNN backend path." << std::endl;
    return false;
  }

  std::string_view args = cmd_args.GetNext();
  std::unordered_map<std::string, std::string> qnn_options;

  if (!ParseQnnRuntimeOptions(std::string(args), qnn_options)) {
    return false;
  }

  auto backend_iter = qnn_options.find("backend_path");
  if (backend_iter == qnn_options.end()) {
    std::cerr << "[ERROR]: Must provide a backend_path for the QNN execution provider." << std::endl;
    return false;
  }

  app_args.session_options.AppendExecutionProvider("QNN", qnn_options);
  app_args.uses_qdq_model = backend_iter->second.rfind("QnnHtp") != std::string::npos;
  app_args.supports_multithread_inference = false;  // TODO: Work on enabling multi-threaded inference.
  app_args.execution_provider = "qnn";
  return true;
}
