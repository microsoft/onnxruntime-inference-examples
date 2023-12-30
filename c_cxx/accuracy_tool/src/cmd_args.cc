// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cmd_args.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_set>

#include "ep_cmd_args/qnn_cmd_args.h"

void PrintUsage(std::ostream& stream, std::string_view prog_name) {
  const std::string prog_exec_name = std::filesystem::path(prog_name).filename().string();
  stream << "Usage: " << prog_exec_name << " [OPTIONS...] test_models_path" << std::endl << std::endl;
  stream << "[OPTIONS]:" << std::endl;
  stream << " -h/--help                        Print this help message and exit program" << std::endl;
  stream << " -j/--num_threads num_threads     Number of threads to use for inference." << std::endl;
  stream << "                                  Defaults to number of cores." << std::endl;
  stream << " -l/--load_expected_outputs       Load expected outputs from raw output_<index>.raw files" << std::endl;
  stream << "                                  Defaults to false." << std::endl;
  stream << " -s/--save_expected_outputs       Save outputs from baseline model on CPU EP to disk as " << std::endl;
  stream << "                                  output_<index>.raw files. Defaults to false." << std::endl;
  stream << " -e/--execution_provider ep [EP_ARGS]  The execution provider to test (e.g., qnn or cpu)" << std::endl;
  stream << "                                       Defaults to CPU execution provider running QDQ model." << std::endl;
  stream << " -c/--session_configs \"<key1>|<val1> <key2>|<val2>\"  Session configuration options for EP under test."
         << std::endl;
  stream << "                                                     Refer to onnxruntime_session_options_config_keys.h"
         << std::endl;
  stream << " -o/--output_file path                 The output file into which to save accuracy results" << std::endl;
  stream << " -a/--expected_accuracy_file path      The file containing expected accuracy results" << std::endl;
  stream << " --model model_name                    Model to test. Option can be specified multiple times."
         << std::endl;
  stream << "                                       By default, all found models are tested." << std::endl;
  stream << std::endl;
  stream << "[EP_ARGS]: Specify EP-specific runtime options as key value pairs." << std::endl;
  stream << "  Example: -e <provider_name> \"<key1>|<val1> <key2>|<val2>\"" << std::endl;
  stream << "  [QNN only] [backend_path]: QNN backend path (e.g., 'C:\\Path\\QnnHtp.dll')" << std::endl;
  stream << "  [QNN only] [profiling_level]: QNN profiling level, options: 'basic', 'detailed'," << std::endl;
  stream << "                                default 'off'." << std::endl;
  stream << "  [QNN only] [rpc_control_latency]: QNN rpc control latency. default to 10." << std::endl;
  stream << "  [QNN only] [vtcm_mb]: QNN VTCM size in MB. default to 0 (not set)." << std::endl;
  stream << "  [QNN only] [htp_performance_mode]: QNN performance mode, options: 'burst', 'balanced', " << std::endl;
  stream << "             'default', 'high_performance', 'high_power_saver'," << std::endl;
  stream << "             'low_balanced', 'low_power_saver', 'power_saver'," << std::endl;
  stream << "             'sustained_high_performance'. Defaults to 'default'." << std::endl;
  stream << "  [QNN only] [qnn_context_priority]: QNN context priority, options: 'low', 'normal'," << std::endl;
  stream << "             'normal_high', 'high'. Defaults to 'normal'." << std::endl;
  stream << "  [QNN only] [qnn_saver_path]: QNN Saver backend path. e.g 'C:\\Path\\QnnSaver.dll'." << std::endl;
  stream << "  [QNN only] [htp_graph_finalization_optimization_mode]: QNN graph finalization" << std::endl;
  stream << "             optimization mode, options: '0', '1', '2', '3'. Default is '0'." << std::endl;
}

static bool ParseSessionConfigs(const std::string& configs_string,
                                std::unordered_map<std::string, std::string>& session_configs) {
  std::istringstream ss(configs_string);
  std::string token;

  while (ss >> token) {
    if (token == "") {
      continue;
    }

    std::string_view token_sv(token);

    auto pos = token_sv.find("|");
    if (pos == std::string_view::npos || pos == 0 || pos == token_sv.length()) {
      std::cerr << "Use a '|' to separate the key and value for session configuration options." << std::endl;
      return false;
    }

    std::string key(token_sv.substr(0, pos));
    std::string value(token_sv.substr(pos + 1));

    auto it = session_configs.find(key);
    if (it != session_configs.end()) {
      std::cerr << "[ERROR]: Specified duplicate session config option: " << key << std::endl;
      return false;
    }

    session_configs.insert(std::make_pair(std::move(key), std::move(value)));
  }

  return true;
}

static void SetDefaultCpuEpArgs(AppArgs& app_args) {
  app_args.uses_qdq_model = true;  // TODO: Make configurable?
  app_args.supports_multithread_inference = true;
  app_args.execution_provider = "cpu";
}

bool GetValidPath(std::string_view prog_name, std::string_view provided_path, bool is_dir,
                  std::filesystem::path& valid_path) {
  std::filesystem::path path = provided_path;
  std::error_code error_code;

  if (!std::filesystem::exists(path, error_code)) {
    std::cerr << "[ERROR]: Invalid path " << provided_path << ": " << error_code.message() << std::endl << std::endl;
    return false;
  }

  std::error_code abs_error_code;
  std::filesystem::path abs_path = std::filesystem::absolute(path, abs_error_code);
  if (abs_error_code) {
    std::cerr << "[ERROR]: Invalid path: " << abs_error_code.message() << std::endl << std::endl;
    return false;
  }

  if (is_dir && !std::filesystem::is_directory(abs_path)) {
    std::cerr << "[ERROR]: " << provided_path << " is not a directory" << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  if (!is_dir && !std::filesystem::is_regular_file(abs_path)) {
    std::cerr << "[ERROR]: " << provided_path << " is not a regular file" << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  valid_path = std::move(abs_path);

  return true;
}

bool ParseCmdLineArgs(AppArgs& app_args, int argc, char** argv) {
  CmdArgParser cmd_args(argc, argv);
  std::string_view prog_name = cmd_args.GetNext();

  app_args.num_threads = std::max(static_cast<unsigned int>(1), std::thread::hardware_concurrency());

  // Parse command-line arguments.
  while (cmd_args.HasNext()) {
    std::string_view arg = cmd_args.GetNext();

    if (arg == "-h" || arg == "--help") {
      PrintUsage(std::cout, prog_name);
      std::exit(0);
    } else if (arg == "-o" || arg == "--output_file") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      app_args.output_file = cmd_args.GetNext();
    } else if (arg == "-j" || arg == "--num_threads") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      int n = std::stoi(std::string(cmd_args.GetNext()));
      if (n <= 0) {
        std::cerr << "[ERROR]: Must specify a positive non-zero number of threads." << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      app_args.num_threads = std::min(static_cast<unsigned int>(n), std::thread::hardware_concurrency());
    } else if (arg == "--model") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      arg = cmd_args.GetNext();
      app_args.only_models.insert(std::string(arg));
    } else if (arg == "-a" || arg == "--expected_accuracy_file") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      arg = cmd_args.GetNext();
      if (!GetValidPath(prog_name, arg, false, app_args.expected_accuracy_file)) {
        return false;
      }
    } else if (arg == "-e" || arg == "--execution_provider") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      arg = cmd_args.GetNext();
      if (arg == "qnn") {
        if (!ParseQnnEpArgs(app_args, cmd_args)) {
          return false;
        }
      } else if (arg == "cpu") {
        SetDefaultCpuEpArgs(app_args);
      } else {
        std::cerr << "[ERROR]: Unsupported execution provider: " << arg << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }
    } else if (arg == "-c" || arg == "--session_configs") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      arg = cmd_args.GetNext();
      std::unordered_map<std::string, std::string> session_configs;

      if (!ParseSessionConfigs(std::string(arg), session_configs)) {
        return false;
      }

      for (auto& it : session_configs) {
        app_args.session_options.AddConfigEntry(it.first.c_str(), it.second.c_str());
      }
    } else if (arg == "-s" || arg == "--save_expected_outputs") {
      app_args.save_expected_outputs_to_disk = true;
    } else if (arg == "-l" || arg == "--load_expected_outputs") {
      app_args.load_expected_outputs_from_disk = true;
    } else if (app_args.test_dir.empty()) {
      if (!GetValidPath(prog_name, arg, true, app_args.test_dir)) {
        return false;
      }
    } else {
      std::cerr << "[ERROR]: unknown command-line argument `" << arg << "`" << std::endl << std::endl;
      PrintUsage(std::cerr, prog_name);
      return false;
    }
  }

  //
  // Final argument validation:
  //

  if (app_args.test_dir.empty()) {
    std::cerr << "[ERROR]: Must provide a models directory." << std::endl << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  if (app_args.execution_provider.empty()) {
    SetDefaultCpuEpArgs(app_args);
  }

  if (app_args.load_expected_outputs_from_disk && app_args.save_expected_outputs_to_disk) {
    std::cerr << "[ERROR]: Cannot enable both -s/--save_expected_outputs and -l/--load_expected_outputs" << std::endl
              << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  return true;
}