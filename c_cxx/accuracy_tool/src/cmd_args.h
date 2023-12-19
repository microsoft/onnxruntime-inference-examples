// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <cassert>
#include <filesystem>
#include <ostream>
#include <string>
#include <string_view>

struct CmdArgParser {
  CmdArgParser(int argc, char** argv) noexcept : argc_(argc), argv_(argv), index_(0) {}

  [[nodiscard]] bool HasNext() const { return index_ < argc_; }

  std::string_view GetNext() {
    assert(HasNext());
    return argv_[index_++];
  }

  [[nodiscard]] std::string_view PeekNext() {
    assert(HasNext());
    return argv_[index_];
  }

 private:
  int argc_;
  char** argv_;
  int index_;
};

struct AppArgs {
  std::filesystem::path test_dir;
  std::string output_file;
  std::filesystem::path expected_accuracy_file;
  std::string execution_provider;
  bool uses_qdq_model = false;
  bool supports_multithread_inference = true;
  bool save_expected_outputs_to_disk = false;
  bool load_expected_outputs_from_disk = false;
  size_t num_threads = 1;
  Ort::SessionOptions session_options;
};

bool ParseCmdLineArgs(AppArgs& app_args, int argc, char** argv);
void PrintUsage(std::ostream& stream, std::string_view prog_name);
bool GetValidPath(std::string_view prog_name, std::string_view provided_path, bool is_dir,
                  std::filesystem::path& valid_path);
