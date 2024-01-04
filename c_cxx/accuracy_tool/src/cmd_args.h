// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <cassert>
#include <filesystem>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_set>

/// <summary>
/// Convenience structure for getting individual command-line arguments.
/// </summary>
struct CmdArgParser {
  CmdArgParser(int argc, char** argv) noexcept : argc_(argc), argv_(argv), index_(0) {}

  /// <summary>
  /// Checks if there are any more command-line arguments remaining.
  /// </summary>
  /// <returns>True if have at least one more command-line argument</returns>
  [[nodiscard]] bool HasNext() const { return index_ < argc_; }

  /// <summary>
  /// Gets the next command-line argument.
  /// Must ensure HasNext() returns true before calling GetNext().
  /// </summary>
  /// <returns>The next command-line argument</returns>
  std::string_view GetNext() {
    assert(HasNext());
    return argv_[index_++];
  }

  /// <summary>
  /// Peeks at the next command-line argument without consuming it.
  /// Must ensure HasNext() returns true before calling PeekNext().
  /// </summary>
  /// <returns>The next command-line argument</returns>
  [[nodiscard]] std::string_view PeekNext() {
    assert(HasNext());
    return argv_[index_];
  }

 private:
  int argc_;
  char** argv_;
  int index_;
};

/// <summary>
/// The application's input arguments after parsing the command-line.
/// </summary>
struct AppArgs {
  std::filesystem::path test_dir;
  std::string output_file;
  std::filesystem::path expected_accuracy_file;
  std::unordered_set<std::string> only_models;  // Only run these models.
  std::string execution_provider;
  bool uses_qdq_model = false;
  bool supports_multithread_inference = true;
  bool save_expected_outputs_to_disk = false;
  bool load_expected_outputs_from_disk = false;
  size_t num_threads = 1;
  Ort::SessionOptions session_options;
};

/// <summary>
/// Gets the application's input arguments from the command-line.
/// </summary>
/// <param name="app_args">The output args data structure to initialize</param>
/// <param name="argc">The number of command-line arguments given to main()</param>
/// <param name="argv">The array of command-line arguments given to main()</param>
/// <returns>True if successfully parsed command-line arguments</returns>
bool ParseCmdLineArgs(AppArgs& app_args, int argc, char** argv);

/// <summary>
/// Prints how to call the program (i.e., the "help" message).
/// </summary>
/// <param name="stream">The stream (e.g., cout or cerr) used to print out the usage message</param>
/// <param name="prog_name">The program's name (typically argv[0])</param>
void PrintUsage(std::ostream& stream, std::string_view prog_name);

/// <summary>
/// Validates a path string provided via command-line arguments and returns the absolute path
/// if the path exists.
/// </summary>
/// <param name="prog_name">The program's name (typically argv[0])</param>
/// <param name="provided_path">The path from the command-line</param>
/// <param name="is_dir">True if the path should be a directory, otherwise it is assumed to be a file</param>
/// <param name="valid_path">The resulting absolute path</param>
/// <returns>True if the path is valid</returns>
bool GetValidPath(std::string_view prog_name, std::string_view provided_path, bool is_dir,
                  std::filesystem::path& valid_path);
