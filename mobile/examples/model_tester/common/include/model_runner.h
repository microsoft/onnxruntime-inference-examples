#pragma once

#include <cstddef>
#include <cstdint>

#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace model_runner {

using Clock = std::chrono::steady_clock;
using Duration = Clock::duration;

struct RunConfig {
  using ModelPathOrBytes = std::variant<std::string, std::span<const std::byte>>;

  // Path or bytes of the model to run.
  ModelPathOrBytes model_path_or_bytes{};

  // Whether to run a warmup iteration before running the measured (timed) iterations.
  bool run_warmup_iteration{true};

  // Number of iterations to run.
  size_t num_iterations{10};

  // Configuration for an Execution Provider (EP).
  struct EpConfig {
    std::string provider_name{};
    std::unordered_map<std::string, std::string> provider_options{};
  };

  // Specifies the EP to use in the session.
  std::optional<EpConfig> ep{};

  // Specifies the onnxruntime log level.
  std::optional<int> log_level{};
};

struct RunResult {
  // Time taken to load the model.
  Duration load_duration;

  // Times taken to run the model.
  std::vector<Duration> run_durations;
};

RunResult Run(const RunConfig& run_config);

std::string GetRunSummary(const RunConfig& run_config, const RunResult& run_result);

}  // namespace model_runner
