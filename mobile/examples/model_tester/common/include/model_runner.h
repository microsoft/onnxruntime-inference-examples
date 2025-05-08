#pragma once

#include <cstdint>

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace model_runner {

struct RunConfig {
  std::string model_path{};

  size_t num_warmup_iterations{};
  size_t num_iterations{};

  std::optional<int> log_level{};
};

using Clock = std::chrono::steady_clock;
using Duration = Clock::duration;

struct RunResult {
  Duration load_duration;
  std::vector<Duration> run_durations;
};

RunResult Run(const RunConfig& run_config);

std::string GetRunSummary(const RunConfig& run_config, const RunResult& run_result);

}  // namespace model_runner
