#include "model_runner.h"

#include <cstddef>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <iterator>
#include <numeric>
#include <span>

#include "onnxruntime_cxx_api.h"

namespace model_runner {

namespace {

size_t GetDataTypeSizeInBytes(ONNXTensorElementDataType data_type) {
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return 8;
    default:
      throw std::invalid_argument(std::format("unsupported tensor data type: {}", static_cast<int>(data_type)));
  }
}

void FillTensorWithZeroes(Ort::Value& value) {
  const auto tensor_info = value.GetTensorTypeAndShapeInfo();
  const auto data_type = tensor_info.GetElementType();
  const auto num_elements = tensor_info.GetElementCount();
  const auto data_type_size_in_bytes = GetDataTypeSizeInBytes(data_type);
  const auto data_size_in_bytes = num_elements * data_type_size_in_bytes;

  std::byte* data = static_cast<std::byte*>(value.GetTensorMutableRawData());
  std::fill(data, data + data_size_in_bytes, std::byte{0});
}

std::vector<Ort::Value> GetModelInputValues(const Ort::Session& session) {
  const auto num_inputs = session.GetInputCount();

  std::vector<Ort::Value> input_values{};
  input_values.reserve(num_inputs);

  Ort::AllocatorWithDefaultOptions allocator{};

  for (size_t i = 0; i < num_inputs; ++i) {
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    auto tensor_shape = tensor_info.GetShape();
    // make this a static shape
    for (auto& dim : tensor_shape) {
      if (dim == -1) {
        dim = 1;
      }
    }

    const auto tensor_data_type = tensor_info.GetElementType();

    auto value = Ort::Value::CreateTensor(allocator, tensor_shape.data(), tensor_shape.size(), tensor_data_type);

    FillTensorWithZeroes(value);

    input_values.emplace_back(std::move(value));
  }

  return input_values;
}

std::vector<std::string> GetModelInputOrOutputNames(const Ort::Session& session, bool is_input) {
  const auto num_inputs_or_outputs = is_input ? session.GetInputCount() : session.GetOutputCount();

  std::vector<std::string> names{};
  names.reserve(num_inputs_or_outputs);

  auto allocator = Ort::AllocatorWithDefaultOptions{};
  for (size_t i = 0; i < num_inputs_or_outputs; ++i) {
    auto name = is_input ? session.GetInputNameAllocated(i, allocator)
                         : session.GetOutputNameAllocated(i, allocator);
    names.emplace_back(name.get());
  }

  return names;
}

std::vector<std::string> GetModelInputNames(const Ort::Session& session) {
  return GetModelInputOrOutputNames(session, /* is_input */ true);
}

std::vector<std::string> GetModelOutputNames(const Ort::Session& session) {
  return GetModelInputOrOutputNames(session, /* is_input */ false);
}

std::vector<const char*> GetCstrs(std::span<const std::string> strs) {
  std::vector<const char*> cstrs{};
  cstrs.reserve(strs.size());
  std::transform(strs.begin(), strs.end(), std::back_inserter(cstrs),
                 [](const std::string& str) { return str.c_str(); });
  return cstrs;
}

class Timer {
 public:
  Timer() { Reset(); }

  void Reset() { start_ = Clock::now(); }

  Duration Elapsed() const { return Clock::now() - start_; }

 private:
  Clock::time_point start_;
};

struct RunResultStats {
  using DurationFp = std::chrono::duration<float, Duration::period>;

  size_t n;
  DurationFp average;
  Duration min, max;
  Duration p50, p90, p99;
};

RunResultStats ComputeRunResultStats(const RunResult& run_result) {
  using DurationFp = RunResultStats::DurationFp;

  const auto& run_durations = run_result.run_durations;

  RunResultStats stats{};
  const auto n = run_durations.size();
  stats.n = n;
  if (n > 0) {
    const auto total_run_duration = std::accumulate(run_durations.begin(), run_durations.end(),
                                                    DurationFp{0.0f});
    stats.average = DurationFp{total_run_duration.count() / n};

    auto sorted_run_durations = run_durations;
    std::sort(sorted_run_durations.begin(), sorted_run_durations.end());
    stats.min = sorted_run_durations.front();
    stats.max = sorted_run_durations.back();
    stats.p50 = sorted_run_durations[static_cast<size_t>(0.5f * n)];
    stats.p90 = sorted_run_durations[static_cast<size_t>(0.9f * n)];
    stats.p99 = sorted_run_durations[static_cast<size_t>(0.99f * n)];
  }

  return stats;
}

}  // namespace

RunResult Run(const RunConfig& run_config) {
  RunResult run_result{};

  auto env = Ort::Env{};

  if (run_config.log_level.has_value()) {
    env.UpdateEnvWithCustomLogLevel(static_cast<OrtLoggingLevel>(*run_config.log_level));
  }

  auto session_options = Ort::SessionOptions{};

  if (const auto& ep_config = run_config.ep; ep_config.has_value()) {
    session_options.AppendExecutionProvider(ep_config->provider_name, ep_config->provider_options);
  }

  Timer timer{};

  auto session = Ort::Session{nullptr};
  if (std::holds_alternative<std::string>(run_config.model_path_or_bytes)) {
    const auto& model_path = std::get<std::string>(run_config.model_path_or_bytes);
    timer.Reset();
    session = Ort::Session{env, model_path.c_str(), session_options};
    run_result.load_duration = timer.Elapsed();
  } else {
    const auto& model_bytes = std::get<std::span<const std::byte>>(run_config.model_path_or_bytes);
    timer.Reset();
    session = Ort::Session{env, model_bytes.data(), model_bytes.size(), session_options};
    run_result.load_duration = timer.Elapsed();
  }

  auto input_names = GetModelInputNames(session);
  auto input_name_cstrs = GetCstrs(input_names);

  auto input_values = GetModelInputValues(session);

  auto output_names = GetModelOutputNames(session);
  auto output_name_cstrs = GetCstrs(output_names);

  auto run_options = Ort::RunOptions{};

  run_result.run_durations.reserve(run_config.num_iterations);

  // warmup
  if (run_config.run_warmup_iteration) {
    auto outputs = session.Run(run_options,
                               input_name_cstrs.data(), input_values.data(), input_values.size(),
                               output_name_cstrs.data(), output_name_cstrs.size());
  }

  // measure runs
  for (size_t i = 0; i < run_config.num_iterations; ++i) {
    timer.Reset();
    auto outputs = session.Run(run_options,
                               input_name_cstrs.data(), input_values.data(), input_values.size(),
                               output_name_cstrs.data(), output_name_cstrs.size());
    run_result.run_durations.push_back(timer.Elapsed());
  }

  return run_result;
}

std::string GetRunSummary(const RunConfig& /*run_config*/, const RunResult& run_result) {
  auto to_display_duration = []<typename Rep, typename Period>(std::chrono::duration<Rep, Period> d) {
    using DisplayPeriod = std::chrono::microseconds::period;
    using DisplayDuration = std::chrono::duration<Rep, DisplayPeriod>;
    return std::chrono::duration_cast<DisplayDuration>(d);
  };

  const auto stats = ComputeRunResultStats(run_result);

  const auto summary = std::format(
      "Load time: {}\n"
      "N (number of runs): {}\n"
      "Latency\n"
      "  avg: {}\n"
      "  p50: {}\n"
      "  p90: {}\n"
      "  p99: {}\n"
      "  min: {}\n"
      "  max: {}\n",
      to_display_duration(run_result.load_duration),
      stats.n,
      to_display_duration(stats.average),
      to_display_duration(stats.p50),
      to_display_duration(stats.p90),
      to_display_duration(stats.p99),
      to_display_duration(stats.min),
      to_display_duration(stats.max));

  return summary;
}

}  // namespace model_runner
