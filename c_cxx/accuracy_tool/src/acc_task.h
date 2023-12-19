// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <functional>
#include <variant>

#include "basic_utils.h"
#include "model_io_utils.h"

class Task {
 private:
  struct Inference {
    Span<char> output_buffer;
  };

  struct AccuracyCheck {
    Span<const char> expected_output_buffer;
    Span<AccMetrics> output_acc_metric;
  };

 public:
  Task(Task&& other) = default;
  Task(const Task& other) = default;

  static Task CreateInferenceTask(Ort::Session& session, const ModelIOInfo& model_io_info,
                                  Span<const char> input_buffer, Span<char> output_buffer);

  static Task CreateAccuracyCheckTask(Ort::Session& session, const ModelIOInfo& model_io_info,
                                      Span<const char> input_buffer, Span<const char> expected_output_buffer,
                                      Span<AccMetrics> output_acc_metric);

  void Run();

 private:
  Task(Ort::Session& session, const ModelIOInfo& model_io_info, Span<const char> input_buffer,
       Span<char> output_buffer);
  Task(Ort::Session& session, const ModelIOInfo& model_io_info, Span<const char> input_buffer,
       Span<const char> expected_output_buffer, Span<AccMetrics> output_acc_metric);

  void RunAsInferenceTask(Inference& inference_args);
  void RunAsAccuracyCheckTask(AccuracyCheck& accuracy_check_args);

  std::reference_wrapper<Ort::Session> session_;
  std::reference_wrapper<const ModelIOInfo> model_io_info_;
  Span<const char> input_buffer_;
  std::variant<Inference, AccuracyCheck> variant_;
};
