// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <functional>
#include <variant>

#include "basic_utils.h"
#include "model_io_utils.h"

/// <summary>
/// A class representing an "inference" or "accuracy-check" task that can be executed
/// on a separate thread. The task is created with a *dedicated* region of memory into which it can
/// write its results.
/// </summary>
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

  /// <summary>
  /// Creates a Task that runs a session and stores the inference results in an output buffer.
  /// </summary>
  /// <param name="session">The initialized ONNX Runtime session</param>
  /// <param name="model_io_info">Information about the model's input and output tensors</param>
  /// <param name="input_buffer">Constant byte buffer containing the model's input data</param>
  /// <param name="output_buffer">Output byte buffer into which to store the model's output</param>
  /// <returns>The new inference task</returns>
  static Task CreateInferenceTask(Ort::Session& session, const ModelIOInfo& model_io_info,
                                  Span<const char> input_buffer, Span<char> output_buffer);

  /// <summary>
  /// Creates a Task that runs a session and computes the accuracy when compared against expected results.
  /// </summary>
  /// <param name="session">The initialized ONNX Runtime session</param>
  /// <param name="model_io_info">Information about the model's input and output tensors</param>
  /// <param name="input_buffer">Constant byte buffer containing the model's input data</param>
  /// <param name="expected_output_buffer">Constant byte buffer containing the expected inference results</param>
  /// <param name="output_acc_metric">Output buffer into which to store the accuracy results</param>
  /// <returns>The new accuracy-check task</returns>
  static Task CreateAccuracyCheckTask(Ort::Session& session, const ModelIOInfo& model_io_info,
                                      Span<const char> input_buffer, Span<const char> expected_output_buffer,
                                      Span<AccMetrics> output_acc_metric);

  /// <summary>
  /// Runs the task.
  /// </summary>
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
