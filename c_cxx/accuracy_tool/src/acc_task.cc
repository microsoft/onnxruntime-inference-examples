// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "acc_task.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <variant>
#include <vector>

static std::vector<Ort::Value> RunInference(Ort::Session& session, const ModelIOInfo& model_io_info,
                                            Span<const char> input_buffer) {
  // Setup input
  const std::vector<IOInfo>& input_infos = model_io_info.inputs;
  const size_t num_inputs = input_infos.size();
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  ort_inputs.reserve(num_inputs);
  ort_input_names.reserve(num_inputs);

  for (size_t input_offset = 0, i = 0; i < num_inputs; input_offset += input_infos[i].total_data_size, i++) {
    assert(input_offset < input_buffer.size());
    const IOInfo& input_info = input_infos[i];
    Span<const char> input_data(&input_buffer[input_offset], input_info.total_data_size);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    ort_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, (void*)input_data.data(), input_data.size(),
                                                     input_info.shape.data(), input_info.shape.size(),
                                                     input_info.data_type));
    ort_input_names.push_back(input_info.name.c_str());
  }

  const size_t num_outputs = model_io_info.outputs.size();
  std::vector<const char*> ort_output_names;
  ort_output_names.reserve(num_outputs);

  for (size_t i = 0; i < num_outputs; i++) {
    ort_output_names.push_back(model_io_info.outputs[i].name.c_str());
  }

  return session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(), ort_inputs.size(),
                     ort_output_names.data(), ort_output_names.size());
}

Task::Task(Ort::Session& session, const ModelIOInfo& model_io_info, Span<const char> input_buffer,
           Span<char> output_buffer)
    : session_(session),
      model_io_info_(model_io_info),
      input_buffer_(input_buffer),
      variant_(Inference{output_buffer}) {}

Task::Task(Ort::Session& session, const ModelIOInfo& model_io_info, Span<const char> input_buffer,
           Span<const char> expected_output_buffer, Span<AccMetrics> output_acc_metric)
    : session_(session),
      model_io_info_(model_io_info),
      input_buffer_(input_buffer),
      variant_(AccuracyCheck{expected_output_buffer, output_acc_metric}) {}

Task Task::CreateInferenceTask(Ort::Session& session, const ModelIOInfo& model_io_info, Span<const char> input_buffer,
                               Span<char> output_buffer) {
  return Task(session, model_io_info, input_buffer, output_buffer);
}

Task Task::CreateAccuracyCheckTask(Ort::Session& session, const ModelIOInfo& model_io_info,
                                   Span<const char> input_buffer, Span<const char> expected_output_buffer,
                                   Span<AccMetrics> output_acc_metric) {
  return Task(session, model_io_info, input_buffer, expected_output_buffer, output_acc_metric);
}

void Task::Run() {
  AccuracyCheck* accuracy_check_data = std::get_if<AccuracyCheck>(&variant_);
  if (accuracy_check_data) {
    RunAsAccuracyCheckTask(*accuracy_check_data);
    return;
  }

  Inference* inference_data = std::get_if<Inference>(&variant_);
  if (inference_data) {
    RunAsInferenceTask(*inference_data);
    return;
  }

  // Should not reach this line unless we add a new (unhandled) std::variant type.
  std::cerr << "[ERROR]: Unhandled std::variant type for Task::variant_ member." << std::endl;
  std::abort();
}

void Task::RunAsInferenceTask(Inference& inference_args) {
  std::vector<Ort::Value> ort_output_vals = RunInference(session_, model_io_info_, input_buffer_);
  Span<char>& output_buffer = inference_args.output_buffer;

  // Unfortunately, we have to copy output values (Ort::Value is not copyable, so it is limited when stored in a
  // std::vector)
  const std::vector<IOInfo>& output_infos = model_io_info_.get().outputs;
  const size_t num_outputs = output_infos.size();

  for (size_t output_offset = 0, i = 0; i < num_outputs; output_offset += output_infos[i].total_data_size, i++) {
    assert(output_offset < output_buffer.size());
    std::memcpy(&output_buffer[output_offset], ort_output_vals[i].GetTensorRawData(), output_infos[i].total_data_size);
  }
}

void Task::RunAsAccuracyCheckTask(AccuracyCheck& accuracy_check_args) {
  std::vector<Ort::Value> ort_output_vals = RunInference(session_, model_io_info_, input_buffer_);

  const std::vector<IOInfo>& output_infos = model_io_info_.get().outputs;
  const size_t num_outputs = output_infos.size();
  Span<const char> expected_output_buffer = accuracy_check_args.expected_output_buffer;

  for (size_t output_offset = 0, i = 0; i < num_outputs; output_offset += output_infos[i].total_data_size, i++) {
    assert(output_offset < expected_output_buffer.size());
    const IOInfo& output_info = output_infos[i];
    Span<const char> raw_expected_output(&expected_output_buffer[output_offset], output_info.total_data_size);

    accuracy_check_args.output_acc_metric[i] =
        ComputeAccuracyMetric(ort_output_vals[i].GetConst(), raw_expected_output, output_info);
  }
}
