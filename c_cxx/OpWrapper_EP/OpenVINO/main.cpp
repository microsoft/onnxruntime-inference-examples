// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include <iomanip>

#include "utils.h"

static void PrintUsage(std::ostream& stream, std::string_view prog_name) {
  stream << "Usage: " << prog_name << " [OPTIONS] <onnx_mnist_model_path> <custom_op_lib_path>"
         << std::endl;
  stream << "OPTIONS:" << std::endl;
  stream << "    -h/--help      Print this help message" << std::endl << std::endl;
}

struct CmdArgs {
  CmdArgs(int argc, char** argv) noexcept : argc_(argc), argv_(argv), index_(0) {}

  [[nodiscard]] bool HasNext() const { return index_ < argc_; }

  [[nodiscard]] std::string_view GetNext() {
    assert(HasNext());
    return argv_[index_++];
  }

 private:
  int argc_;
  char** argv_;
  int index_;
};

constexpr size_t MODEL_EXPECTED_WIDTH = 28;
constexpr size_t MODEL_EXPECTED_HEIGHT = 28;
constexpr size_t MODEL_EXPECTED_CHANNELS = 1;

static std::array<const uint8_t, MODEL_EXPECTED_HEIGHT * MODEL_EXPECTED_WIDTH> digit_1_bytes = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xca, 0xff, 0xca, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xda, 0xfe, 0xda, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xff, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xfe, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfa, 0xff, 0xfa, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xfe, 0xff, 0xfe, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0xf1, 0xfe, 0xf1, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0x1a, 0xca, 0xca, 0xca, 0x1a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

int main(int argc, char** argv) {
  try {
    CmdArgs args(argc, argv);
    std::string_view prog_name = args.GetNext();

#ifdef _WIN32
    const wchar_t* default_model_path = L"data/custom_op_mnist_ov_wrapper.onnx";
    const char* default_custom_op_lib_path = "openvino_wrapper.dll";
    std::wstring wide_model_path;
    std::wstring_view model_path;
#else
    const char* default_model_path = "data/custom_op_mnist_ov_wrapper.onnx";
    const char* default_custom_op_lib_path = "libopenvino_wrapper.so";
    std::string_view model_path;
#endif

    std::string_view custom_op_lib_path;

    // Parse command-line arguments.
    while (args.HasNext()) {
      std::string_view arg = args.GetNext();

      if (arg == "-h" || arg == "--help") {
        PrintUsage(std::cout, prog_name);
        return 0;
      } else if (model_path.empty()) {
#ifdef _WIN32
        wide_model_path = ConvertString(arg);
        model_path = wide_model_path;
#else
        model_path = arg;
#endif
      } else if (custom_op_lib_path.empty()) {
        custom_op_lib_path = arg;
      } else {
        std::cerr << "[ERROR]: unknown command-line argument `" << arg << "`" << std::endl << std::endl;
        PrintUsage(std::cerr, prog_name);
        return 1;
      }
    }

    if (model_path.empty()) {
      model_path = default_model_path;
#ifdef _WIN32
      std::wcout << L"[WARNING]: Did not specify model path argument, trying default: "
                << default_model_path << std::endl;
#else
      std::cout << "[WARNING]: Did not specify model path argument, trying default: " << default_model_path
                 << std::endl;
#endif
    }

    if (custom_op_lib_path.empty()) {
      custom_op_lib_path = default_custom_op_lib_path;
      std::cout << "[WARNING]: Did not specify custom op lib path argument, trying default: "
                << default_custom_op_lib_path
                << std::endl;
    }

    // Create session.
    Ort::Env env;
    Ort::SessionOptions session_opts;

    // Add session config entry for the custom op.
    session_opts.AddCustomOpConfigEntry("OpenVINO_Wrapper", "device_type", "CPU");

    std::unique_ptr<void, decltype(&CleanUpCustomOpLib)> custom_op_lib(
        session_opts.RegisterCustomOpsLibrary(custom_op_lib_path.data()), CleanUpCustomOpLib);

    Ort::Session session(env, model_path.data(), session_opts);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Setup input: convert image to HWC floats
    std::array<int64_t, 4> input_shape{1, MODEL_EXPECTED_CHANNELS, MODEL_EXPECTED_HEIGHT, MODEL_EXPECTED_WIDTH};
    std::vector<float> input_vals;
    ConvertHWCToCHW(input_vals, digit_1_bytes.data(), MODEL_EXPECTED_WIDTH, MODEL_EXPECTED_HEIGHT, MODEL_EXPECTED_CHANNELS);
    assert(input_vals.size() == GetShapeSize(input_shape));

    std::array<const char*, 1> input_names{"Input3"};
    std::array<Ort::Value, 1> ort_inputs{Ort::Value::CreateTensor<float>(
        memory_info, input_vals.data(), input_vals.size(), input_shape.data(), input_shape.size())};

    // Run session and get outputs
    std::array<const char*, 1> output_names{"Plus214_Output_0"};
    std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(),
                                                      ort_inputs.size(), output_names.data(), output_names.size());

    // Extract raw output probabilities.
    Ort::Value& ort_output = ort_outputs[0];

    auto typeshape = ort_output.GetTensorTypeAndShapeInfo();
    const size_t num_outputs = typeshape.GetElementCount();
    const float* output_vals = ort_output.GetTensorData<float>();

    // Apply softmax to output values in order to convert to probabilites [0.0, 1.0].
    // Note: Could also add a Softmax node to model.
    std::vector<float> probabilities;
    Softmax(probabilities, output_vals, num_outputs);

    // Print probabilities for each digit class.
    std::ostringstream probs_sstream;
    size_t digit = 0;
    for (const auto& prob : probabilities) {
      probs_sstream << std::left << std::setw(7) << digit++ << ": " << std::fixed << std::setprecision(2) << prob
                    << std::endl;
    }

    std::cout << "Digit probabilities:" << std::endl
              << std::endl
              << "Digit   Probability" << std::endl
              << "------- -----------" << std::endl
              << probs_sstream.str() << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "[EXCEPTION]: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
