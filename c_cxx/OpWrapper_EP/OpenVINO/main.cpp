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
  stream << "Usage: " << prog_name << " [OPTIONS] <bmp_image_path> <onnx_model_path> <custom_op_lib_path>"
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

static void ConvertHWCToCHW(std::vector<float>& output, const BmpInfo& bmp_info) {
  const size_t stride = bmp_info.Height() * bmp_info.Width();
  const size_t num_colors = bmp_info.BytesPerPixel();

  output.resize(stride * num_colors);

  for (size_t i = 0; i < stride; ++i) {
    for (size_t c = 0; c < num_colors; ++c) {
      const size_t out_index = (c * stride) + i;
      const size_t inp_index = (i * num_colors) + c;

      assert(out_index < output.size());
      assert(inp_index < bmp_info.Size());

      output[out_index] = bmp_info.Data()[inp_index];
    }
  }
}

int main(int argc, char** argv) {
  try {
    CmdArgs args(argc, argv);
    std::string_view prog_name = args.GetNext();

#ifdef _WIN32
    const wchar_t* default_model_path = L"data/custom_op_ov_ep_wrapper.onnx";
    const char* default_custom_op_lib_path = "openvino_wrapper.dll";
    std::wstring wide_model_path;
    std::wstring_view model_path;
#else
    const char* default_model_path = "data/custom_op_ov_ep_wrapper.onnx";
    const char* default_custom_op_lib_path = "libopenvino_wrapper.so";
    std::string_view model_path;
#endif

    std::string_view custom_op_lib_path;
    std::string_view image_path;

    // Parse command-line arguments.
    while (args.HasNext()) {
      std::string_view arg = args.GetNext();

      if (arg == "-h" || arg == "--help") {
        PrintUsage(std::cout, prog_name);
        return 0;
      } else if (image_path.empty()) {
        image_path = arg;
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

    if (image_path.empty()) {
      std::cerr << "[ERROR]: missing <bmp_image_path> argument" << std::endl << std::endl;
      PrintUsage(std::cerr, prog_name);
      return 1;
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

    session_opts.AddConfigEntry("custom_op.OpenVINO_EP_Wrapper.device_type", "CPU");

    std::unique_ptr<void, decltype(&CleanUpCustomOpLib)> custom_op_lib(
        session_opts.RegisterCustomOpsLibrary(custom_op_lib_path.data()), CleanUpCustomOpLib);

    Ort::Session session(env, model_path.data(), session_opts);

    constexpr size_t MODEL_EXPECTED_WIDTH = 224;
    constexpr size_t MODEL_EXPECTED_HEIGHT = 224;

    // Load bmp.
    BmpInfo bmp_info(image_path.data());
    BmpInfo::LoadStatus bmp_status = bmp_info.Load();
    if (bmp_status != BmpInfo::LoadStatus::Ok) {
      std::cerr << "[ERROR]: Unable to load/parse BMP image " << image_path << ": "
                << BmpInfo::LoadStatusString(bmp_status) << std::endl;
      return 1;
    }

    // TODO: Support arbitrarily sized BMP images with alpha. Need to do a bilinear resize.
    if (bmp_info.BytesPerPixel() != 3) {
      std::cerr << "[ERROR]: Input BMP image must have a bit depth of 24 bits (no alpha)." << std::endl;
      return 1;
    }

    if (bmp_info.Width() != MODEL_EXPECTED_WIDTH || bmp_info.Height() != MODEL_EXPECTED_HEIGHT) {
      std::cerr << "[ERROR]: Please resize image to " << MODEL_EXPECTED_WIDTH << "x" << MODEL_EXPECTED_HEIGHT
                << std::endl;
      return 1;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Setup input: convert image to HWC floats
    std::array<int64_t, 4> input_shape{1, 3, MODEL_EXPECTED_HEIGHT, MODEL_EXPECTED_WIDTH};
    std::vector<float> input_vals;
    ConvertHWCToCHW(input_vals, bmp_info);
    assert(input_vals.size() == GetShapeSize(input_shape));

    std::array<const char*, 1> input_names{"data"};
    std::array<Ort::Value, 1> ort_inputs{Ort::Value::CreateTensor<float>(
        memory_info, input_vals.data(), input_vals.size(), input_shape.data(), input_shape.size())};

    // Run session and get outputs
    std::array<const char*, 1> output_names{"prob"};
    std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(),
                                                      ort_inputs.size(), output_names.data(), output_names.size());

    // Extract raw output probabilities.
    Ort::Value& ort_output = ort_outputs[0];

    auto typeshape = ort_output.GetTensorTypeAndShapeInfo();
    const size_t num_probs = typeshape.GetElementCount();
    const float* probs = ort_output.GetTensorData<float>();

    // Create sorted class IDs + probabilities.
    std::vector<std::pair<size_t, float>> sorted_probs;
    sorted_probs.reserve(num_probs);

    for (size_t cls = 0; cls < num_probs; ++cls) {
      sorted_probs.emplace_back(cls, probs[cls]);
    }

    std::sort(sorted_probs.begin(), sorted_probs.end(),
        [](std::pair<size_t, float> a, std::pair<size_t, float> b) { return a.second > b.second; });

    std::ostringstream probs_sstream;
    for (size_t i = 0; i < sorted_probs.size() && i < 10; ++i) {
      probs_sstream << std::left << std::setw(7) << sorted_probs[i].first << " " << sorted_probs[i].second
                    << std::endl;
    }

    std::cout << "Image: " << image_path << std::endl
              << "Top 10 results:" << std::endl
              << std::endl
              << "classid probability" << std::endl
              << "------- -----------" << std::endl << probs_sstream.str() << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "[EXCEPTION]: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
