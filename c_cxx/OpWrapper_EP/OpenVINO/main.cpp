// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>
#include <opwrapper_cxx_api.h>

#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <objbase.h>

static std::wstring ConvertString(std::string_view str) {
  int str_len = static_cast<int>(str.size());
  int size = MultiByteToWideChar(CP_UTF8, 0, str.data(), str_len, NULL, 0);  // Query size.

  std::wstring wide_str(size, 0);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), str_len, &wide_str[0], size);

  return wide_str;
}
#endif

template <size_t N>
static int64_t GetShapeSize(const std::array<int64_t, N>& shape) {
  int64_t size = 1;

  for (auto dim : shape) {
    size *= dim;
  }

  return size;
}

static void PrintUsage(std::ostream& stream, std::string_view prog_name) {
  stream << "Usage: " << prog_name << " [OPTIONS] <onnx_model_path> <custom_op_lib_path>" << std::endl;
  stream << "OPTIONS:" << std::endl;
  stream << "    -h/--help      Print this help message" << std::endl << std::endl;
}

struct CmdArgs {
  CmdArgs(int argc, char** argv) noexcept : argc_(argc), argv_(argv), index_(0) {}

  [[nodiscard]] bool HasArgs() const { return index_ < argc_; }

  [[nodiscard]] std::string_view NextArg() {
    if (!HasArgs()) {
      throw std::exception("Out-of-bounds access when parsing command-line arguments.");
    }

    return argv_[index_++];
  }

 private:
  int argc_;
  char** argv_;
  int index_;
};

int main(int argc, char** argv) {
  try {
    CmdArgs args(argc, argv);
    std::string_view prog_name = args.NextArg();

#ifdef _WIN32
    const wchar_t* default_model_path = L"data/custom_op_ov_ep_wrapper.onnx";
    const char* default_custom_op_lib_path = "openvino_wrapper.dll";
    std::wstring wide_model_path;
    std::wstring_view model_path;
    std::string_view custom_op_lib_path;
#else
    const char* default_model_path = "data/custom_op_ov_ep_wrapper.onnx";
    const char* default_custom_op_lib_path = "libopenvino_wrapper.so";
    std::string_view model_path;
    std::string_view custom_op_lib_path;
#endif

    // Parse command-line arguments.
    while (args.HasArgs()) {
      std::string_view arg = args.NextArg();

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

    // TODO: Add RegisterCustomOpsLibrary as a method of Ort::SessionOptions.
    void* lib_handle = nullptr;
    Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(static_cast<OrtSessionOptions*>(session_opts),
                                                             custom_op_lib_path.data(), &lib_handle));
    Ort::OpWrapper::ProviderOptions op_options;
    op_options.UpdateOptions({{"device_type", "CPU"}});

    Ort::OpWrapper::AppendExecutionProvider(session_opts, "OpenVINO_EP_Wrapper", op_options);
    Ort::Session session(env, model_path.data(), session_opts);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Setup input
    std::array<int64_t, 4> input_shape{1, 3, 224, 224};
    std::vector<float> input_vals(GetShapeSize(input_shape), 1.0f);
    std::array<const char*, 1> input_names{"data"};
    std::array<Ort::Value, 1> ort_inputs{Ort::Value::CreateTensor<float>(
        memory_info, input_vals.data(), input_vals.size(), input_shape.data(), input_shape.size())};

    // Run session and get output
    std::array<const char*, 1> output_names{"prob"};
    std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(),
                                                      ort_inputs.size(), output_names.data(), output_names.size());

    // Print probabilities.
    Ort::Value& ort_output = ort_outputs[0];

    auto typeshape = ort_output.GetTensorTypeAndShapeInfo();
    const size_t num_probs = typeshape.GetElementCount();
    const float* probs = ort_output.GetTensorData<float>();

    std::ostringstream probs_sstream;
    for (size_t i = 0; i < num_probs; ++i) {
      probs_sstream << i << ": " << probs[i] << std::endl;
    }

    std::cout << "Probabilities:" << std::endl << probs_sstream.str() << std::endl;

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
