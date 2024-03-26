// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <cassert>
#include <cmath>

#ifdef _WIN32
std::wstring ConvertString(std::string_view str) {
  int str_len = static_cast<int>(str.size());
  int size = MultiByteToWideChar(CP_UTF8, 0, str.data(), str_len, NULL, 0);  // Query size.

  std::wstring wide_str(size, 0);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), str_len, &wide_str[0], size);

  return wide_str;
}
#endif

void CleanUpCustomOpLib(void* lib_handle) {
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(lib_handle));
#else
  dlclose(lib_handle);
#endif
}

void ConvertHWCToCHW(std::vector<float>& output, const uint8_t* input, size_t width, size_t height,
                     size_t num_colors, bool normalize) {
  const size_t stride = width * height;
  const size_t total_size = stride * num_colors;

  output.resize(total_size);

  for (size_t i = 0; i < stride; ++i) {
    for (size_t c = 0; c < num_colors; ++c) {
      const size_t out_index = (c * stride) + i;
      const size_t inp_index = (i * num_colors) + c;

      assert(out_index < output.size());
      assert(inp_index < total_size);

      output[out_index] = normalize ? input[inp_index] / 255.0f : input[inp_index];
    }
  }
}

void Softmax(std::vector<float>& output, const float* inputs, size_t num_inputs) {
  output.resize(num_inputs);

  float exp_sum = 0.0f;

  for (size_t i = 0; i < num_inputs; ++i) {
    float exp_val = std::expf(inputs[i]);

    exp_sum += exp_val;
    output[i] = exp_val;
  }

  for (auto& prob : output) {
    prob = prob / exp_sum;
  }
}
