// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <string>

#ifdef _WIN32
#include <objbase.h>  // MultiByteToWideChar
#else
#include <dlfcn.h>  // dlclose
#endif

template <size_t N>
int64_t GetShapeSize(const std::array<int64_t, N>& shape) {
  int64_t size = 1;

  for (auto dim : shape) {
    size *= dim;
  }

  return size;
}

#ifdef _WIN32
std::wstring ConvertString(std::string_view str);
#endif

void CleanUpCustomOpLib(void* lib_handle);

void ConvertHWCToCHW(std::vector<float>& output, const uint8_t* input, size_t width, size_t height, size_t num_colors,
                     bool normalize = true);

void Softmax(std::vector<float>& output, const float* inputs, size_t num_inputs);