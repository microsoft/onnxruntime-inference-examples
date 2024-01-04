// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

#include "basic_utils.h"

bool GetTensorElemDataSize(ONNXTensorElementDataType data_type, size_t& size);

struct IOInfo {
  IOInfo() = default;
  IOInfo(IOInfo&& other) = default;
  IOInfo(const IOInfo& other) = default;

  IOInfo& operator=(const IOInfo& other) = default;
  IOInfo& operator=(IOInfo&& other) = default;

  static bool Init(IOInfo& io_info, const char* name, ONNXTensorElementDataType data_type, std::vector<int64_t> shape);

  friend bool operator==(const IOInfo& l, const IOInfo& r) {
    if (l.name != r.name || l.data_type != r.data_type || l.shape.size() != r.shape.size()) {
      return false;
    }

    for (size_t i = 0; i < l.shape.size(); i++) {
      if (l.shape[i] != r.shape[i]) {
        return false;
      }
    }

    return true;
  }

  friend bool operator!=(const IOInfo& l, const IOInfo& r) { return !(l == r); }

  std::string name;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t total_data_size = 0;
};

struct ModelIOInfo {
  ModelIOInfo() = default;
  ModelIOInfo(ModelIOInfo&& other) = default;
  ModelIOInfo(const ModelIOInfo& other) = default;

  ModelIOInfo& operator=(const ModelIOInfo& other) = default;
  ModelIOInfo& operator=(ModelIOInfo&& other) = default;

  friend bool operator==(const ModelIOInfo& l, const ModelIOInfo& r) {
    return l.inputs == r.inputs && l.outputs == r.outputs;
  }

  friend bool operator!=(const ModelIOInfo& l, const ModelIOInfo& r) { return !(l == r); }

  static bool Init(ModelIOInfo& model_info, Ort::ConstSession session);

  size_t GetTotalInputSize() const;
  size_t GetTotalOutputSize() const;

  std::vector<IOInfo> inputs;
  std::vector<IOInfo> outputs;
};

AccMetrics ComputeAccuracyMetric(Ort::ConstValue ort_output, Span<const char> raw_expected_output,
                                 const IOInfo& output_info);
