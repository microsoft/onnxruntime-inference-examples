// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include <opwrapper_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <openvino/openvino.hpp>
#include <string>

struct KernelOpenVINO {
  KernelOpenVINO(const OrtApi& api, const OrtKernelInfo* info, const char* op_name);

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  ov::CompiledModel compiled_model_;
  ov::OutputVector ov_inputs_;
  ov::OutputVector ov_outputs_;
  std::string weights_;
  std::string device_type_;
};

struct CustomOpOpenVINO : Ort::CustomOpBase<CustomOpOpenVINO, KernelOpenVINO> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t index) const;
  const char* GetExecutionProviderType() const;
};