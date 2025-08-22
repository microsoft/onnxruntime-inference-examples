// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "tensorrt_execution_provider.h"
#include "ep_utils.h"
// #include "nv_includes.h"

#include <string>
#include <filesystem>
#include <memory>
#include <gsl/span>

class EPContextNodeHelper : public ApiPtrs {
 public:
  EPContextNodeHelper(TensorrtExecutionProvider& ep,
                      const OrtGraph* graph,
                      const OrtNode* fused_node)
      : ApiPtrs{static_cast<const ApiPtrs&>(ep)}, graph_(graph), fused_node_(fused_node) {}

  static bool GraphHasCtxNode(const OrtGraph* graph, const OrtApi& ort_api);

  OrtStatus* CreateEPContextNode(const std::string& engine_cache_path,
                                 char* engine_data,
                                 size_t size,
                                 const int64_t embed_mode,
                                 const std::string& compute_capability,
                                 const std::string& onnx_model_path,
                                 OrtNode** ep_context_node);

 private:
  const OrtGraph* graph_ = nullptr;
  const OrtNode* fused_node_ = nullptr;
};

class EPContextNodeReader : public ApiPtrs {
 public:
  EPContextNodeReader(TensorrtExecutionProvider& ep,
                      std::unique_ptr<nvinfer1::ICudaEngine>* trt_engine,
                      nvinfer1::IRuntime* trt_runtime,
                      std::string ep_context_model_path,
                      std::string compute_capability,
                      bool weight_stripped_engine_refit,
                      std::string onnx_model_folder_path,
                      const void* onnx_model_bytestream,
                      size_t onnx_model_bytestream_size,
                      const void* onnx_external_data_bytestream,
                      size_t onnx_external_data_bytestream_size,
                      bool detailed_build_log)
      : ApiPtrs{static_cast<const ApiPtrs&>(ep)},
        trt_engine_(trt_engine),
        trt_runtime_(trt_runtime),
        ep_context_model_path_(ep_context_model_path),
        compute_capability_(compute_capability),
        weight_stripped_engine_refit_(weight_stripped_engine_refit),
        onnx_model_folder_path_(onnx_model_folder_path),
        onnx_model_bytestream_(onnx_model_bytestream),
        onnx_model_bytestream_size_(onnx_model_bytestream_size),
        onnx_external_data_bytestream_(onnx_external_data_bytestream),
        onnx_external_data_bytestream_size_(onnx_external_data_bytestream_size),
        detailed_build_log_(detailed_build_log) {
  }

  // bool ValidateEPCtxNode(const OrtGraph& graph);

  OrtStatus* GetEpContextFromGraph(const OrtGraph& graph);

 private:
  std::unique_ptr<nvinfer1::ICudaEngine>* trt_engine_;
  nvinfer1::IRuntime* trt_runtime_;
  std::string ep_context_model_path_;  // If using context model, it implies context model and engine cache is in the same directory
  std::string compute_capability_;
  bool weight_stripped_engine_refit_;
  std::string onnx_model_folder_path_;
  const void* onnx_model_bytestream_;
  size_t onnx_model_bytestream_size_;
  const void* onnx_external_data_bytestream_;
  size_t onnx_external_data_bytestream_size_;
  bool detailed_build_log_;
};  // TRTCacheModelHandler
