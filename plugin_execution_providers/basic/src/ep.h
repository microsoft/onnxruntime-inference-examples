// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

class MulKernel;
class BasicPluginEpFactory;

struct FloatInitializer {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

/// <summary>
/// Basic plugin EP.
/// Can only compile/execute a single Mul node.
/// </summary>
class BasicPluginEp : public OrtEp {
 public:
  struct Config {
    // EP configs (typically extracted from OrtSessionOptions or OrtHardwareDevice(s))
  };

  BasicPluginEp(BasicPluginEpFactory& factory, const Config& config, const OrtLogger& logger);
  ~BasicPluginEp();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }

  MulKernel* FindKernelForFusedNode(const std::string& fused_node_name);

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;

  OrtStatus* SaveConstantInitializers(const OrtGraph* graph);

  Config config_{};
  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtModelEditorApi& model_editor_api_;
  std::string name_;
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<MulKernel>> kernels_;
  std::unordered_map<std::string, FloatInitializer> float_initializers_;
};
