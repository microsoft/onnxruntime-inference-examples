// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "tensorrt_execution_provider.h"
#include "ep_utils.h"
#include "nv_includes.h"

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
                                 OrtNode** ep_context_node
                                 ); 

 private:
  const OrtGraph* graph_ = nullptr;
  const OrtNode* fused_node_ = nullptr;
};
