// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <filesystem>

#include "tensorrt_execution_provider_utils.h"
#include "onnx_ctx_model_helper.h"

extern TensorrtLogger& GetTensorrtLogger(bool verbose_log);

/*
 *  Check whether the graph has the EP context node.
 *  The node can contain the precompiled engine info for TRT EP to directly load the engine.
 *
 *  Note: Please see more details about "EPContext" contrib op in contrib_defs.cc
 */
bool EPContextNodeHelper::GraphHasCtxNode(const OrtGraph* graph, const OrtApi& ort_api) {
  size_t num_nodes = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

  std::vector<const OrtNode*> nodes(num_nodes);

  for (size_t i = 0; i < num_nodes; ++i) {
    auto node = nodes[i];

    const char* op_type = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetOperatorType(node, &op_type));
    if (node != nullptr && op_type == "EPContext") {
      return true;
    }
  }
  return false;
}

/*
 * Create EPContext OrtNode from a fused_node
 */
OrtStatus* EPContextNodeHelper::CreateEPContextNode(const std::string& engine_cache_path,
                                                    char* engine_data,
                                                    size_t size,
                                                    const int64_t embed_mode,
                                                    const std::string& compute_capability,
                                                    const std::string& onnx_model_path,
                                                    OrtNode** ep_context_node) {

  // Helper to collect input or output names from an array of OrtValueInfo instances.
  auto collect_input_output_names = [&](gsl::span<const OrtValueInfo* const> value_infos,
                                        std::vector<const char*>& result) -> OrtStatus* {
    size_t num_values = value_infos.size();
    std::vector<const char*> value_names(num_values);

    for (size_t i = 0; i < num_values; ++i) {
      const OrtValueInfo* value_info = value_infos[i];
      RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &value_names[i]));
    }

    result = std::move(value_names);
    return nullptr;
  };

  const char* fused_node_name = nullptr;

  RETURN_IF_ERROR(ort_api.Node_GetName(fused_node_, &fused_node_name));

  size_t num_fused_node_inputs = 0;
  size_t num_fused_node_outputs = 0;
  RETURN_IF_ERROR(ort_api.Node_GetNumInputs(fused_node_, &num_fused_node_inputs));
  RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(fused_node_, &num_fused_node_outputs));

  std::vector<const OrtValueInfo*> fused_node_inputs(num_fused_node_inputs);
  std::vector<const OrtValueInfo*> fused_node_outputs(num_fused_node_outputs);
  RETURN_IF_ERROR(ort_api.Node_GetInputs(fused_node_, fused_node_inputs.data(), fused_node_inputs.size()));
  RETURN_IF_ERROR(ort_api.Node_GetOutputs(fused_node_, fused_node_outputs.data(), fused_node_outputs.size()));

  std::vector<const char*> input_names;
  std::vector<const char*> output_names;

  RETURN_IF_ERROR(collect_input_output_names(fused_node_inputs, /*out*/ input_names));
  RETURN_IF_ERROR(collect_input_output_names(fused_node_outputs, /*out*/ output_names));

  // Create node attributes. The CreateNode() function copies the attributes, so we have to release them.
  std::array<OrtOpAttr*, 4> attributes = {};
  DeferOrtRelease<OrtOpAttr> defer_release_attrs(attributes.data(), attributes.size(), ort_api.ReleaseOpAttr);

  RETURN_IF_ERROR(ort_api.CreateOpAttr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT, &attributes[0]));

  std::string engine_data_str = "";
  if (embed_mode) {
    if (size > 0) {
      engine_data_str.assign(engine_data, size);
    }
    RETURN_IF_ERROR(
        ort_api.CreateOpAttr("ep_cache_context", engine_data_str.c_str(), 1, ORT_OP_ATTR_STRING, &attributes[1]));
  } else {
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_cache_context", engine_cache_path.c_str(), 1, ORT_OP_ATTR_STRING, &attributes[1]));
  }

 
  ort_api.CreateOpAttr("hardware_architecture", compute_capability.c_str(), 1, ORT_OP_ATTR_STRING, &attributes[2]);
  ort_api.CreateOpAttr("onnx_model_filename", std::filesystem::path(onnx_model_path).filename().string().c_str(), 1,
                       ORT_OP_ATTR_STRING, &attributes[3]);


  RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name, input_names.data(),
                                              input_names.size(), output_names.data(), output_names.size(),
                                              attributes.data(), attributes.size(), ep_context_node));
  
  return nullptr;
}
