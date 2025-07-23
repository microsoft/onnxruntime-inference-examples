// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <filesystem>

#include "ep_utils.h"
#include "onnx_ctx_model_helper.h"
#include "onnx/onnx_pb.h"

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
  RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

  for (size_t i = 0; i < num_nodes; ++i) {
    auto node = nodes[i];

    const char* op_type = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetOperatorType(node, &op_type));
    if (node != nullptr && std::string(op_type) == "EPContext") {
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

  RETURN_IF_ERROR(ort_api.CreateOpAttr("embed_mode", &embed_mode, sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[0]));

  std::string engine_data_str = "";
  if (embed_mode) {
    if (size > 0) {
      engine_data_str.assign(engine_data, size);
    }
    RETURN_IF_ERROR(
        ort_api.CreateOpAttr("ep_cache_context", engine_data_str.c_str(), engine_data_str.size(), ORT_OP_ATTR_STRING, &attributes[1]));
  } else {
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_cache_context", engine_cache_path.c_str(), engine_cache_path.size(), ORT_OP_ATTR_STRING, &attributes[1]));
  }

 
  ort_api.CreateOpAttr("hardware_architecture", compute_capability.c_str(), compute_capability.size(), ORT_OP_ATTR_STRING, &attributes[2]);
  ort_api.CreateOpAttr("onnx_model_filename", std::filesystem::path(onnx_model_path).filename().string().c_str(), 1,
                       ORT_OP_ATTR_STRING, &attributes[3]);


  RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name, input_names.data(),
                                              input_names.size(), output_names.data(), output_names.size(),
                                              attributes.data(), attributes.size(), ep_context_node));
  
  return nullptr;
}

OrtStatus* EPContextNodeReader::GetEpContextFromGraph(const OrtGraph& graph) {
  /*
  if (!ValidateEPCtxNode(graph)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "It's not a valid EP Context node");
  }
  */

  size_t num_nodes = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(&graph, &num_nodes));

  std::vector<const OrtNode*> nodes(num_nodes);
  RETURN_IF_ERROR(ort_api.Graph_GetNodes(&graph, nodes.data(), nodes.size()));

  auto node = nodes[0];

  size_t num_node_attributes = 0;
  RETURN_IF_ERROR(ort_api.Node_GetNumAttributes(node, &num_node_attributes));

  /*
  std::vector<const OrtOpAttr*> node_attributes(num_node_attributes);
  RETURN_IF_ERROR(ort_api.Node_GetAttributes(node, node_attributes.data(), node_attributes.size()));
  */

  const OrtOpAttr* node_attr = nullptr;
  RETURN_IF_ERROR(ort_api.Node_GetAttributeByName(node, "embed_mode", &node_attr));
  const int64_t embed_mode = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(node_attr)->i();

  // Only make path checks if model not provided as byte buffer
  //bool make_secure_path_checks = !GetModelPath(graph_viewer).empty();
  bool make_secure_path_checks = false;

  if (embed_mode) {
    // Get engine from byte stream.
    node_attr = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetAttributeByName(node, "ep_cache_context", &node_attr));
    const std::string& context_binary = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(node_attr)->s();

    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(const_cast<char*>(context_binary.c_str()),
                                                                                                static_cast<size_t>(context_binary.length())));
    //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Read engine as binary data from \"ep_cache_context\" attribute of ep context node and deserialized it";
    if (!(*trt_engine_)) {
      return ort_api.CreateStatus(ORT_EP_FAIL, "TensorRT EP could not deserialize engine from binary data");
    }

    /*
    if (weight_stripped_engine_refit_) {
      const std::string onnx_model_filename = attrs.at(ONNX_MODEL_FILENAME).s();
      std::string placeholder;
      auto status = TensorrtExecutionProvider::RefitEngine(onnx_model_filename,
                                                           onnx_model_folder_path_,
                                                           placeholder,
                                                           make_secure_path_checks,
                                                           onnx_model_bytestream_,
                                                           onnx_model_bytestream_size_,
                                                           onnx_external_data_bytestream_,
                                                           onnx_external_data_bytestream_size_,
                                                           (*trt_engine_).get(),
                                                           false, // serialize refitted engine to disk
                                                           detailed_build_log_);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
    }
    */
  } else {
    // Get engine from cache file.
    node_attr = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetAttributeByName(node, "ep_cache_context", &node_attr));
    std::string cache_path = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(node_attr)->s();

    /*
    // For security purpose, in the case of running context model, TRT EP won't allow
    // engine cache path to be the relative path like "../file_path" or the absolute path.
    // It only allows the engine cache to be in the same directory or sub directory of the context model.
    if (IsAbsolutePath(cache_path)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "For security purpose, the ep_cache_context attribute should be set with a relative path, but it is an absolute path:  " + cache_path);
    }
    if (IsRelativePathToParentPath(cache_path)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "The file path in ep_cache_context attribute has '..'. For security purpose, it's not allowed to point outside the directory.");
    }

    // The engine cache and context model (current model) should be in the same directory
    std::filesystem::path ctx_model_dir(GetPathOrParentPathOfCtxModel(ep_context_model_path_));
    auto engine_cache_path = ctx_model_dir.append(cache_path);
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] GetEpContextFromGraph engine_cache_path: " + engine_cache_path.string();

    // If it's a weight-stripped engine cache, it needs to be refitted even though the refit flag is not enabled
    if (!weight_stripped_engine_refit_) {
      weight_stripped_engine_refit_ = IsWeightStrippedEngineCache(engine_cache_path);
    }

    // If the serialized refitted engine is present, use it directly without refitting the engine again
    if (weight_stripped_engine_refit_) {
      const std::filesystem::path refitted_engine_cache_path = GetWeightRefittedEnginePath(engine_cache_path.string());
      if (std::filesystem::exists(refitted_engine_cache_path)) {
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " + refitted_engine_cache_path.string() + " exists.";
        engine_cache_path = refitted_engine_cache_path.string();
        weight_stripped_engine_refit_ = false;
      }
    }
    */

    std::filesystem::path engine_cache_path(cache_path);
    if (!std::filesystem::exists(engine_cache_path)) {
      std::string error_msg =
          "TensorRT EP can't find engine cache: " + engine_cache_path.string() +
          ". Please make sure engine cache is in the same directory or sub-directory of context model.";
      return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    std::ifstream engine_file(engine_cache_path.string(), std::ios::binary | std::ios::in);
    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engine_buf{new char[engine_size]};
    engine_file.read((char*)engine_buf.get(), engine_size);
    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
    if (!(*trt_engine_)) {
      std::string error_msg = "TensorRT EP could not deserialize engine from cache: " + engine_cache_path.string();
      return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }
    // LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path.string();

    /*
    if (weight_stripped_engine_refit_) {
      const std::string onnx_model_filename = attrs.at(ONNX_MODEL_FILENAME).s();
      std::string weight_stripped_engine_cache = engine_cache_path.string();
      auto status = TensorrtExecutionProvider::RefitEngine(onnx_model_filename,
                                                           onnx_model_folder_path_,
                                                           weight_stripped_engine_cache,
                                                           make_secure_path_checks,
                                                           onnx_model_bytestream_,
                                                           onnx_model_bytestream_size_,
                                                           onnx_external_data_bytestream_,
                                                           onnx_external_data_bytestream_size_,
                                                           (*trt_engine_).get(),
                                                           true, // serialize refitted engine to disk
                                                           detailed_build_log_);
      if (status != Status::OK()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, status.ErrorMessage());
      }
    }
    */
  }
  return nullptr;
}

/*
 * Get the weight-refitted engine cache path from a weight-stripped engine cache path
 *
 * Weight-stipped engine:
 * An engine with weights stripped and its size is smaller than a regualr engine.
 * The cache name of weight-stripped engine is TensorrtExecutionProvider_TRTKernel_XXXXX.stripped.engine
 *
 * Weight-refitted engine:
 * An engine that its weights have been refitted and it's simply a regular engine.
 * The cache name of weight-refitted engine is TensorrtExecutionProvider_TRTKernel_XXXXX.engine
 */
std::string GetWeightRefittedEnginePath(std::string stripped_engine_cache) {
  std::filesystem::path stripped_engine_cache_path(stripped_engine_cache);
  std::string refitted_engine_cache_path = stripped_engine_cache_path.stem().stem().string() + ".engine";
  return refitted_engine_cache_path;
}