// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>  //#incldue "core/providers/cuda/cuda_pch.h"

#include "tensorrt_execution_provider_info.h"
#include "provider_options_utils.h"
#include "cuda/cuda_common.h"
#include "ep_utils.h"

namespace tensorrt {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kHasUserComputeStream = "has_user_compute_stream";
constexpr const char* kUserComputeStream = "user_compute_stream";
constexpr const char* kMaxPartitionIterations = "trt_max_partition_iterations";
constexpr const char* kMinSubgraphSize = "trt_min_subgraph_size";
constexpr const char* kMaxWorkspaceSize = "trt_max_workspace_size";
constexpr const char* kFp16Enable = "trt_fp16_enable";
constexpr const char* kInt8Enable = "trt_int8_enable";
constexpr const char* kInt8CalibTable = "trt_int8_calibration_table_name";
constexpr const char* kInt8UseNativeCalibTable = "trt_int8_use_native_calibration_table";
constexpr const char* kDLAEnable = "trt_dla_enable";
constexpr const char* kDLACore = "trt_dla_core";
constexpr const char* kDumpSubgraphs = "trt_dump_subgraphs";
constexpr const char* kEngineCacheEnable = "trt_engine_cache_enable";
constexpr const char* kEngineCachePath = "trt_engine_cache_path";
constexpr const char* kWeightStrippedEngineEnable = "trt_weight_stripped_engine_enable";
constexpr const char* kOnnxModelFolderPath = "trt_onnx_model_folder_path";
constexpr const char* kEngineCachePrefix = "trt_engine_cache_prefix";
constexpr const char* kDecryptionEnable = "trt_engine_decryption_enable";
constexpr const char* kDecryptionLibPath = "trt_engine_decryption_lib_path";
constexpr const char* kForceSequentialEngineBuild = "trt_force_sequential_engine_build";
// add new provider option name here.
constexpr const char* kContextMemorySharingEnable = "trt_context_memory_sharing_enable";
constexpr const char* kLayerNormFP32Fallback = "trt_layer_norm_fp32_fallback";
constexpr const char* kTimingCacheEnable = "trt_timing_cache_enable";
constexpr const char* kTimingCachePath = "trt_timing_cache_path";
constexpr const char* kForceTimingCacheMatch = "trt_force_timing_cache";
constexpr const char* kDetailedBuildLog = "trt_detailed_build_log";
constexpr const char* kBuildHeuristics = "trt_build_heuristics_enable";
constexpr const char* kSparsityEnable = "trt_sparsity_enable";
constexpr const char* kBuilderOptimizationLevel = "trt_builder_optimization_level";
constexpr const char* kAuxiliaryStreams = "trt_auxiliary_streams";
constexpr const char* kTacticSources = "trt_tactic_sources";
constexpr const char* kExtraPluginLibPaths = "trt_extra_plugin_lib_paths";
constexpr const char* kProfilesMinShapes = "trt_profile_min_shapes";
constexpr const char* kProfilesMaxShapes = "trt_profile_max_shapes";
constexpr const char* kProfilesOptShapes = "trt_profile_opt_shapes";
constexpr const char* kCudaGraphEnable = "trt_cuda_graph_enable";
constexpr const char* kEpContextEmbedMode = "trt_ep_context_embed_mode";
constexpr const char* kEpContextFilePath = "trt_ep_context_file_path";
constexpr const char* kDumpEpContextModel = "trt_dump_ep_context_model";
constexpr const char* kEngineHwCompatible = "trt_engine_hw_compatible";
constexpr const char* kONNXBytestream = "trt_onnx_bytestream";
constexpr const char* kONNXBytestreamSize = "trt_onnx_bytestream_size";
constexpr const char* kExternalDataBytestream = "trt_external_data_bytestream";
constexpr const char* kExternalDataBytestreamSize = "trt_external_data_bytestream_size";
constexpr const char* kOpTypesToExclude = "trt_op_types_to_exclude";

}  // namespace provider_option_names
}  // namespace tensorrt

TensorrtExecutionProviderInfo TensorrtExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  TensorrtExecutionProviderInfo info{};

  void* user_compute_stream = nullptr;
  void* onnx_bytestream = nullptr;
  void* external_data_bytestream = nullptr;
  THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              tensorrt::provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> OrtStatus* {
                RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                int num_devices{};
                CUDA_RETURN_IF_ERROR(cudaGetDeviceCount(&num_devices));
                RETURN_IF_NOT(
                    0 <= info.device_id && info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return nullptr;
              })
          .AddAssignmentToReference(tensorrt::provider_option_names::kMaxPartitionIterations, info.max_partition_iterations)
          .AddAssignmentToReference(tensorrt::provider_option_names::kHasUserComputeStream, info.has_user_compute_stream)
          .AddValueParser(
              tensorrt::provider_option_names::kUserComputeStream,
              [&user_compute_stream](const std::string& value_str) -> OrtStatus* {
                size_t address;
                RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                user_compute_stream = reinterpret_cast<void*>(address);
                return nullptr;
              })
          .AddAssignmentToReference(tensorrt::provider_option_names::kMinSubgraphSize, info.min_subgraph_size)
          .AddAssignmentToReference(tensorrt::provider_option_names::kMaxWorkspaceSize, info.max_workspace_size)
          .AddAssignmentToReference(tensorrt::provider_option_names::kFp16Enable, info.fp16_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8Enable, info.int8_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8CalibTable, info.int8_calibration_table_name)
          .AddAssignmentToReference(tensorrt::provider_option_names::kInt8UseNativeCalibTable, info.int8_use_native_calibration_table)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDLAEnable, info.dla_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDLACore, info.dla_core)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDumpSubgraphs, info.dump_subgraphs)
          .AddAssignmentToReference(tensorrt::provider_option_names::kEngineCacheEnable, info.engine_cache_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kEngineCachePath, info.engine_cache_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kWeightStrippedEngineEnable, info.weight_stripped_engine_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kOnnxModelFolderPath, info.onnx_model_folder_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kEngineCachePrefix, info.engine_cache_prefix)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDecryptionEnable, info.engine_decryption_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDecryptionLibPath, info.engine_decryption_lib_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kForceSequentialEngineBuild, info.force_sequential_engine_build)
          .AddAssignmentToReference(tensorrt::provider_option_names::kContextMemorySharingEnable, info.context_memory_sharing_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kLayerNormFP32Fallback, info.layer_norm_fp32_fallback)
          .AddAssignmentToReference(tensorrt::provider_option_names::kTimingCacheEnable, info.timing_cache_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kTimingCachePath, info.timing_cache_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kForceTimingCacheMatch, info.force_timing_cache)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDetailedBuildLog, info.detailed_build_log)
          .AddAssignmentToReference(tensorrt::provider_option_names::kBuildHeuristics, info.build_heuristics_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kSparsityEnable, info.sparsity_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kBuilderOptimizationLevel, info.builder_optimization_level)
          .AddAssignmentToReference(tensorrt::provider_option_names::kAuxiliaryStreams, info.auxiliary_streams)
          .AddAssignmentToReference(tensorrt::provider_option_names::kTacticSources, info.tactic_sources)
          .AddAssignmentToReference(tensorrt::provider_option_names::kExtraPluginLibPaths, info.extra_plugin_lib_paths)
          .AddAssignmentToReference(tensorrt::provider_option_names::kProfilesMinShapes, info.profile_min_shapes)
          .AddAssignmentToReference(tensorrt::provider_option_names::kProfilesMaxShapes, info.profile_max_shapes)
          .AddAssignmentToReference(tensorrt::provider_option_names::kProfilesOptShapes, info.profile_opt_shapes)
          .AddAssignmentToReference(tensorrt::provider_option_names::kCudaGraphEnable, info.cuda_graph_enable)
          .AddAssignmentToReference(tensorrt::provider_option_names::kDumpEpContextModel, info.dump_ep_context_model)
          .AddAssignmentToReference(tensorrt::provider_option_names::kEpContextFilePath, info.ep_context_file_path)
          .AddAssignmentToReference(tensorrt::provider_option_names::kEpContextEmbedMode, info.ep_context_embed_mode)
          .AddAssignmentToReference(tensorrt::provider_option_names::kEngineHwCompatible, info.engine_hw_compatible)
          .AddValueParser(
              tensorrt::provider_option_names::kONNXBytestream,
              [&onnx_bytestream](const std::string& value_str) -> OrtStatus* {
                size_t address;
                RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                onnx_bytestream = reinterpret_cast<void*>(address);
                return nullptr;
              })
          .AddAssignmentToReference(tensorrt::provider_option_names::kONNXBytestreamSize, info.onnx_bytestream_size)
          .AddValueParser(
              tensorrt::provider_option_names::kExternalDataBytestream,
              [&external_data_bytestream](const std::string& value_str) -> OrtStatus* {
                size_t address;
                RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                external_data_bytestream = reinterpret_cast<void*>(address);
                return nullptr;
              })
          .AddAssignmentToReference(tensorrt::provider_option_names::kExternalDataBytestreamSize, info.external_data_bytestream_size)
          .AddAssignmentToReference(tensorrt::provider_option_names::kOpTypesToExclude, info.op_types_to_exclude)
          .Parse(options));  // add new provider option here.

  info.user_compute_stream = user_compute_stream;
  info.has_user_compute_stream = (user_compute_stream != nullptr);
  info.onnx_bytestream = onnx_bytestream;
  info.external_data_bytestream = external_data_bytestream;
  return info;
}