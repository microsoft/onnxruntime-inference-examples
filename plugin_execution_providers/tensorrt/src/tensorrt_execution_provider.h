#pragma once

#include "tensorrt_provider_factory.h"
#include "utils/provider_options.h"
#include "tensorrt_execution_provider_info.h"
#include "nv_includes.h"

#include <ctime>
#include <string>
#include <unordered_set>
#include <mutex>
#include <gsl/span>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

using HashValue = uint64_t;
using AllocateFunc = void* (*)(void*, size_t, size_t);
using DestroyFunc = void (*)(void*, void*);

namespace trt_ep {

class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;
  const OrtLogger& ort_default_logger_;
  const OrtApi* ort_api_ = nullptr;

 public:
  TensorrtLogger(const OrtLogger& ort_default_logger,
                 const OrtApi* ort_api,
                 Severity verbosity = Severity::kWARNING)
      : ort_default_logger_{ort_default_logger}, ort_api_{ort_api}, verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= verbosity_) {
      time_t rawtime = std::time(0);
      struct tm stm;
#ifdef _MSC_VER
      gmtime_s(&stm, &rawtime);
#else
      gmtime_r(&rawtime, &stm);
#endif
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               &stm);
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR ? "  ERROR"
                                                                            : severity == Severity::kWARNING ? "WARNING"
                                                                            : severity == Severity::kINFO    ? "   INFO"
                                                                                                             : "UNKNOWN");
      OrtLoggingLevel ort_severity;
      if (severity <= Severity::kERROR) {
        ort_severity = ORT_LOGGING_LEVEL_ERROR;
      } else {
        ort_severity = ORT_LOGGING_LEVEL_WARNING;
      }

      std::string message = "[" + std::string(buf) + " " + std::string(sevstr) + "] " + std::string(msg);

      Ort::ThrowOnError(ort_api_->Logger_LogMessage(&ort_default_logger_,
                                                    ort_severity,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
  }
  void set_level(Severity verbosity) {
    verbosity_ = verbosity;
  }
  Severity get_level() const {
    return verbosity_;
  }
};

namespace tensorrt_ptr {

template <typename T>
using unique_pointer = std::unique_ptr<T, std::default_delete<T>>;
};  // namespace tensorrt_ptr

class OutputAllocator : public nvinfer1::IOutputAllocator {
 public:
#if NV_TENSORRT_MAJOR >= 10
  void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t stream) noexcept override;
#else
  void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override;
#endif
  void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;

  void* getBuffer() {
    return outputPtr;
  }

  std::vector<int64_t>& getOutputShape() {
    return output_shapes;
  }

  uint64_t getSize() {
    return allocated_size;
  }

  ~OutputAllocator() override {
    cudaFree(outputPtr);
  }

 private:
  void* outputPtr{nullptr};
  uint64_t allocated_size = 0;
  std::vector<int64_t> output_shapes;
};

struct TensorrtComputeState {
  uint32_t device_id;
  std::string fused_node_name;
  nvinfer1::IBuilder* builder;
  tensorrt_ptr::unique_pointer<nvonnxparser::IParser>* parser = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext>* context = nullptr;
  std::unique_ptr<nvinfer1::INetworkDefinition>* network = nullptr;
  std::vector<std::unordered_map<std::string, size_t>> input_info;
  std::vector<std::unordered_map<std::string, size_t>> output_info;
  std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> input_shape_ranges;
  std::mutex* tensorrt_mu_ptr = nullptr;
  std::string compute_capability;
  size_t max_workspace_size = 1 << 30;  // 1GB;
  bool fp16_enable = false;
  bool int8_enable = false;
  bool int8_calibration_cache_available = false;
  bool dla_enable = false;
  int dla_core = 0;
  std::string trt_node_name_with_precision;
  bool engine_cache_enable = false;
  std::string engine_cache_path;
  nvinfer1::IRuntime* runtime = nullptr;
  std::vector<nvinfer1::IOptimizationProfile*> profiles;
  bool context_memory_sharing_enable = false;
  size_t* max_context_mem_size_ptr = nullptr;
  AllocatorUniquePtr<void>* context_memory = nullptr;
  std::unordered_map<std::string, float> dynamic_range_map;
  bool engine_decryption_enable = false;
  int (*engine_decryption)(const char*, char*, size_t*) = nullptr;
  int (*engine_encryption)(const char*, char*, size_t) = nullptr;
  bool timing_cache_enable = true;
  std::string timing_cache_path;
  bool force_timing_cache = false;
  bool detailed_build_log = false;
  bool build_heuristics_enable = false;
  bool sparsity_enable = false;
  int builder_optimization_level = 3;
  int auxiliary_streams = -1;
  bool filter_tactic_sources = false;
  nvinfer1::TacticSources tactic_sources;
  bool cuda_graph_enable = false;
  bool weight_stripped_engine_enable = false;
  bool weight_stripped_engine_refit = false;
  char* model_path;
  std::string onnx_model_folder_path;
  const void* onnx_model_bytestream;
  size_t onnx_model_bytestream_size;
  const void* onnx_external_data_bytestream;
  size_t onnx_external_data_bytestream_size;
  std::string cache_prefix;
  std::string cache_suffix;
  bool engine_hw_compatible = false;
  bool sync_stream_after_enqueue = true;
};

// Minimum information to construct kernel function state for EPContext workflow
struct TensorrtComputeStateForEPContext {
  uint32_t device_id;
  std::string fused_node_name;
  std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext>* context = nullptr;
  std::vector<std::unordered_map<std::string, size_t>> input_info;
  std::vector<std::unordered_map<std::string, size_t>> output_info;
  bool context_memory_sharing_enable = false;
  size_t* max_context_mem_size_ptr = nullptr;
  AllocatorUniquePtr<void>* context_memory = nullptr;
  std::mutex* tensorrt_mu_ptr = nullptr;
  bool sync_stream_after_enqueue = true;
};

using ShapeRangesMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>>;
using DDSOutputAllocatorMap = std::unordered_map<std::string, std::unique_ptr<OutputAllocator>>;
std::string GetWeightRefittedEnginePath(std::string engine_cache_path);

static const std::string k_cc_hw_compatible = "80+";
static const std::string k_ep_ctx_hardware_architecture = "hardware_architecture";
static const std::string k_ep_ctx_onnx_model_filename = "onnx_model_filename";

/// <summary>
///
/// Plugin TensorRT EP implementing OrtEp.
///
/// </summary>
struct TensorrtExecutionProvider : public OrtEp, public ApiPtrs {
  TensorrtExecutionProvider(TensorrtExecutionProviderFactory& factory, const std::string& name,
                            const OrtSessionOptions& session_options,
                            const OrtLogger& logger);
  ~TensorrtExecutionProvider();

  TensorrtExecutionProviderFactory& factory_;
  std::string name_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;

  std::unordered_map<std::string, std::unique_ptr<TensorrtComputeState>> compute_states_;
  std::unordered_map<std::string, std::unique_ptr<TensorrtComputeStateForEPContext>> compute_states_for_ep_context_;

  SubGraphCollection_t GetSupportedList(SubGraphCollection_t supported_nodes_list, int iterations,
                                        const int max_iterations, const OrtGraph* graph, bool* early_termination) const;

  OrtStatus* CreateNodeComputeInfoFromPrecompiledEngine(OrtEp* this_ptr, const OrtGraph* graph,
                                                        const OrtNode* fused_node,
                                                        std::unordered_map<std::string, size_t>& input_map,
                                                        std::unordered_map<std::string, size_t>& output_map,
                                                        OrtNodeComputeInfo** node_compute_info);

  OrtStatus* CreateNodeComputeInfoFromGraph(OrtEp* this_ptr, const OrtGraph* graph, const OrtNode* fused_node,
                                            std::unordered_map<std::string, size_t>& input_map,
                                            std::unordered_map<std::string, size_t>& output_map,
                                            OrtNodeComputeInfo** node_compute_info,
                                            OrtNode** ep_context_node);

  OrtStatus* RefitEngine(std::string onnx_model_filename,
                         std::string& onnx_model_folder_path,
                         std::string& weight_stripped_engine_cath_path,
                         bool path_check,
                         const void* onnx_model_bytestream,
                         size_t onnx_model_bytestream_size,
                         const void* onnx_external_data_bytestream,
                         size_t onnx_external_data_bytestream_size,
                         nvinfer1::ICudaEngine* trt_engine,
                         bool serialize_refitted_engine,
                         bool detailed_build_log);

  std::unordered_map<std::string, DDSOutputAllocatorMap>& GetDDSOutputAllocators() {
    return dds_output_allocator_maps_;
  }

  /**
  Get a unique_lock object to control the concurrency behavior.
  Every api call not in the thread-safe operations(https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading)
  should be protected by a lock when invoked by multiple threads concurrently.
  */
  std::unique_lock<std::mutex> GetApiLock() const;

  std::unordered_map<std::string, std::string> trt_node_name_with_precision_;
  std::unordered_map<std::string, std::unordered_map<std::string, float>> dynamic_range_map_;
  std::unordered_map<std::string, std::string> cache_suffix_;
  bool external_stream_ = false;
  cudaStream_t stream_ = nullptr;

  // The OrtAllocator object will be get during ep compute time
  // and should be kept for the lifetime of TRT EP object.
  OrtAllocator* alloc_ = nullptr;

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;
  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr, OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                               _In_ const OrtMemoryDevice* memory_device,
                                                               _Outptr_ OrtSyncStreamImpl** stream) noexcept;

  mutable TensorrtExecutionProviderInfo info_;
  int max_partition_iterations_ = 1000;
  size_t min_subgraph_size_ = 1;
  size_t max_workspace_size_ = 1 << 30;  // 1GB
  bool fp16_enable_ = false;
  bool int8_enable_ = false;
  bool dla_enable_ = false;
  int dla_core_ = 0;
  bool force_sequential_engine_build_ = false;
  std::string int8_calibration_cache_name_;
  bool int8_calibration_cache_available_ = false;
  bool int8_use_native_tensorrt_calibration_table_ = false;
  bool dump_subgraphs_ = false;
  bool engine_cache_enable_ = false;
  bool weight_stripped_engine_enable_ = false;
  bool weight_stripped_engine_refit_ = false;
  std::string onnx_model_folder_path_;
  const void* onnx_model_bytestream_;
  size_t onnx_model_bytestream_size_;
  const void* onnx_external_data_bytestream_ = nullptr;
  size_t onnx_external_data_bytestream_size_ = 0;
  bool build_heuristics_enable_ = false;
  bool sparsity_enable_ = false;
  int builder_optimization_level_ = 3;
  int auxiliary_streams_ = -1;
  std::string tactic_sources_;
  std::string global_cache_path_, cache_path_, engine_decryption_lib_path_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
  std::mutex tensorrt_mu_;
  int device_id_;
  std::string compute_capability_;
  bool context_memory_sharing_enable_ = false;
  bool layer_norm_fp32_fallback_ = false;
  size_t max_ctx_mem_size_ = 0;
  AllocatorUniquePtr<void> context_memory_ = nullptr;
  mutable char model_path_[4096] = {};  // Reserved for max path length
  bool engine_decryption_enable_ = false;
  int (*engine_decryption_)(const char*, char*, size_t*) = nullptr;
  int (*engine_encryption_)(const char*, char*, size_t) = nullptr;
  bool timing_cache_enable_ = false;
  bool force_timing_cache_match_ = false;
  bool detailed_build_log_ = false;
  bool cuda_graph_enable_ = false;
  std::string cache_prefix_;
  bool engine_hw_compatible_ = false;
  std::string op_types_to_exclude_;

  // For create/dump EP context node model
  bool dump_ep_context_model_ = false;
  std::string ep_context_file_path_;
  int ep_context_embed_mode_ = 0;
  std::string ctx_model_path_;
  std::string ep_cache_context_attr_;
  std::string engine_cache_relative_path_to_context_model_dir_;

  OrtGraph* ep_ctx_graph_ = nullptr;
  std::vector<const char*> extra_attr_keys_;
  std::vector<const char*> extra_attr_values_;

  std::unordered_set<std::string> control_flow_op_set_ = {"If", "Loop", "Scan"};

  mutable std::unique_ptr<nvinfer1::IBuilder> builder_;

  // Following maps that hold TRT objects will be accessible by different threads if ORT is using multithreading.
  // In general, TensorRT objects are not thread safe; accesses to an object from different threads must be serialized by the client.
  // But there are still some thread safe operations, please see here https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
  // For those non thread safe operations, TRT EP uses (1) lock_guard or (2) PerThreadContext to make sure synchronization.
  std::unordered_map<std::string, tensorrt_ptr::unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::IBuilder>> builders_;
  std::unordered_map<std::string, std::unique_ptr<nvinfer1::INetworkDefinition>> networks_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> input_info_;
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> output_info_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_min_shapes_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_max_shapes_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_opt_shapes_;
  std::unordered_map<std::string, ShapeRangesMap> input_shape_ranges_;  // The profile shape ranges that the engine is built with
  std::unordered_map<std::string, std::vector<nvinfer1::IOptimizationProfile*>> profiles_;
  std::unordered_map<std::string, DDSOutputAllocatorMap> dds_output_allocator_maps_;

  // TODO: Add support for external cudnn and cublas.
  // for external stream, we need to create its cudnn/cublass handle before cuda EP enable cuda graph capture
  // cudnnHandle_t external_cudnn_handle_ = nullptr;
  // cublasHandle_t external_cublas_handle_ = nullptr;

  // Call cudaStreamSynchronize() after TRT enqueueV3()
  mutable bool sync_stream_after_enqueue_ = true;

  // TODO: Add support for CUDA graph for plugin ep.
  /*
  CUDAGraph cuda_graph_;
  bool is_graph_captured_ = false;
  int regular_run_count_before_graph_capture_ = 0;
  // There is chance (currently only happens in CUDA EP) that the second regular run allocates GPU memory for causes like:
  // (1) memory pattern is enabled. (2) arena allocation for stream.
  // Since no GPU memory allocation is allowed during graph capturing, we need at least two regular runs
  // to allocate enough memory in Arena before graph capturing.
  const int min_num_runs_before_cuda_graph_capture_ = 1;  // required min regular runs before graph capture for the necessary memory allocations.
  */

  bool IsGraphCaptureAllowed() const { return false; };

  nvinfer1::IBuilder* GetBuilder(TensorrtLogger& trt_logger) const;

  /**Check whether all the nodes of the graph are assigned to specific ep*/
  bool AllNodesAssignedToSpecificEP(const OrtGraph* graph, const std::string& provider_type) const;

  /**Check the graph is the subgraph of control flow op*/
  bool IsSubGraphOfControlFlowOp(const OrtGraph* graph) const;

  /**Check whether all the nodes of subgraph are supported*/
  bool IsSubGraphFullySupported(const OrtGraph* graph, SubGraphCollection_t supported_nodes_vector) const;
};

/// <summary>
///
/// Plugin TensorRT EP OrtNodeComputeInfo that represents the computation function for a compiled OrtGraph.
///
/// </summary>
struct TRTEpNodeComputeInfo : OrtNodeComputeInfo {
  explicit TRTEpNodeComputeInfo(TensorrtExecutionProvider& ep);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr, OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  TensorrtExecutionProvider& ep;
};

struct TRTEpEpContextNodeComputeInfo : OrtNodeComputeInfo {
  explicit TRTEpEpContextNodeComputeInfo(TensorrtExecutionProvider& ep);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr, OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  TensorrtExecutionProvider& ep;
};
}  // namespace trt_ep
