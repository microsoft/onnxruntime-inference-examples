#include <memory>
#include <fstream>
#include <list>
#include <functional>
#include <iostream>
#include <numeric>
#include <cuda_runtime.h>

#include "onnxruntime_cxx_api.h"

#define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
#include "ort_graph_to_proto.h"

#include "tensorrt_execution_provider_utils.h"
#include "tensorrt_execution_provider.h"
#include "cuda_allocator.h"
#include "onnx_ctx_model_helper.h"
#include "tensorrt_execution_provider_stream_support.h"
#include "onnx/onnx_pb.h"
#include "cuda/unary_elementwise_ops_impl.h"
#include "ep_utils.h"

#ifdef _WIN32
#include <windows.h>
#define LIBTYPE HINSTANCE
#define OPENLIB(libname) LoadLibrary(libname)
#define LIBFUNC(lib, fn) GetProcAddress((lib), (fn))
#else
#include <dlfcn.h>
#define LIBTYPE void*
#define OPENLIB(libname) dlopen((libname), RTLD_LAZY)
#define LIBFUNC(lib, fn) dlsym((lib), (fn))
#endif

const OrtApi* g_ort_api = nullptr;
const OrtEpApi* g_ep_api = nullptr;
const OrtModelEditorApi* g_model_editor_api = nullptr;

namespace ONNX_NAMESPACE {
using int64s = google::protobuf::RepeatedField<int64_t>;
using float32s = google::protobuf::RepeatedField<float>;
using StringStringEntryProtos = google::protobuf::RepeatedPtrField<StringStringEntryProto>;
using TensorProtos = google::protobuf::RepeatedPtrField<TensorProto>;
using TensorShapeProto_Dimensions = google::protobuf::RepeatedPtrField<TensorShapeProto_Dimension>;
using ValueInfoProtos = google::protobuf::RepeatedPtrField<ValueInfoProto>;
using FunctionProtos = google::protobuf::RepeatedPtrField<FunctionProto>;
}  // namespace ONNX_NAMESPACE

namespace trt_ep {

void CUDA_RETURN_IF_ERROR(cudaError_t res) {
  if (res != cudaSuccess) abort();
}

#if NV_TENSORRT_MAJOR >= 10
void* OutputAllocator::reallocateOutputAsync(char const* /*tensorName*/, void* /*currentMemory*/, uint64_t size,
                                             uint64_t /*alignment*/, cudaStream_t /*stream*/) noexcept {
  // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
  // even for empty tensors, so allocate a dummy byte.
  size = std::max(size, static_cast<uint64_t>(1));
  if (size > allocated_size) {
    cudaFree(outputPtr);
    outputPtr = nullptr;
    allocated_size = 0;
    if (cudaMalloc(&outputPtr, size) == cudaSuccess) {
      allocated_size = size;
    }
  }
  // if cudaMalloc fails, returns nullptr.
  return outputPtr;
}
#else
// Only override this method when TensorRT <= 8.6
void* OutputAllocator::reallocateOutput(char const* /*tensorName*/, void* /*currentMemory*/, uint64_t size,
                                        uint64_t /*alignment*/) noexcept {
  // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
  // even for empty tensors, so allocate a dummy byte.
  size = std::max(size, static_cast<uint64_t>(1));
  if (size > allocated_size) {
    cudaFree(outputPtr);
    outputPtr = nullptr;
    allocated_size = 0;
    if (cudaMalloc(&outputPtr, size) == cudaSuccess) {
      allocated_size = size;
    }
  }
  // if cudaMalloc fails, returns nullptr.
  return outputPtr;
}
#endif

void OutputAllocator::notifyShape(char const* /*tensorName*/, nvinfer1::Dims const& dims) noexcept {
  output_shapes.clear();
  output_shapes.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; i++) {
    output_shapes.push_back(dims.d[i]);
  }
}

TensorrtLogger& GetTensorrtLogger(bool verbose_log,
                                  const OrtLogger& ort_default_logger,
                                  const OrtApi* ort_api) {
  const auto log_level = verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING;
  static TensorrtLogger trt_logger(ort_default_logger, ort_api, log_level);
  if (log_level != trt_logger.get_level()) {
    trt_logger.set_level(verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
  }
  return trt_logger;
}

std::unique_lock<std::mutex> TensorrtExecutionProvider::GetApiLock() const {
  static std::mutex singleton;
  return std::unique_lock<std::mutex>(singleton);
}

nvinfer1::IBuilder* TensorrtExecutionProvider::GetBuilder(TensorrtLogger& trt_logger) const {
  if (!builder_) {
    {
      auto lock = GetApiLock();
      builder_ = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
    }
  }
  return builder_.get();
}

template <typename T>
void GetShapeOfShapeTensor(Ort::ConstValue& input_tensor,
                           void* shape_values,
                           int shape_size,
                           cudaStream_t stream) {
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(shape_values,
                                       input_tensor.GetTensorData<T>(),
                                       shape_size * sizeof(T),
                                       cudaMemcpyDeviceToHost,
                                       stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
}

bool ApplyProfileShapesFromProviderOptions(std::vector<nvinfer1::IOptimizationProfile*>& trt_profiles,
                                           nvinfer1::ITensor* input,
                                           std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_min_shapes,
                                           std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_max_shapes,
                                           std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_opt_shapes,
                                           ShapeRangesMap& input_explicit_shape_ranges,
                                           const OrtLogger* logger) {
  if (trt_profiles.size() == 0) {
    std::string message = "[TensorRT EP] Number of optimization profiles should be greater than 0, but it's 0.";
    Ort::ThrowOnError(g_ort_api->Logger_LogMessage(logger,
                                                   OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                   message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    return false;
  }

  const std::string& input_name = input->getName();
  if (profile_min_shapes.find(input_name) == profile_min_shapes.end()) {
    return false;
  }

  if (input_explicit_shape_ranges.find(input_name) == input_explicit_shape_ranges.end()) {
    std::unordered_map<size_t, std::vector<std::vector<int64_t>>> inner_map;
    input_explicit_shape_ranges[input_name] = inner_map;
  }

  std::string message = "[TensorRT EP] Begin to apply profile shapes ...\n" +
                        std::string("[TensorRT EP] Input tensor name is '") + input_name + std::string("', number of profiles found is ") + std::to_string(trt_profiles.size());
  Ort::ThrowOnError(g_ort_api->Logger_LogMessage(logger,
                                                 OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                 message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

  for (size_t i = 0; i < trt_profiles.size(); i++) {
    nvinfer1::Dims dims = input->getDimensions();
    int nb_dims = dims.nbDims;

    auto trt_profile = trt_profiles[i];

    // Shape tensor
    if (input->isShapeTensor()) {
      int shape_size = nb_dims == 0 ? 1 : static_cast<int>(profile_min_shapes[input_name][i].size());
      std::vector<int32_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);

      std::string message = "[TensorRT EP] shape size of this shape tensor is " + std::to_string(shape_size);
      Ort::ThrowOnError(g_ort_api->Logger_LogMessage(logger,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

      for (int j = 0; j < shape_size; j++) {
        auto min_value = profile_min_shapes[input_name][i][j];
        auto max_value = profile_max_shapes[input_name][i][j];
        auto opt_value = profile_opt_shapes[input_name][i][j];
        shapes_min[j] = static_cast<int32_t>(min_value);
        shapes_max[j] = static_cast<int32_t>(max_value);
        shapes_opt[j] = static_cast<int32_t>(opt_value);
        std::string message = "[TensorRT EP] shapes_min.d[" + std::to_string(j) + std::string("] is ") + std::to_string(shapes_min[j]) + std::string("\n") +
                              std::string("[TensorRT EP] shapes_max.d[") + std::to_string(j) + std::string("] is ") + std::to_string(shapes_max[j]) + std::string("\n") +
                              std::string("[TensorRT EP] shapes_opt.d[") + std::to_string(j) + std::string("] is ") + std::to_string(shapes_opt[j]);
        Ort::ThrowOnError(g_ort_api->Logger_LogMessage(logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                       message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

        if (input_explicit_shape_ranges[input_name].find(j) == input_explicit_shape_ranges[input_name].end()) {
          std::vector<std::vector<int64_t>> profile_vector(trt_profiles.size());
          input_explicit_shape_ranges[input_name][j] = profile_vector;
        }
        input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(min_value);
        input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(max_value);
        input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(opt_value);
      }

      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
    }
    // Execution tensor
    else {
      nvinfer1::Dims dims_min, dims_opt, dims_max;
      dims_min.nbDims = nb_dims;
      dims_max.nbDims = nb_dims;
      dims_opt.nbDims = nb_dims;

      std::string message = "[TensorRT EP] number of dimension of this execution tensor is " + std::to_string(nb_dims);
      Ort::ThrowOnError(g_ort_api->Logger_LogMessage(logger,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

      for (int j = 0; j < nb_dims; j++) {
        if (dims.d[j] == -1) {
          auto min_value = profile_min_shapes[input_name][i][j];
          auto max_value = profile_max_shapes[input_name][i][j];
          auto opt_value = profile_opt_shapes[input_name][i][j];
          dims_min.d[j] = static_cast<int32_t>(min_value);
          dims_max.d[j] = static_cast<int32_t>(max_value);
          dims_opt.d[j] = static_cast<int32_t>(opt_value);

          std::string message = "[TensorRT EP] dims_min.d[" + std::to_string(j) + std::string("] is ") + std::to_string(dims_min.d[j]) + std::string("\n") +
                                std::string("[TensorRT EP] dims_max.d[") + std::to_string(j) + std::string("] is ") + std::to_string(dims_max.d[j]) + std::string("\n") +
                                std::string("[TensorRT EP] dims_opt.d[") + std::to_string(j) + std::string("] is ") + std::to_string(dims_opt.d[j]);
          Ort::ThrowOnError(g_ort_api->Logger_LogMessage(logger,
                                                         OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                         message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

          if (input_explicit_shape_ranges[input_name].find(j) == input_explicit_shape_ranges[input_name].end()) {
            std::vector<std::vector<int64_t>> profile_vector(trt_profiles.size());
            input_explicit_shape_ranges[input_name][j] = profile_vector;
          }
          input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(min_value);
          input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(max_value);
          input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(opt_value);
        } else {
          dims_min.d[j] = dims.d[j];
          dims_max.d[j] = dims.d[j];
          dims_opt.d[j] = dims.d[j];
        }
      }

      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
    }
  }
  return true;
}

OrtStatusPtr ApplyProfileShapesFromInputTensorValue(std::vector<nvinfer1::IOptimizationProfile*>& trt_profiles,
                                                    Ort::KernelContext ctx,
                                                    nvinfer1::ITensor* input,
                                                    ShapeRangesMap& shape_ranges,
                                                    const std::unordered_map<std::string, size_t>& input_indexes,
                                                    std::unordered_map<std::string, std::vector<int32_t>>& shape_tensor_values,
                                                    std::unordered_map<std::string, std::vector<int64_t>>& shape_tensor_values_int64,
                                                    cudaStream_t stream,
                                                    bool* engine_update) {
  for (size_t i = 0; i < trt_profiles.size(); i++) {
    const std::string& input_name = input->getName();
    nvinfer1::Dims dims = input->getDimensions();
    int nb_dims = dims.nbDims;

    size_t input_index = 0;
    const auto& iter = input_indexes.find(input_name);
    if (iter != input_indexes.end()) {
      input_index = iter->second;
    }

    auto input_tensor = ctx.GetInput(input_index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shapes = tensor_info.GetShape();
    auto& shape_ranges_per_input = shape_ranges[input_name];

    auto trt_profile = trt_profiles[i];

    // If there are multiple profiles, for second and rest of profiles, simply copy the min/max/opt profile values from the first profile.
    // Following "if statement" won't be executed since TRT EP currently only allows single profile for non-explicit profiles case.
    if (i > 0) {
      if (input->isShapeTensor()) {
        // shape tensor
        int shape_size = nb_dims == 0 ? 1 : static_cast<int>(tensor_shapes[0]);
        std::vector<int32_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);
        for (int j = 0; j < shape_size; j++) {
          shapes_min[j] = *(trt_profiles[0]->getShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN));
          shapes_max[j] = *(trt_profiles[0]->getShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX));
          shapes_opt[j] = *(trt_profiles[0]->getShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT));
        }
        trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
        trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);
        trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
      } else {
        // execution tensor
        nvinfer1::Dims dims_min, dims_opt, dims_max;
        dims_min = trt_profiles[0]->getDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN);
        dims_max = trt_profiles[0]->getDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX);
        dims_opt = trt_profiles[0]->getDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT);
        trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
        trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
        trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
      }
      continue;
    }

    // Create shape profile
    if (input->isShapeTensor()) {
      // Get shape values for shape tensor input
      const auto tensor_type = tensor_info.GetElementType();
      // The shape of the "shape tensor" is either zero dimension (scalar) or 1-dimension
      int shape_size = dims.nbDims == 0 ? 1 : static_cast<int>(tensor_shapes[0]);
      // For setting TRT optimization profile. (Note: the min/opt/max profile values are still int32 even though int64 is supported after TRT 10)
      std::vector<int32_t> values(shape_size);

      switch (tensor_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
          auto buffer = std::make_unique<int32_t[]>(shape_size);
          GetShapeOfShapeTensor<int32_t>(input_tensor, buffer.get(), shape_size, stream);
          shape_tensor_values[input_name].resize(shape_size);
          for (int j = 0; j < shape_size; ++j) {
            shape_tensor_values[input_name][j] = buffer[j];
            values[j] = buffer[j];
          }
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
          auto buffer = std::make_unique<int64_t[]>(shape_size);
          GetShapeOfShapeTensor<int64_t>(input_tensor, buffer.get(), shape_size, stream);
          shape_tensor_values_int64[input_name].resize(shape_size);
          for (int j = 0; j < shape_size; ++j) {
            shape_tensor_values_int64[input_name][j] = buffer[j];
            values[j] = static_cast<int32_t>(buffer[j]);
          }
          break;
        }
        default: {
          return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, std::string("TensorRT shape tensor data type: " + std::to_string(tensor_type) + " not supported.").c_str());
        }
      }

      // Update shape ranges
      std::vector<int32_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);
      int shape_range_size = static_cast<int>(shape_ranges_per_input.size());
      if (shape_size == shape_range_size) {
        // If shape size matches, check/update shape range
        for (int j = 0; j < shape_size; ++j) {
          auto& shape_range = shape_ranges_per_input[j][0];  // only has one profile
          shapes_min[j] = static_cast<int32_t>(shape_range[0]);
          shapes_max[j] = static_cast<int32_t>(shape_range[1]);
          shapes_opt[j] = static_cast<int32_t>(shape_range[2]);

          const auto& tensor_shape_value = values[j];
          // Update shape range lower bound
          if (tensor_shape_value < shape_range[0]) {
            shape_range[0] = tensor_shape_value;
            shapes_min[j] = tensor_shape_value;
            *engine_update = true;
          }
          // Update shape range upper bound
          if (tensor_shape_value > shape_range[1]) {
            shape_range[1] = tensor_shape_value;
            shape_range[2] = tensor_shape_value;
            shapes_max[j] = tensor_shape_value;
            shapes_opt[j] = tensor_shape_value;
            *engine_update = true;
          }
        }
      } else {
        // If shape size doesn't match, initialize shape_range with the new shape value
        shape_ranges_per_input.clear();
        for (int j = 0; j < shape_size; ++j) {
          const auto& tensor_shape_value = values[j];
          std::vector<std::vector<int64_t>> profile_vector;
          std::vector<int64_t> shape_vector{tensor_shape_value, tensor_shape_value, tensor_shape_value};
          profile_vector.push_back(shape_vector);  // only one profile needed
          shape_ranges_per_input[j] = profile_vector;
          shapes_min[j] = tensor_shape_value;
          shapes_opt[j] = tensor_shape_value;
          shapes_max[j] = tensor_shape_value;
        }
        *engine_update = true;
      }

      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);
      trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
    } else {  // Execution tensor
      nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);
      for (int j = 0, end = nb_dims; j < end; ++j) {
        const auto& tensor_shape = tensor_shapes[j];
        if (shape_ranges_per_input.find(j) != shape_ranges_per_input.end()) {
          auto& shape_range = shape_ranges_per_input[j][0];  // only has one profile
          dims_min.d[j] = static_cast<int32_t>(shape_range[0]);
          dims_max.d[j] = static_cast<int32_t>(shape_range[1]);
          dims_opt.d[j] = static_cast<int32_t>(shape_range[2]);

          // Update minimum dimension
          if (tensor_shape < shape_range[0]) {
            shape_range[0] = tensor_shape;
            dims_min.d[j] = static_cast<int32_t>(tensor_shape);
            *engine_update = true;
          }
          // Update maximum dimension
          if (tensor_shape > shape_range[1]) {
            shape_range[1] = tensor_shape;
            shape_range[2] = tensor_shape;
            dims_max.d[j] = static_cast<int32_t>(tensor_shape);
            dims_opt.d[j] = static_cast<int32_t>(tensor_shape);
            *engine_update = true;
          }
        }
      }

      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
      trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
    }
  }
  return nullptr;
}

#define CASE_GET_INPUT_TENSOR(DATA_TYPE, SrcT)                                  \
  case DATA_TYPE: {                                                             \
    auto input_tensor_ptr = input_tensor.GetTensorData<SrcT>();                 \
    if (input_tensor_ptr != nullptr && elem_cnt > 0) {                          \
      data = const_cast<SrcT*>(input_tensor_ptr);                               \
    } else {                                                                    \
      scratch_buffers.push_back(MakeUniquePtrFromOrtAllocator<void>(alloc, 1)); \
      data = scratch_buffers.back().get();                                      \
    }                                                                           \
    break;                                                                      \
  }

#define CASE_GET_CAST_INPUT_TENSOR(DATA_TYPE, SrcT, DstT)                                             \
  case DATA_TYPE: {                                                                                   \
    auto input_tensor_ptr = input_tensor.GetTensorData<SrcT>();                                       \
    if (input_tensor_ptr != nullptr && elem_cnt > 0) {                                                \
      scratch_buffers.push_back(MakeUniquePtrFromOrtAllocator<void>(alloc, elem_cnt * sizeof(DstT))); \
      data = scratch_buffers.back().get();                                                            \
      cuda::Impl_Cast<SrcT, DstT>(stream, input_tensor_ptr, reinterpret_cast<DstT*>(data), elem_cnt); \
    } else {                                                                                          \
      scratch_buffers.push_back(MakeUniquePtrFromOrtAllocator<void>(alloc, 1));                       \
      data = scratch_buffers.back().get();                                                            \
    }                                                                                                 \
    break;                                                                                            \
  }

#define CASE_GET_OUTPUT_TENSOR(DATA_TYPE, SrcT)                                 \
  case DATA_TYPE: {                                                             \
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<SrcT>();        \
    if (output_tensor_ptr != nullptr && elem_cnt > 0) {                         \
      buffers[output_name] = output_tensor_ptr;                                 \
    } else {                                                                    \
      scratch_buffers.push_back(MakeUniquePtrFromOrtAllocator<void>(alloc, 1)); \
      buffers[output_name] = scratch_buffers.back().get();                      \
    }                                                                           \
    break;                                                                      \
  }

#define CASE_GET_CAST_OUTPUT_TENSOR(DATA_TYPE, SrcT, DstT)                                            \
  case DATA_TYPE: {                                                                                   \
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<SrcT>();                              \
    if (output_tensor_ptr != nullptr && elem_cnt > 0) {                                               \
      scratch_buffers.push_back(MakeUniquePtrFromOrtAllocator<void>(alloc, elem_cnt * sizeof(DstT))); \
      buffers[output_name] = scratch_buffers.back().get();                                            \
      output_dim_sizes[i] = static_cast<int>(elem_cnt);                                               \
    } else {                                                                                          \
      scratch_buffers.push_back(MakeUniquePtrFromOrtAllocator<void>(alloc, 1));                       \
      buffers[output_name] = scratch_buffers.back().get();                                            \
      output_dim_sizes[i] = 1;                                                                        \
    }                                                                                                 \
    break;                                                                                            \
  }

#define CASE_COPY_TENSOR(DATA_TYPE, DstT)                                                                                                          \
  case DATA_TYPE: {                                                                                                                                \
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<DstT>();                                                                           \
    if (output_tensor_ptr != nullptr && elem_cnt > 0) {                                                                                            \
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_tensor_ptr, allocator->getBuffer(), elem_cnt * sizeof(DstT), cudaMemcpyDeviceToDevice, stream)); \
    }                                                                                                                                              \
    break;                                                                                                                                         \
  }

#define CASE_CAST_TENSOR(DATA_TYPE, SrcT, DstT)                                                                                                   \
  case DATA_TYPE: {                                                                                                                               \
    auto output_tensor_ptr = output_tensor.GetTensorMutableData<DstT>();                                                                          \
    if (output_tensor_ptr != nullptr && elem_cnt > 0) {                                                                                           \
      cuda::Impl_Cast<SrcT, DstT>(stream, reinterpret_cast<SrcT*>(allocator->getBuffer()), reinterpret_cast<DstT*>(output_tensor_ptr), elem_cnt); \
    }                                                                                                                                             \
    break;                                                                                                                                        \
  }

OrtStatusPtr BindContextInput(Ort::KernelContext& ctx,
                              nvinfer1::ICudaEngine* trt_engine,
                              nvinfer1::IExecutionContext* trt_context,
                              const char* input_name,
                              size_t input_index,
                              std::unordered_map<std::string, std::vector<int32_t>>& shape_tensor_values,
                              std::unordered_map<std::string, std::vector<int64_t>>& shape_tensor_values_int64,
                              std::vector<AllocatorUniquePtr<void>>& scratch_buffers,
                              OrtAllocator* alloc,
                              cudaStream_t stream) {
  try {
    auto input_tensor = ctx.GetInput(input_index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shapes = tensor_info.GetShape();
    const auto tensor_type = tensor_info.GetElementType();
    /*
     * Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
     * For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
     *
     * Examples:<br>
     * [] = 1<br>
     * [1,3,4] = 12<br>
     * [2,0,4] = 0<br>
     * [-1,3,4] = -1<br>
     */
    const auto elem_cnt = tensor_info.GetElementCount();

    if (trt_engine->isShapeInferenceIO(input_name)) {
      // Bind "shape tensor" input buffer

      // The shape of the "shape tensor" is either zero dimension (scalar) or 1-dimension
      int shape_size = trt_engine->getTensorShape(input_name).nbDims == 0 ? 1 : static_cast<int>(tensor_shapes[0]);
      switch (tensor_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
          // get shape tensor value if not present
          if (shape_tensor_values.find(input_name) == shape_tensor_values.end()) {
            auto input = std::make_unique<int32_t[]>(shape_size);
            GetShapeOfShapeTensor<int32_t>(input_tensor, input.get(), shape_size, stream);
            shape_tensor_values[input_name].resize(shape_size);
            for (int i = 0; i < shape_size; ++i) {
              shape_tensor_values[input_name][i] = input[i];
            }
          }

          if (!trt_context->setTensorAddress(input_name, &shape_tensor_values[input_name][0])) {
            std::string error_input_name = input_name;
            std::string error_msg =
                "TensorRT EP failed to call nvinfer1::IExecutionContext::setTensorAddress() for shape input '" +
                error_input_name + "'";
            return g_ort_api->CreateStatus(ORT_EP_FAIL, error_msg.c_str());
          }
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
          // get shape tensor value if not present
          if (shape_tensor_values_int64.find(input_name) == shape_tensor_values_int64.end()) {
            auto input = std::make_unique<int64_t[]>(shape_size);
            GetShapeOfShapeTensor<int64_t>(input_tensor, input.get(), shape_size, stream);
            shape_tensor_values_int64[input_name].resize(shape_size);
            for (int i = 0; i < shape_size; ++i) {
              shape_tensor_values_int64[input_name][i] = input[i];
            }
          }

          if (!trt_context->setTensorAddress(input_name, &shape_tensor_values_int64[input_name][0])) {
            std::string error_input_name = input_name;
            std::string error_msg =
                "TensorRT EP failed to call nvinfer1::IExecutionContext::setTensorAddress() for shape input '" +
                error_input_name + "'";
            return g_ort_api->CreateStatus(ORT_EP_FAIL, error_msg.c_str());
          }
          break;
        }
        default: {
          std::string error_input_name = input_name;
          return g_ort_api->CreateStatus(ORT_EP_FAIL, std::string("The data type of shape tensor should be INT32 or INT64. Please check the data type of " + error_input_name).c_str());
        }
      }
    } else {
      // Set shape for input tensor which is execution tensor
      nvinfer1::Dims dims = trt_context->getTensorShape(input_name);
      int nb_dims = dims.nbDims;
      for (int j = 0, end = nb_dims; j < end; ++j) {
        dims.d[j] = static_cast<int32_t>(tensor_shapes[j]);
      }
      if (!trt_context->setInputShape(input_name, dims)) {
        std::string error_input_name = input_name;
        return g_ort_api->CreateStatus(ORT_EP_FAIL, std::string("TensorRT EP failed to call nvinfer1::IExecutionContext::setInputShape() for input '" + error_input_name + "'").c_str());
      }

      // Bind "execution tensor" input buffer
      //
      // Note: If an engine binding is an empty tensor, it still needs a non-null memory address, and different tensors should have different addresses.
      //       Therefore, in the case of empty tensor, TRT EP always allocates a dummy byte.
      //       https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#empty-tensors
      void* data = nullptr;
      switch (tensor_type) {
        CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
        CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
        CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
        CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
        CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
        CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
#if NV_TENSORRT_MAJOR >= 10
        CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
#else
        // Cast int64 input to int32 input because TensorRT < 10 doesn't support int64
        CASE_GET_CAST_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t, int32_t)
#endif
        // Cast double input to float because TensorRT doesn't support double
        CASE_GET_CAST_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, double, float)
        default: {
          return g_ort_api->CreateStatus(ORT_EP_FAIL, std::string("TensorRT EP input onnx tensor data type: " + std::to_string(tensor_type) + " not supported.").c_str());
        }
      }
      trt_context->setTensorAddress(input_name, data);
    }
  } catch (const Ort::Exception& e) {
    return g_ort_api->CreateStatus(ORT_EP_FAIL, e.what());
  }
  return nullptr;
}

OrtStatusPtr BindContextOutput(Ort::KernelContext& ctx,
                               nvinfer1::IExecutionContext* trt_context,
                               const char* output_name,
                               size_t output_index,
                               size_t output_type,
                               size_t i,
                               std::unordered_map<size_t, Ort::UnownedValue>& output_tensors,
                               std::unordered_map<size_t, int>& output_dim_sizes,
                               DDSOutputAllocatorMap& dds_output_allocator_map,
                               std::vector<AllocatorUniquePtr<void>>& scratch_buffers,
                               OrtAllocator* alloc,
                               std::unordered_map<char const*, void*>& buffers) {
  // Get output shape
  nvinfer1::Dims dims = trt_context->getTensorShape(output_name);
  int nb_dims = dims.nbDims;
  bool is_DDS = false;
  std::vector<int64_t> output_shapes(nb_dims);
  for (int j = 0, end = nb_dims; j < end; ++j) {
    // data-dependent shape
    if (dims.d[j] == -1) {
      is_DDS = true;
      break;
    }
    output_shapes[j] = dims.d[j];
  }

  auto known_DDS = dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end();

  // If the output tensor has data-dependent shape, TRT EP will provide an IOutputAllocator for enqueueV3 to dynamically allocate memory buffer.
  // Once enqueueV3 returns, TRT EP will then bind the output allocation to ORT kernel context output.
  // (Please note that we take strategy A mentioned in https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dynamic-shaped-output,
  //  which we defer allocation until the size is known and don't call IExecution::setTensorAddress)
  //
  // Otherwise, if the shape of the output tensor is known prior to the runtime, ORT will pre-allocate memory buffer for the output tensor for enqueueV3.
  if (is_DDS || known_DDS) {
    if (!known_DDS) {
      auto allocatorPtr = std::make_unique<OutputAllocator>();
      trt_context->setOutputAllocator(output_name, allocatorPtr.get());
      dds_output_allocator_map[output_name] = std::move(allocatorPtr);
    }
  } else {
    try {
      output_tensors[i] = ctx.GetOutput(output_index, output_shapes);
      auto& output_tensor = output_tensors[i];
      const auto elem_cnt = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

      switch (output_type) {
        CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
        CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
        CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
        CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
        CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
        CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
#if NV_TENSORRT_MAJOR >= 10
        CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
#else
        // Allocate int32 CUDA memory for int64 output type because TensorRT < 10 doesn't support int64
        CASE_GET_CAST_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t, int32_t)
#endif
        // Allocate float CUDA memory for double output type because TensorRT doesn't support double
        CASE_GET_CAST_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, double, float)
        default: {
          return g_ort_api->CreateStatus(ORT_EP_FAIL, std::string("TensorRT EP output tensor data type: " + std::to_string(output_type) + " not supported.").c_str());
        }
      }
      trt_context->setTensorAddress(output_name, buffers[output_name]);
    } catch (const Ort::Exception& e) {
      return g_ort_api->CreateStatus(ORT_EP_FAIL, e.what());
    }
  }

  return nullptr;
}

OrtStatusPtr BindKernelOutput(Ort::KernelContext& ctx,
                              const OrtMemoryInfo* /*mem_info*/,
                              DDSOutputAllocatorMap& allocator_map,
                              char const* output_name,
                              size_t output_index,
                              size_t output_type,
                              cudaStream_t stream) {
  try {
    auto allocator = allocator_map[output_name].get();
    auto& shape = allocator->getOutputShape();
    auto output_tensor = ctx.GetOutput(output_index, shape);

    /*
     * Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
     * For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
     *
     * Examples:<br>
     * [] = 1<br>
     * [1,3,4] = 12<br>
     * [2,0,4] = 0<br>
     * [-1,3,4] = -1<br>
     */
    auto elem_cnt = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    /*
     * Copy output data from allocation buffer to ORT kernel context output location or
     * cast (int32 or float) -> (int64 or double) to ORT kernel context output location.
     *
     * Note:
     * 1. If the output tensor is empty tensor (i.e. any of the dimension is 0) which means element count is 0,
     *    TRT EP does not perform cuda memory copy nor cuda cast to prevent overwriting other location that might belong to other tensors.
     * 2. The cudaMemcpyAsync() and cuda::Impl_Cast() (implemented as _UnaryElementWise() in cuda ep) are all async, but we
     *    don't need to explicitly call cudaStreamSynchronize() after those APIs due to CUDA EP and TRT EP uses same stream,
     *    and within the same stream, operations are guaranteed to be executed in order.
     */
    switch (output_type) {
      CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
      CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
      CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
      CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
      CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
      CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
#if NV_TENSORRT_MAJOR >= 10
      CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
#else
      // The allocation buffer holds the int32 output data since TRT doesn't support int64. So, we need to cast the data (int32 -> int64) for ORT kernel output.
//    CASE_CAST_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int32_t, int64_t)
#endif
        // The allocation buffer holds the float output data since TRT doesn't support double. So, we need to cast the data (float -> double) for ORT kernel output.
        //    CASE_CAST_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, float, double)
      default: {
        return g_ort_api->CreateStatus(ORT_EP_FAIL, std::string("TensorRT EP output tensor data type: " + std::to_string(output_type) + " not supported.").c_str());
      }
    }
  } catch (const Ort::Exception& e) {
    return g_ort_api->CreateStatus(ORT_EP_FAIL, e.what());
  }
  return nullptr;
}

bool TensorrtExecutionProvider::AllNodesAssignedToSpecificEP(const OrtGraph* graph, const std::string& provider_type) const {
  size_t num_nodes = 0;
  THROW_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

  // Get all the nodes from the graph
  std::vector<const OrtNode*> nodes(num_nodes);
  THROW_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

  for (const auto node : nodes) {
    const char* ep_name;
    THROW_IF_ERROR(ort_api.Node_GetEpName(node, &ep_name));

    if (std::string(ep_name) != provider_type) {
      return false;
    }
  }

  return num_nodes != 0;
}

// Check the graph is the subgraph of control flow op
bool TensorrtExecutionProvider::IsSubGraphOfControlFlowOp(const OrtGraph* graph) const {
  const OrtNode* parent_node = nullptr;
  THROW_IF_ERROR(ort_api.Graph_GetParentNode(graph, &parent_node));
  if (parent_node) {
    const char* op_type = nullptr;
    THROW_IF_ERROR(ort_api.Node_GetOperatorType(parent_node, &op_type));

    if (control_flow_op_set_.find(std::string(op_type)) != control_flow_op_set_.end()) {
      return true;
    }
  }
  return false;
}

// Check whether all the nodes of subgraph are supported
bool TensorrtExecutionProvider::IsSubGraphFullySupported(const OrtGraph* graph, SubGraphCollection_t supported_nodes_vector) const {
  size_t num_nodes = 0;
  THROW_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

  int number_of_trt_nodes = 0;
  for (const auto& group : supported_nodes_vector) {
    if (!group.first.empty()) {
      number_of_trt_nodes += static_cast<int>(group.first.size());
    }
  }

  return number_of_trt_nodes == num_nodes;
}

SubGraphCollection_t TensorrtExecutionProvider::GetSupportedList(SubGraphCollection_t nodes_vector_input,
                                                                 int iterations, const int max_iterations,
                                                                 const OrtGraph* graph, bool* early_termination) const {
  // Temporarily make all nodes supported
  SubGraphCollection_t nodes_list_output = nodes_vector_input;

  return nodes_list_output;
}

OrtStatus* ORT_API_CALL TensorrtExecutionProvider::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                                     OrtEpGraphSupportInfo* graph_support_info) noexcept {
  TensorrtExecutionProvider* ep = static_cast<TensorrtExecutionProvider*>(this_ptr);
  const OrtApi& ort_api = ep->ort_api;
  auto ort_graph = Ort::ConstGraph(graph);

  size_t num_nodes = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

  // Get all the nodes from the graph
  std::vector<const OrtNode*> nodes(num_nodes);
  RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

  SubGraphCollection_t parser_nodes_vector, supported_nodes_vector;
  bool new_subgraph = true;

  std::unordered_set<std::string> control_flow_op_set = {"If", "Loop", "Scan"};

  // Get pre-excluded op list from provider options
  auto get_exclude_ops_set = [&](std::string node_list_to_exclude) -> std::set<std::string> {
    std::set<std::string> set;
    if (!node_list_to_exclude.empty()) {
      std::stringstream node_list(node_list_to_exclude);
      std::string node;
      while (std::getline(node_list, node, ',')) {
        set.insert(node);
      }
    }
    return set;
  };

  auto exclude_ops_set = get_exclude_ops_set(ep->op_types_to_exclude_);

  /* Iterate all the nodes and exclude the node if:
   *   1. It's a control flow op and its subgraph(s) is not fully TRT eligible.
   *   2. Its op type is in the exclusion list.
   */
  for (size_t index = 0; index < nodes.size(); index++) {
    const OrtNode* node = nodes[index];
    bool supported_node = true;

    /* If current node is control flow op, we take different approach based on following four cases:
     *
     * (1) control flow op is supported by TRT, and its subgraphs are all supported by TRT. Assign this node to TRT.
     * (2) control flow op is supported by TRT, but not all its subgraphs supported by TRT. Don't assign this node to TRT.
     * (3) control flow op is not supported by TRT, but its subgraphs all supported by TRT. Don't assign this node to TRT.
     * (4) control flow op is not supported by TRT, and not all its subgraphs supported by TRT. Don't assign this node to TRT.
     *
     * For cases 2, 3, 4, even though the control flow op is not assigned to TRT, any portion of its subgraphs that can run in TRT will be still fused and assigned to TRT EP.
     */
    const char* op_type = nullptr;
    RETURN_IF_ERROR(ep->ort_api.Node_GetOperatorType(node, &op_type));

    if (control_flow_op_set.find(op_type) != control_flow_op_set.end()) {
      auto supported_control_flow_op = [&](const OrtNode* node) {
        OrtStatus* status = nullptr;
        size_t num_subgraphs = 0;
        RETURN_FALSE_AND_PRINT_IF_ERROR(ort_api.Node_GetNumSubgraphs(node, &num_subgraphs));

        std::vector<const OrtGraph*> node_subgraphs(num_subgraphs);
        RETURN_FALSE_AND_PRINT_IF_ERROR(ort_api.Node_GetSubgraphs(node, node_subgraphs.data(), node_subgraphs.size(), nullptr));

        // Iterate the node's subgraphs
        for (size_t subgraph_idx = 0; subgraph_idx < num_subgraphs; subgraph_idx++) {
          const OrtGraph* subgraph = node_subgraphs[subgraph_idx];

          // Get number of subgraph's nodes
          size_t num_subgraph_nodes = 0;
          RETURN_FALSE_AND_PRINT_IF_ERROR(ort_api.Graph_GetNumNodes(subgraph, &num_subgraph_nodes));

          // TRT EP should consider the empty subgraph is fully supported by TRT.
          if (num_subgraph_nodes == 0) {
            continue;
          }

          if (!ep->AllNodesAssignedToSpecificEP(subgraph, ep->name_)) {
            // if not all its subgraphs are supported, we need to exclude this control flow op
            return false;
          }
        }
        return true;
      };
      supported_node = supported_control_flow_op(node);
    }

    // Exclude any ops, if applicable
    if (exclude_ops_set.find(op_type) != exclude_ops_set.end()) {
      supported_node = false;
    }

    if (supported_node) {
      if (new_subgraph) {
        parser_nodes_vector.emplace_back();
        // Mark all new graphs as "UnKnown" which will later be parsed by TRT parser
        parser_nodes_vector.back().second = false;
        new_subgraph = false;
      }
      parser_nodes_vector.back().first.emplace_back(index);
    } else {
      new_subgraph = true;
    }
  }

  // Use this local definitions for now
  // TODO: Use provider option
  int max_partition_iterations = 1000;
  int min_subgraph_size = 1;

  bool early_termination = false;
  supported_nodes_vector = ep->GetSupportedList(parser_nodes_vector, 0, max_partition_iterations, graph, &early_termination);
  if (early_termination) {
    supported_nodes_vector.clear();
  }

  // Remove subgraphs if its size is less than the predefined minimal size
  for (auto it = supported_nodes_vector.begin(); it != supported_nodes_vector.end(); ++it) {
    const size_t subgraph_size = it->first.size();
    if (subgraph_size < min_subgraph_size) {
      supported_nodes_vector.erase(it--);
    }
  }

  // TODO: Detect and remove cycles from supported node list

  // TODO: Consolidate supported node list

  // Handle the case where the graph is subgraph of control flow op.
  // The purpose is to make control flow op as well as its subgraphs run on TRT.
  // Here we need to check whether subgraph is fully supported by TRT and don't fuse the nodes of the subgraph until control flow op level.
  if (ep->IsSubGraphOfControlFlowOp(graph) && ep->IsSubGraphFullySupported(graph, supported_nodes_vector)) {
    // const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder(1);
    bool all_subgraphs_are_supported = true;

    // "If" control flow op has two subgraph bodies, "then" body and "else" body respectively.
    // Check its parent node's another subgraph to see whether that subgraph is also fully supported by TRT.
    Ort::ConstNode parent_node = ort_graph.GetParentNode();
    if (parent_node.GetOperatorType() == "If") {
      all_subgraphs_are_supported = false;
      SubGraphCollection_t subgraph_supported_nodes_vector;

      std::vector<Ort::AttrNameSubgraph> attr_name_subgraphs = parent_node.GetSubgraphs();
      for (auto attr_name_subgraph : attr_name_subgraphs) {
        auto subgraph = attr_name_subgraph.sub_graph;
        const OrtGraph* subgraph_raw_pointer = subgraph;
        if (subgraph_raw_pointer != graph) {
          size_t num_subgraph_nodes = 0;
          RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(subgraph, &num_subgraph_nodes));

          // Another subgraph of "If" control flow op has no nodes.
          // In this case, TRT EP should consider this empty subgraph is fully supported by TRT.
          if (num_subgraph_nodes == 0) {
            all_subgraphs_are_supported = true;
            break;
          }
          // Another subgraph of "If" control flow op has been parsed by GetCapability before and all subgraph's nodes assigned to TRT EP.
          else if (ep->AllNodesAssignedToSpecificEP(subgraph, ep->name_)) {
            all_subgraphs_are_supported = true;
            break;
          }
          // Another subgraph of "If" control flow has been parsed by GetCapability and not all subgraph's nodes assigned to TRT EP.
          // (Note: GetExecutionProviderType() returns "" meaning node has not yet been assigned to any EPs)
          else if (!ep->AllNodesAssignedToSpecificEP(subgraph, "")) {
            all_subgraphs_are_supported = false;
            break;
          }

          std::vector<size_t> subgraph_nodes_vector(num_subgraph_nodes);
          std::iota(std::begin(subgraph_nodes_vector), std::end(subgraph_nodes_vector), 0);
          SubGraphCollection_t parser_subgraph_nodes_vector = {{subgraph_nodes_vector, false}};
          bool subgraph_early_termination = false;

          // Another subgraph of "If" control flow has not yet been parsed by GetCapability.
          subgraph_supported_nodes_vector = ep->GetSupportedList(parser_subgraph_nodes_vector, 0, ep->max_partition_iterations_, subgraph, &subgraph_early_termination);
          all_subgraphs_are_supported = ep->IsSubGraphFullySupported(subgraph, subgraph_supported_nodes_vector);
          break;
        }
      }
    }

    if (all_subgraphs_are_supported) {
      // We want the subgraph nodes to be assigned to TRT EP but don't want them to be fused until later at the control flow op level.
      // Simply request the subgraph nodes with a single ComputeCapability for each with no MetaDef (i.e. what the default implementation for IExecutionProvider::GetCapability does).
      for (const auto& group : supported_nodes_vector) {
        if (!group.first.empty()) {
          for (const auto& index : group.first) {
            const OrtNode* supported_node = nodes[index];
            RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddSingleNode(graph_support_info, supported_node));
          }
        }
      }
      std::string message = "[TensorRT EP] Whole graph will run on TensorRT execution provider";
      Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                      OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                      message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

      return nullptr;
    }
  }

  int number_of_trt_nodes = 0;
  for (const auto& group : supported_nodes_vector) {
    if (!group.first.empty()) {
      std::vector<const OrtNode*> supported_nodes;
      supported_nodes.reserve(group.first.size());

      for (const auto& index : group.first) {
        const OrtNode* supported_node = nodes[index];

        supported_nodes.push_back(supported_node);
      }

      // Create (optional) fusion options for the supported nodes to fuse.
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;

      // Set "drop constant initializers" to true as TRT doesn't need ORT to provide constant initializers
      // as inputs to the fused/compiled node at inference time. This allows ORT to release unused initializers.
      node_fusion_options.drop_constant_initializers = true;
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, supported_nodes.data(),
                                                                   supported_nodes.size(), &node_fusion_options));
      number_of_trt_nodes += static_cast<int>(group.first.size());
    }
  }

  const size_t number_of_subgraphs = supported_nodes_vector.size();
  if (number_of_trt_nodes == 0) {
    std::string message = "[TensorRT EP] No graph will run on TensorRT execution provider";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  } else if (number_of_trt_nodes == nodes.size()) {
    std::string message = "[TensorRT EP] Whole graph will run on TensorRT execution provider";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  } else {
    std::string message = "[TensorRT EP] Graph is partitioned and number of subgraphs running on TensorRT execution provider is " + std::to_string(number_of_subgraphs);
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }

  return nullptr;
}

OrtStatus* TensorrtExecutionProvider::CreateNodeComputeInfoFromGraph(OrtEp* this_ptr,
                                                                     const OrtGraph* graph,
                                                                     const OrtNode* fused_node,
                                                                     std::unordered_map<std::string, size_t>& input_map,
                                                                     std::unordered_map<std::string, size_t>& output_map,
                                                                     /* out */ OrtNodeComputeInfo** node_compute_info,
                                                                     /* out */ OrtNode** ep_context_node) {
  TensorrtExecutionProvider* ep = static_cast<TensorrtExecutionProvider*>(this_ptr);

  // Comment out following code if you want the "large" initializers to be saved to a external file.
  /*
  //Save initializers to external file
  std::string ext_ini_file_path = "model_serialized.bin";
  std::filesystem::remove(ext_ini_file_path);
  std::ofstream ext_ini_ofs(ext_ini_file_path, std::ios::binary);
  auto handle_initializer_data = [&ext_ini_ofs, &ext_ini_file_path](
                                     const OrtValueInfo* value_info, const void* data, size_t bytes, bool& is_external,
                                     std::string& location, int64_t& offset) -> Ort::Status {
    // OrtValueInfo* could be used to query initializer's name, type, shape,
    // node consumers, etc.
    (void)value_info;

    if (bytes <= 127) {
      is_external = false;  // Keep small initializers stored inside the TensorProto.
      return Ort::Status{nullptr};
    }

    offset = ext_ini_ofs.tellp();
    location = ext_ini_file_path;
    ext_ini_ofs.write(static_cast<const char*>(data), bytes);
    ext_ini_ofs.flush();
    is_external = true;  // True if is external initializer.

    return Ort::Status{nullptr};
  };
  */

  // Construct ModelProto from OrtGraph
  ONNX_NAMESPACE::ModelProto model_proto;

  // add back handle_initializer_data to save initializer to external file
  OrtEpUtils::OrtGraphToProto(*graph, model_proto /*, handle_initializer_data */);

  std::string string_buf;
  model_proto.SerializeToString(&string_buf);

  if (dump_subgraphs_) {
    // Dump TensorRT subgraphs
    const char* name = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetName(fused_node, &name));
    std::string subgraph_name = name;
    subgraph_name += ".onnx";
    std::fstream dump(subgraph_name, std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto.SerializeToOstream(&dump);
  }

  TensorrtLogger& trt_logger = GetTensorrtLogger(detailed_build_log_, logger_, &ort_api);
  auto trt_builder = GetBuilder(trt_logger);
  auto network_flags = 0;
#if NV_TENSORRT_MAJOR > 8
  network_flags |= (fp16_enable_ || int8_enable_)
                       ? 0
                       : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
  network_flags |= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  auto trt_network = std::unique_ptr<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(network_flags));
  auto trt_config = std::unique_ptr<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
  auto trt_parser =
      tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
  trt_parser->parse(string_buf.data(), string_buf.size(), model_path_);
  if (max_workspace_size_ > 0) {
    trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);
  }

  // Force Pow + Reduce ops in layer norm to run in FP32 to avoid overflow
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  if (fp16_enable_ && layer_norm_fp32_fallback_) {
    for (auto idx = 1; idx < trt_network->getNbLayers() - 1; ++idx) {
      auto layer = trt_network->getLayer(idx);
      auto next_layer = trt_network->getLayer(idx + 1);
      if (layer->getType() == nvinfer1::LayerType::kELEMENTWISE &&
          next_layer->getType() == nvinfer1::LayerType::kREDUCE &&
          (static_cast<nvinfer1::IElementWiseLayer*>(layer))->getOperation() == nvinfer1::ElementWiseOperation::kPOW) {
        std::string message = "[TensorRT EP] Force Pow + Reduce ops in layer norm to run in FP32 to avoid overflow";
        Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                        OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        layer->setPrecision(nvinfer1::DataType::kFLOAT);
        next_layer->setPrecision(nvinfer1::DataType::kFLOAT);
        layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        next_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
      }
    }
  }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

  int num_inputs = trt_network->getNbInputs();
  int num_outputs = trt_network->getNbOutputs();
  std::unordered_map<std::string, size_t> input_indexes(num_inputs);
  std::unordered_map<std::string, size_t> output_indexes(num_outputs);
  std::unordered_map<std::string, size_t> output_types(num_outputs);

  /*
   * Initialize shape range for each dynamic shape input tensor:
   *   1) If user explicitly specifies optimization profiles via provider options, TRT EP will create those profiles
   * during EP compile time. It won't make adjustment for profile values during EP compute time.
   *
   *   2) If no explicit optimization profiles provided by user, TRT EP will firstly set min/max/opt shape to [INT_MAX,
   * INT_MIN, INT_MIN]. Later in EP compute time, the shape will be adjusted to [min_input_value, max_input_value,
   * max_input_value] based on input tensor value.
   *
   *
   * Once the TRT profiles are created:
   *   1) If all the dynamic shape input tensors have associated profiles explicitly provided by user, those profiles
   * will be applied to TRT builder config and the engine will be built at EP compile time.
   *
   *   2) As long as one of the dynamic shape input tensors has no explicitly associated profile, TRT EP will create
   * default shape as described above, and all the profiles won't be applied and engine won't be built until EP compute
   * time.
   */
  bool has_dynamic_shape =
      false;  // True if input tensor has dynamic shape and no explicit profile is specified, otherwise false.
  bool has_explicit_profile = false;
  bool apply_explicit_profile = false;
  int num_profiles = 0;
  std::vector<nvinfer1::IOptimizationProfile*> trt_profiles;

  // Following c++ map data structure is used to help serialize/deserialize profiles where it saves dynamic shape
  // dimension(s) and min/max/opt values for dynamic shape input tensor.
  //
  // (1) Single profile case:
  // For example, assume tensor_a has two dynamic shape dimensions: dim_0 and dim_2, and tensor_b
  // has one dynamic shape dimension: dim_1. The data will be:
  // {
  //   tensor_a: {
  //              dim_0: [[min_shape, max_shape, opt_shape]],
  //              dim_2: [[min_shape, max_shape, opt_shape]]
  //   },
  //   tensor_b: {
  //              dim_1: [[min_shape, max_shape, opt_shape]]
  //   }
  // }
  //
  // (2) Multiple profiles case:
  // For example, assume tensor_a has one dynamic shap dimension: dim 0, and tensor_b has one dynamic shape dimension:
  // dim_1, and both of the tensors have two profiles. The data will be:
  // {
  //   tensor_a: {
  //     dim_0: [[min_shape_0, max_shape_0, opt_shape_0], [min_shape_1, max_shape_1, opt_shape_1]]
  //   },
  //   tensor_b: {
  //     dim_1: [[min_shape_2, max_shape_2, opt_shape_2], [min_shape_3, max_shape_3, opt_shape_3]]
  //   }
  // }
  ShapeRangesMap input_explicit_shape_ranges;
  ShapeRangesMap input_implicit_shape_ranges;

  if ((!profile_min_shapes_.empty()) && (!profile_max_shapes_.empty()) && (!profile_opt_shapes_.empty())) {
    has_explicit_profile = true;
    num_profiles = GetNumProfiles(profile_min_shapes_);
    for (int i = 0; i < num_profiles; i++) {
      trt_profiles.push_back(trt_builder->createOptimizationProfile());
    }
  }

  // Iterate all input tensors to check dynamic shape
  for (unsigned int i = 0, end = num_inputs; i < end; ++i) {
    auto input = trt_network->getInput(i);
    const std::string& input_name = input->getName();
    nvinfer1::Dims dims = input->getDimensions();
    int nb_dims = dims.nbDims;

    // Apply explicit optimization profiles provided by user
    if (has_explicit_profile) {
      apply_explicit_profile =
          ApplyProfileShapesFromProviderOptions(trt_profiles, input, profile_min_shapes_, profile_max_shapes_,
                                                profile_opt_shapes_, input_explicit_shape_ranges, &ep->logger_);
    }

    // If no explicit optimization profile is being applied, TRT EP will later set min/max/opt shape values based on
    // input tensor values at EP compute time
    if (!apply_explicit_profile) {
      if (input->isShapeTensor()) {
        // Shape tensor
        std::vector<std::vector<int64_t>> profile_vector;
        std::vector<int64_t> shape_vector{INT_MAX, INT_MIN, INT_MIN};
        profile_vector.push_back(shape_vector);  // only one profile needed
        input_implicit_shape_ranges[input_name][0] = profile_vector;
        has_dynamic_shape = true;
      } else {
        // Execution tensor
        for (int j = 0, end = nb_dims; j < end; ++j) {
          if (dims.d[j] == -1) {
            std::vector<std::vector<int64_t>> profile_vector;
            std::vector<int64_t> shape_vector{INT_MAX, INT_MIN, INT_MIN};
            profile_vector.push_back(shape_vector);  // only one profile needed
            input_implicit_shape_ranges[input_name][j] = profile_vector;
            has_dynamic_shape = true;
          }
        }
      }
      apply_explicit_profile = false;
    }
  }

  // Set explicit profiles in TRT config if all dynamic shape inputs have associated profiles provided by user
  if (has_explicit_profile) {
    // TRT EP has a constraint here.
    // Users need to provide all the dynamic shape inputs with associated profiles if they want to explicitly specify
    // profiles through provider options.
    if (has_dynamic_shape) {
      std::ostringstream msg;
      msg << "User needs to provide all the dynamic shape inputs with associated profiles if they want to explicitly "
             "set profiles through provider options.\n";
      msg << "Please note that main graph could be partitioned into TRT/CUDA/CPU subgraphs, in this case, user also "
             "needs to provide shape profiles for the TRT subgraph's input if it's dynamic shape input.\n";
      msg << "Following input(s) has no associated shape profiles provided: ";
      auto begin = input_implicit_shape_ranges.begin();
      auto end = input_implicit_shape_ranges.end();
      auto it = begin;
      if (it != end) {
        msg << it->first;
        ++it;
      }
      for (; it != end; ++it) {
        msg << "," << it->first;
      }
      // return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, msg.str());
    } else {
      for (auto trt_profile : trt_profiles) {
        trt_config->addOptimizationProfile(trt_profile);
      }
    }
  }
  // If no explicit profile is applied and the input has dynamic shape, TRT EP simply creates one profile by default.
  // It will later set proper min/max/opt shape values duing EP compute time.
  else if (!has_explicit_profile && has_dynamic_shape) {
    trt_profiles.push_back(trt_builder->createOptimizationProfile());
  }

  // Check platform availability for low precision
  if (fp16_enable_) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    if (!trt_builder->platformHasFastFp16()) {
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      fp16_enable_ = false;
      std::string message = "[TensorRT EP] ORT_TENSORRT_FP16_ENABLE or ORT_TENSORRT_BF16_ENABLE is set, but platform doesn't support fast native fp16/bf16";
      Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                      OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                      message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
  }

  if (int8_enable_) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    if (!trt_builder->platformHasFastInt8()) {
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      int8_enable_ = false;
      std::string message = "[TensorRT EP] ORT_TENSORRT_INT8_ENABLE is set, but platform doesn't support fast native int8";
      Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                      OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                      message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
  }

  // Load INT8 calibration table
  std::unordered_map<std::string, float> dynamic_range_map;
  if (int8_enable_ && int8_calibration_cache_available_) {
    const std::string calibration_cache_path = GetCachePath(cache_path_, int8_calibration_cache_name_);
    if (!ReadDynamicRange(calibration_cache_path, int8_use_native_tensorrt_calibration_table_, dynamic_range_map)) {
      throw std::runtime_error("Failed to read INT8 calibration table " + calibration_cache_path);
    }
  }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  const char* name = nullptr;
  RETURN_IF_ERROR(ort_api.Node_GetName(fused_node, &name));
  std::string fused_node_name = name;

  // Set precision flags
  std::string trt_node_name_with_precision = fused_node_name;
  if (fp16_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    trt_node_name_with_precision += "_fp16";
    std::string message = "[TensorRT EP] FP16 mode is enabled";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
  if (int8_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
    trt_node_name_with_precision += "_int8";
    std::string message = "[TensorRT EP] INT8 mode is enabled";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  // Set DLA
  if (fp16_enable_ || int8_enable_) {
    if (dla_enable_ && dla_core_ >= 0) {  // DLA can only run with FP16 and INT8
      int number_of_dla_core = trt_builder->getNbDLACores();
      if (number_of_dla_core == 0) {
        std::string message = "[TensorRT EP] Try to use DLA core, but platform doesn't have any DLA core";
        Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                        OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        dla_enable_ = false;
      } else {
        if (dla_core_ >= number_of_dla_core) {
          std::string message = "[TensorRT EP] Try to use DLA core #" + std::to_string(dla_core_) +
                                std::string(", but it exceeds platform's maximum DLA core number ") + std::to_string(number_of_dla_core) +
                                std::string(". Use DLA core 0 instead.");
          Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                          OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                          message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
          dla_core_ = 0;
        }
        std::string message = "[TensorRT EP] use DLA core " + dla_core_;
        Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                        OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        trt_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        trt_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        trt_config->setDLACore(dla_core_);
        trt_node_name_with_precision += "_dlacore" + std::to_string(dla_core_);
      }
    }
  }

  // enable sparse weights
  if (sparsity_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    std::string message = "[TensorRT EP] Sparse weights are allowed";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR == 5
  if (build_heuristics_enable_) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kENABLE_TACTIC_HEURISTIC);
    std::string message = "[TensorRT EP] Builder heuristics are enabled." +
                          std::string(" For TRT > 8.5, trt_build_heuristics_enable is deprecated, please set builder ") +
                          std::string("optimization level as 2 to enable builder heuristics.");
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
#elif NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
  // for TRT 8.6 onwards, heuristic-based tactic option is automatically enabled by setting builder optimization level 2
  if (build_heuristics_enable_) {
    if (builder_optimization_level_ == 2) {
      std::string message = "[TensorRT EP] Builder heuristics are automatically enabled by builder optimization " + std::string("level 2. trt_build_heuristics_enable is deprecated on TRT 8.6 onwards.");
      Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                      OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                      message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    } else {
      std::string message = "[TensorRT EP] trt_build_heuristics_enable is deprecated on TRT 8.6 onwards. Please set " + std::string("builder optimization level as 2 to enable builder heuristics.");
      Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                      OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                      message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
  }
#endif

#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
  // switch optimizaion level
  if (builder_optimization_level_ != 3) {
    trt_config->setBuilderOptimizationLevel(builder_optimization_level_);
    std::string message = "[TensorRT EP] Builder optimization level is set to " + builder_optimization_level_;
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }

  // limit auxiliary streams
  if (auxiliary_streams_ >= 0) {
    trt_config->setMaxAuxStreams(auxiliary_streams_);
    std::string message = "[TensorRT EP] Auxiliary streams are se to " + auxiliary_streams_;
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
#else
  if (builder_optimization_level_ != 3) {
    std::string message = "[TensorRT EP] Builder optimization level can only be used on TRT 8.6 onwards!";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
  if (auxiliary_streams_ >= 0) {
    std::string message = "[TensorRT EP] Auxiliary streams can only be set on TRT 8.6 onwards!";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
#endif

  if (weight_stripped_engine_enable_) {
#if NV_TENSORRT_MAJOR >= 10
    trt_config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
    std::string message = "[TensorRT EP] STRIP_PLAN is enabled";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    trt_config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
    message = "[TensorRT EP] REFIT_IDENTICAL is enabled";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
#else
    std::string message = "[TensorRT EP] weight-stripped engines can only be used on TRT 10.0 onwards!";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
#endif
  }

  // limit used tactic sources
  if (!tactic_sources_.empty()) {
    nvinfer1::TacticSources tactics = trt_config->getTacticSources();
    tactics |= GetTacticSourceFromString(tactic_sources_);
    trt_config->setTacticSources(tactics);
    std::string message = "[TensorRT EP] Tactic sources are limited using " + tactic_sources_;
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }

  // Build TRT engine (if needed) and load TRT engine if:
  //   (1) Graph has no dynamic shape input
  //   (2) All the dynamic shape inputs have associated explicit profiles specified by user
  //
  // Otherwise engine will be handled at inference time.
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context;

  std::string cache_path = "";
  std::string cache_suffix = "";
  // Customize cache prefix if assigned
  if (!cache_prefix_.empty()) {
    // Generate cache suffix in case user would like to customize cache prefix
    cache_suffix = "_" + GetCacheSuffix(fused_node_name, trt_node_name_with_precision);
    cache_path = GetCachePath(cache_path_, cache_prefix_) + cache_suffix;
  } else {
    cache_path = GetCachePath(cache_path_, trt_node_name_with_precision);
  }

  std::string cache_hw_compat = "_sm" + compute_capability_;
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
  // Enable hardware compatility mode if assigned
  if (engine_cache_enable_ && engine_hw_compatible_) {
    trt_config->setHardwareCompatibilityLevel(nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS);
    cache_hw_compat = "_sm80+";
    std::string message = "[TensorRT EP] Hardware compatibility is enabled when loading and capturing engine cache.";
    Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
#endif

  // Name the engine cache based on GPU compute capacity and reduce the chance of loading an incompatible cache
  // Note: Engine cache generated on a GPU with large memory might not be loadable on a GPU with smaller memory, even if
  // they share the same compute capacity
  const std::string cache_path_prefix = cache_path + cache_hw_compat;
  std::string engine_cache_path = cache_path_prefix + ".engine";
  const std::string encrypted_engine_cache_path = engine_cache_path + ".encrypted";
  const std::string profile_cache_path = cache_path_prefix + ".profile";

  // If weight-stripped engine is enabled and refitted engine cache is not present,
  // TRT EP will use the engine cache with ".stripped.engine" appended to the end.
  const std::filesystem::path engine_cache_fs_path = engine_cache_path;
  if (weight_stripped_engine_enable_ && !std::filesystem::exists(engine_cache_fs_path)) {
    engine_cache_path = cache_path_prefix + ".stripped.engine";
    weight_stripped_engine_refit_ = true;
  }

  std::unique_ptr<nvinfer1::IHostMemory> serialized_engine;

  if (!has_dynamic_shape) {
    std::string timing_cache_path = "";
    bool engine_update = false;
    if (timing_cache_enable_) {
      timing_cache_path = GetTimingCachePath(global_cache_path_, compute_capability_);
    }
    {
      // ifstream file check, engine serialization/deserialization and engine build are in critical section. It needs
      // lock protection to prevent race condition when inferencing with multithreading.
      auto lock = GetApiLock();

      // If explicit profile flag is on and engine cache enable flag is on,
      // we need to compare explicit profiles and profiles used to build the engine in order to decide whether to
      // rebuild the engine.
      if (has_explicit_profile && engine_cache_enable_) {
        engine_update =
            CompareProfiles(profile_cache_path, profile_min_shapes_, profile_max_shapes_, profile_opt_shapes_);
        if (engine_update) {
          std::string message = "[TensorRT EP] Engine will be built";
          Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                          OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                          message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        } else {
          std::string message = "[TensorRT EP] Engine won't be rebuilt";
          Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                          OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                          message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }
      }

      std::ifstream engine_file(engine_cache_path, std::ios::binary | std::ios::in);
      if (engine_cache_enable_ && !engine_decryption_enable_ && engine_file && !engine_update) {
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        engine_file.read((char*)engine_buf.get(), engine_size);
        trt_engine =
            std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
        std::string message = "[TensorRT EP] DeSerialized " + engine_cache_path;
        Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                        OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        if (trt_engine == nullptr) {
          std::string err_msg = "TensorRT EP could not deserialize engine from cache: " + engine_cache_path;
          return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
        }

      } else if (engine_decryption_enable_ && engine_cache_enable_ &&
                 std::filesystem::exists(encrypted_engine_cache_path) && !engine_update) {
        // Decrypt engine
        size_t engine_size = 0;
        if (!engine_decryption_(encrypted_engine_cache_path.c_str(), nullptr, &engine_size)) {
          std::string err_msg = "TensorRT EP could not get engine buffer size";
          return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
        }
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        if (!engine_decryption_(encrypted_engine_cache_path.c_str(), &engine_buf[0], &engine_size)) {
          std::string err_msg = "TensorRT EP could not call engine decryption function decrypt";
          return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
        }
        // Deserialize engine
        trt_engine =
            std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
        std::string message = "[TensorRT EP] Decrypted and DeSerialized " + encrypted_engine_cache_path;
        Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                        OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        if (trt_engine == nullptr) {
          std::string err_msg = "TensorRT EP could not deserialize engine from encrypted cache: " + encrypted_engine_cache_path;
          return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
        }
      } else {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        // Set INT8 per tensor dynamic range
        if (int8_enable_ && trt_builder->platformHasFastInt8() && int8_calibration_cache_available_) {
          trt_config->setInt8Calibrator(nullptr);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
          if (!SetDynamicRange(*trt_network, dynamic_range_map)) {
            std::string err_msg = "TensorRT EP could not set INT8 dynamic range for fused node: " + fused_node_name;
            return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
          }
        }

        // Load timing cache from file. Create a fresh cache if the file doesn't exist
        std::unique_ptr<nvinfer1::ITimingCache> timing_cache = nullptr;
        if (timing_cache_enable_) {
          std::vector<char> loaded_timing_cache = loadTimingCacheFile(timing_cache_path);
          timing_cache.reset(trt_config->createTimingCache(static_cast<const void*>(loaded_timing_cache.data()),
                                                           loaded_timing_cache.size()));
          if (timing_cache == nullptr) {
            std::string err_msg = "TensorRT EP could not create timing cache: " + timing_cache_path;
            return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
          }
          trt_config->setTimingCache(*timing_cache, force_timing_cache_match_);
          if (detailed_build_log_) {
            std::string message = "[TensorRT EP] Deserialized timing cache from " + timing_cache_path;
            Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                            message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
          }
        }

        // Build engine
        std::chrono::steady_clock::time_point engine_build_start;
        if (detailed_build_log_) {
          engine_build_start = std::chrono::steady_clock::now();
        }

        serialized_engine =
            std::unique_ptr<nvinfer1::IHostMemory>(trt_builder->buildSerializedNetwork(*trt_network, *trt_config));

        if (serialized_engine == nullptr) {
          std::string err_msg = "TensorRT EP failed to create engine from network for fused node: " + fused_node_name;
          return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
        }
        trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
        if (trt_engine == nullptr) {
          std::string err_msg = "TensorRT EP failed to deserialize engine for fused node: " + fused_node_name;
          return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
        }
        if (detailed_build_log_) {
          auto engine_build_stop = std::chrono::steady_clock::now();
          std::string message = "TensorRT engine build for " + trt_node_name_with_precision + std::string(" took: ") +
                                std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(engine_build_stop - engine_build_start).count()) + std::string("ms");
          Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                          OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                          message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }
        if (engine_cache_enable_) {
          // Serialize engine profile if it has explicit profiles
          if (has_explicit_profile) {
            SerializeProfileV2(profile_cache_path, input_explicit_shape_ranges);
            std::string message = "[TensorRT EP] Serialized " + profile_cache_path;
            Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                            message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
          }

          if (engine_decryption_enable_) {
            // Encrypt engine. The library is not always deployed with the encrypt function, so check if it is available
            // first.
            if (engine_encryption_ != nullptr) {
              if (!engine_encryption_(encrypted_engine_cache_path.c_str(),
                                      reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size())) {
                std::string err_msg = "TensorRT EP call to engine encryption library failed";
                return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
              }
              std::string message = "[TensorRT EP] Serialized and encrypted engine " + encrypted_engine_cache_path;
              Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                              message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            } else {
              std::string message = "[TensorRT EP] Engine cache encryption function is not found. No cache is written to disk";
              Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                              message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            }
          } else {
            std::ofstream file(engine_cache_path, std::ios::binary | std::ios::out);
            file.write(reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
            std::string message = "[TensorRT EP] Serialized engine " + engine_cache_path;
            Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                            message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
          }
        }
        // serialize and save timing cache
        if (timing_cache_enable_) {
          auto timing_cache = trt_config->getTimingCache();
          std::unique_ptr<nvinfer1::IHostMemory> timingCacheHostData{timing_cache->serialize()};
          if (timingCacheHostData == nullptr) {
            std::string err_msg = "TensorRT EP could not serialize timing cache: " + timing_cache_path;
            return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
          }
          saveTimingCacheFile(timing_cache_path, timingCacheHostData.get());
          if (detailed_build_log_) {
            std::string message = "[TensorRT EP] Serialized timing cache " + timing_cache_path;
            Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                            message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
          }
        }
      }
    }

    if (weight_stripped_engine_refit_) {
      std::string message = "[TensorRT EP] Refit engine from main ONNX file after engine build";
      Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                      OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                      message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      auto status = RefitEngine(model_path_,
                                onnx_model_folder_path_,
                                engine_cache_path,
                                false /* path check for security */,
                                onnx_model_bytestream_,
                                onnx_model_bytestream_size_,
                                onnx_external_data_bytestream_,
                                onnx_external_data_bytestream_size_,
                                trt_engine.get(),
                                true /* serialize refitted engine to disk */, detailed_build_log_);
      if (status != nullptr) {
        return ort_api.CreateStatus(ORT_EP_FAIL, "RefitEngine failed.");
      }
    }

    // Build context
    // Note: Creating an execution context from an engine is thread safe per TRT doc
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
    if (context_memory_sharing_enable_) {
      // Reset the max_ctx_mem_size_ and context_memory_ since we don't have access to the allocator here.
      max_ctx_mem_size_ = 0;
      context_memory_ = nullptr;
#if NV_TENSORRT_MAJOR < 10
      trt_context =
          std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContextWithoutDeviceMemory());
#else
      trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(
          trt_engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
#endif
    } else {
      trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    }
    if (!trt_context) {
      std::string err_msg = "TensorRT EP could not build execution context for fused node: " + fused_node_name;
      return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    }
  }

  // Create input to index map
  // TRT network input -> ORT fused_node input index
  for (int i = 0; i < num_inputs; ++i) {
    auto input = trt_network->getInput(i);
    const std::string& input_name = input->getName();
    const auto& iter = input_map.find(input_name);
    if (iter != input_map.end()) {
      input_indexes[input_name] = iter->second;
    }
  }

  // Create output to index and type maps
  // TRT network output -> ORT fused_node output index
  const auto& graph_output = model_proto.graph().output();
  for (int i = 0; i < num_outputs; ++i) {
    const std::string& output_name = trt_network->getOutput(i)->getName();
    const auto& iter = output_map.find(output_name);
    if (iter != output_map.end()) {
      output_indexes[output_name] = iter->second;
    }
    const auto& tensor_type = graph_output[i].type().tensor_type();
    output_types[output_name] = tensor_type.elem_type();
  }

  // Save TRT engine, other TRT objects and input/output info to map
  parsers_.emplace(fused_node_name, std::move(trt_parser));
  engines_.emplace(fused_node_name, std::move(trt_engine));
  contexts_.emplace(fused_node_name, std::move(trt_context));
  networks_.emplace(fused_node_name, std::move(trt_network));
  input_info_[fused_node_name].push_back(input_indexes);
  output_info_[fused_node_name].push_back(output_indexes);
  output_info_[fused_node_name].push_back(output_types);
  input_shape_ranges_[fused_node_name] = input_implicit_shape_ranges;
  profiles_.emplace(fused_node_name, std::move(trt_profiles));

  // Create EP Context nodes
  std::unique_ptr<EPContextNodeHelper> ep_ctx_node_helper = std::make_unique<EPContextNodeHelper>(*ep, graph, fused_node);
  if (dump_ep_context_model_) {
    std::string compute_capability_hw_compat = compute_capability_;
    if (engine_cache_enable_ && engine_hw_compatible_) {
      compute_capability_hw_compat = "80+";
    }

    char* serialized_engine_pointer = nullptr;
    size_t serialized_engine_size = 0;

    if (serialized_engine) {
      serialized_engine_pointer = reinterpret_cast<char*>(serialized_engine->data());
      serialized_engine_size = serialized_engine->size();
    } else if (!serialized_engine && ep_context_embed_mode_ && engine_cache_enable_) {
      serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(trt_engine->serialize());
      serialized_engine_pointer = reinterpret_cast<char*>(serialized_engine->data());
      serialized_engine_size = serialized_engine->size();
    }

    ep_ctx_node_helper->CreateEPContextNode(engine_cache_path,
                                            serialized_engine_pointer,
                                            serialized_engine_size,
                                            ep_context_embed_mode_,
                                            compute_capability_hw_compat,
                                            model_path_,
                                            ep_context_node);
  }

  std::unique_ptr<TensorrtComputeState> compute_state = std::make_unique<TensorrtComputeState>();

  // translate tactic sources string to nvinfer1::TacticSources
  nvinfer1::TacticSources tactics = 0;
  if (!tactic_sources_.empty()) {
    tactics = GetTacticSourceFromString(tactic_sources_);
  }
  *compute_state = {
      static_cast<uint32_t>(device_id_),
      fused_node_name,
      builder_.get(),
      &parsers_[fused_node_name],
      &engines_[fused_node_name],
      &contexts_[fused_node_name],
      &networks_[fused_node_name],
      input_info_[fused_node_name],
      output_info_[fused_node_name],
      input_shape_ranges_[fused_node_name],
      &tensorrt_mu_,
      compute_capability_,
      max_workspace_size_,
      fp16_enable_,
      int8_enable_,
      int8_calibration_cache_available_,
      dla_enable_,
      dla_core_,
      trt_node_name_with_precision,
      engine_cache_enable_,
      cache_path_,
      runtime_.get(),
      profiles_[fused_node_name],
      context_memory_sharing_enable_,
      &max_ctx_mem_size_,
      &context_memory_,
      dynamic_range_map,
      engine_decryption_enable_,
      engine_decryption_,
      engine_encryption_,
      timing_cache_enable_,
      global_cache_path_,
      force_timing_cache_match_,
      detailed_build_log_,
      build_heuristics_enable_,
      sparsity_enable_,
      builder_optimization_level_,
      auxiliary_streams_,
      !tactic_sources_.empty(),
      tactics,
      cuda_graph_enable_,
      weight_stripped_engine_enable_,
      weight_stripped_engine_refit_,
      model_path_,
      onnx_model_folder_path_,
      onnx_model_bytestream_,
      onnx_model_bytestream_size_,
      onnx_external_data_bytestream_,
      onnx_external_data_bytestream_size_,
      cache_prefix_,
      cache_suffix,
      engine_hw_compatible_,
      sync_stream_after_enqueue_};

  ep->compute_states_[fused_node_name] = std::move(compute_state);

  // Update the OrtNodeComputeInfo associated with the graph.
  auto ep_node_compute_info = std::make_unique<TRTEpNodeComputeInfo>(*ep);
  *node_compute_info = ep_node_compute_info.release();

  return nullptr;
}

OrtStatus* TensorrtExecutionProvider::CreateNodeComputeInfoFromPrecompiledEngine(OrtEp* this_ptr, const OrtGraph* graph,
                                                                                 const OrtNode* fused_node,
                                                                                 std::unordered_map<std::string, size_t>& input_map,
                                                                                 std::unordered_map<std::string, size_t>& output_map,
                                                                                 OrtNodeComputeInfo** node_compute_info) {
  TensorrtExecutionProvider* ep = static_cast<TensorrtExecutionProvider*>(this_ptr);

  const char* name = nullptr;
  RETURN_IF_ERROR(ort_api.Node_GetName(fused_node, &name));
  std::string fused_node_name = name;

  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context;
  std::unordered_map<std::string, size_t> input_indexes;   // TRT engine input name -> ORT kernel context input index
  std::unordered_map<std::string, size_t> output_indexes;  // TRT engine output name -> ORT kernel context output index
  std::unordered_map<std::string, size_t> output_types;    // TRT engine output name -> ORT output tensor type

  // Get engine binary data and deserialize it
  std::unique_ptr<EPContextNodeReader> ep_context_node_reader = std::make_unique<EPContextNodeReader>(*ep,
                                                                                                      logger_,
                                                                                                      &trt_engine,
                                                                                                      runtime_.get(),
                                                                                                      model_path_,
                                                                                                      compute_capability_,
                                                                                                      weight_stripped_engine_enable_,
                                                                                                      onnx_model_folder_path_,
                                                                                                      onnx_model_bytestream_,
                                                                                                      onnx_model_bytestream_size_,
                                                                                                      onnx_external_data_bytestream_,
                                                                                                      onnx_external_data_bytestream_size_,
                                                                                                      detailed_build_log_);
  RETURN_IF_ERROR(ep_context_node_reader->GetEpContextFromGraph(*graph));

  // Build context
  // Note: Creating an execution context from an engine is thread safe per TRT doc
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
  if (context_memory_sharing_enable_) {
    // Reset the max_ctx_mem_size_ and context_memory_ since we don't have access to the allocator here.
    max_ctx_mem_size_ = 0;
    context_memory_ = nullptr;
#if NV_TENSORRT_MAJOR < 10
    trt_context =
        std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContextWithoutDeviceMemory());
#else
    trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(
        trt_engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
#endif
  } else {
    trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
  }
  if (!trt_context) {
    std::string err_msg = "TensorRT EP could not build execution context for fused node: " + fused_node_name;
    return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
  }

  // Create input/output to index maps
  // TRT engine input -> ORT fused_node input index
  // TRT engine output -> ORT fused_node output index
  for (int32_t i = 0; i < trt_engine->getNbIOTensors(); ++i) {
    auto const& name = trt_engine->getIOTensorName(i);
    auto const& mode = trt_engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      const auto& iter = input_map.find(name);
      if (iter != input_map.end()) {
        input_indexes[name] = iter->second;
      }
    } else {
      const auto& iter = output_map.find(name);
      if (iter != output_map.end()) {
        output_indexes[name] = iter->second;
      }
    }
  }

  // Create output to type map
  size_t num_graph_outputs = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumOutputs(graph, &num_graph_outputs));

  std::vector<const OrtValueInfo*> graph_outputs(num_graph_outputs);
  RETURN_IF_ERROR(ort_api.Graph_GetOutputs(graph, graph_outputs.data(), graph_outputs.size()));

  for (size_t i = 0; i < graph_outputs.size(); i++) {
    const OrtValueInfo* value_info = graph_outputs[i];

    const char* value_info_name = nullptr;
    RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &value_info_name));

    const OrtTypeInfo* type_info = nullptr;
    RETURN_IF_ERROR(ort_api.GetValueInfoTypeInfo(value_info, &type_info));

    const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
    RETURN_IF_ERROR(ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape));

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape, &elem_type));

    output_types[value_info_name] = elem_type;
  }

  // Save TRT engine, TRT context and input/output info to map
  engines_.emplace(fused_node_name, std::move(trt_engine));
  contexts_.emplace(fused_node_name, std::move(trt_context));
  input_info_[fused_node_name].push_back(input_indexes);
  output_info_[fused_node_name].push_back(output_indexes);
  output_info_[fused_node_name].push_back(output_types);

  std::unique_ptr<TensorrtComputeStateForEPContext> compute_state = std::make_unique<TensorrtComputeStateForEPContext>();

  *compute_state = {
      static_cast<uint32_t>(device_id_),
      fused_node_name,
      &engines_[fused_node_name],
      &contexts_[fused_node_name],
      input_info_[fused_node_name],
      output_info_[fused_node_name],
      context_memory_sharing_enable_,
      &max_ctx_mem_size_,
      &context_memory_,
      &tensorrt_mu_,
      sync_stream_after_enqueue_};

  ep->compute_states_for_ep_context_[fused_node_name] = std::move(compute_state);

  // Update the OrtNodeComputeInfo associated with the graph.
  auto ep_node_compute_info = std::make_unique<TRTEpEpContextNodeComputeInfo>(*ep);
  *node_compute_info = ep_node_compute_info.release();

  return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtExecutionProvider::CompileImpl(_In_ OrtEp* this_ptr,
                                                               _In_ const OrtGraph** graphs,
                                                               _In_ const OrtNode** fused_nodes,
                                                               _In_ size_t count,
                                                               _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                                               _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  TensorrtExecutionProvider* ep = static_cast<TensorrtExecutionProvider*>(this_ptr);
  const OrtApi& ort_api = ep->ort_api;

  gsl::span<OrtNodeComputeInfo*> node_compute_infos_result(node_compute_infos, count);
  gsl::span<OrtNode*> ep_context_nodes_result(ep_context_nodes, count);

  for (size_t fused_node_idx = 0; fused_node_idx < count; fused_node_idx++) {
    auto fused_node = fused_nodes[fused_node_idx];

    // Gets number of node's inputs and outputs
    size_t num_node_inputs = 0;
    RETURN_IF_ERROR(ort_api.Node_GetNumInputs(fused_node, &num_node_inputs));

    std::vector<const OrtValueInfo*> node_inputs(num_node_inputs);
    RETURN_IF_ERROR(ort_api.Node_GetInputs(fused_node, node_inputs.data(), node_inputs.size()));

    // Builds map from input name to its index in input list
    std::unordered_map<std::string, size_t> input_map;
    input_map.reserve(num_node_inputs);
    for (size_t i = 0; i < num_node_inputs; i++) {
      const OrtValueInfo* value_info = node_inputs[i];
      const char* name = nullptr;
      RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &name));

      input_map.emplace(name, i);
    }

    // Gets number of node's outputs
    size_t num_node_outputs = 0;
    RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(fused_node, &num_node_outputs));

    std::vector<const OrtValueInfo*> node_outputs(num_node_outputs);
    RETURN_IF_ERROR(ort_api.Node_GetOutputs(fused_node, node_outputs.data(), node_outputs.size()));

    // Builds map from output name to its index in output list
    std::unordered_map<std::string, size_t> output_map;
    output_map.reserve(num_node_outputs);
    for (size_t i = 0; i < num_node_outputs; i++) {
      const OrtValueInfo* value_info = node_outputs[i];
      const char* name = nullptr;
      RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &name));

      output_map.emplace(name, i);
    }

    OrtStatus* status;
    if (EPContextNodeReader::GraphHasCtxNode(graphs[fused_node_idx], ort_api)) {
      RETURN_IF_ERROR(ep->CreateNodeComputeInfoFromPrecompiledEngine(this_ptr, graphs[fused_node_idx], fused_node,
                                                                     input_map, output_map,
                                                                     &node_compute_infos_result[fused_node_idx]));
    } else {
      RETURN_IF_ERROR(ep->CreateNodeComputeInfoFromGraph(this_ptr, graphs[fused_node_idx], fused_node, input_map,
                                                         output_map, &node_compute_infos_result[fused_node_idx],
                                                         &ep_context_nodes_result[fused_node_idx]));
    }
  }

  return nullptr;
}

const char* ORT_API_CALL TensorrtExecutionProvider::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const TensorrtExecutionProvider*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* ORT_API_CALL TensorrtExecutionProvider::CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                                                 _In_ const OrtMemoryDevice* memory_device,
                                                                                 _Outptr_ OrtSyncStreamImpl** stream) noexcept {
  // A per-session OrtSyncStreamImpl can be created here if the session options affect the implementation.
  // Logging of any issues should use logger_ which is the session logger.

  TensorrtExecutionProvider* ep = static_cast<TensorrtExecutionProvider*>(this_ptr);

  // we only create streams for the default device memory.
  if (auto mem_type = ep->factory_.ep_api.MemoryDevice_GetMemoryType(memory_device);
      mem_type != OrtDeviceMemoryType_DEFAULT) {
    std::string error = "Invalid OrtMemoryDevice. Expected OrtDeviceMemoryType_DEFAULT(0). Got ";
    error += std::to_string(mem_type);
    return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
  }

  auto device_id = ep->factory_.ep_api.MemoryDevice_GetDeviceId(memory_device);

  auto sync_stream = std::make_unique<TrtSyncStreamImpl>(ep->factory_, ep, device_id, nullptr);
  *stream = sync_stream.release();

  return nullptr;
}

/**
 * Refit the weight-stripped engine
 */
OrtStatus* TensorrtExecutionProvider::RefitEngine(std::string onnx_model_filename,
                                                  std::string& onnx_model_folder_path,
                                                  std::string& weight_stripped_engine_cath_path,
                                                  bool path_check,
                                                  const void* onnx_model_bytestream,
                                                  size_t onnx_model_bytestream_size,
                                                  const void* onnx_external_data_bytestream,
                                                  size_t onnx_external_data_bytestream_size,
                                                  nvinfer1::ICudaEngine* trt_engine,
                                                  bool serialize_refitted_engine,
                                                  bool detailed_build_log) {
#if NV_TENSORRT_MAJOR >= 10
  bool refit_from_file = onnx_model_bytestream == nullptr && onnx_model_bytestream_size == 0;
  bool refit_with_external_data = onnx_external_data_bytestream != nullptr && onnx_external_data_bytestream_size != 0;
  bool refit_complete = false;
  std::filesystem::path onnx_model_path{onnx_model_folder_path};
  if (refit_from_file) {
    if (!onnx_model_filename.empty()) {
      onnx_model_path.append(onnx_model_filename);
    }
    if (onnx_model_path.empty()) {
      std::string err_msg = "The ONNX model was not provided as path. Please use provide an ONNX bytestream to enable refitting the weightless engine.";
      return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    } else {
      // check if file path to ONNX is legal
      if (path_check && IsAbsolutePath(onnx_model_path.string())) {
        std::string err_msg =
            "For security purpose, the ONNX model path should be set with a relative path, but it is an absolute path: " + onnx_model_path.string();
        "weightless engine.";
        return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      if (path_check && IsRelativePathToParentPath(onnx_model_path.string())) {
        std::string err_msg =
            "The ONNX model path has '..'. For security purpose, it's not allowed to point outside the directory.";
        return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }

      if (!(std::filesystem::exists(onnx_model_path) && std::filesystem::is_regular_file(onnx_model_path))) {
        std::string err_msg = "The ONNX model " + onnx_model_path.string() + " does not exist.";
        return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
    }
  }

  // weight-stripped engine refit logic
  TensorrtLogger& trt_logger = GetTensorrtLogger(detailed_build_log, logger_, &ort_api);
  auto refitter = std::unique_ptr<nvinfer1::IRefitter>(nvinfer1::createInferRefitter(*trt_engine, trt_logger));
  auto parser_refitter =
      std::unique_ptr<nvonnxparser::IParserRefitter>(nvonnxparser::createParserRefitter(*refitter, trt_logger));

#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 12) || NV_TENSORRT_MAJOR > 10
  // New refit APIs
  if (refit_with_external_data) {
    // A valid model bytestream must be passed.
    if (refit_from_file) {
      std::string err_msg = "TensorRT EP's refit with external data must be called with a valid ONNX model bytestream";
      return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    }

    if (!parser_refitter->loadModelProto(onnx_model_bytestream, onnx_model_bytestream_size, nullptr)) {
      std::string err_msg = "TensorRT EP's IParserRefitter could not load model from provided onnx_model_bytestream";
      return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    }

    // Extract weight information from the Refitter.
    int required_weights = refitter->getAllWeights(0, nullptr);
    std::vector<char const*> refit_names(required_weights);
    refitter->getAllWeights(required_weights, refit_names.data());
    std::string message = "[TensorRT EP] Refitter requires " + std::to_string(required_weights) + " weights";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

    // Vectors to keep track of data pointers.
    std::vector<std::string> names;
    names.reserve(required_weights);
    std::vector<const char*> bytes;
    bytes.reserve(required_weights);
    std::vector<int64_t> sizes;
    sizes.reserve(required_weights);

    auto onnx_model = std::make_unique<ONNX_NAMESPACE::ModelProto>();
    ONNX_NAMESPACE::TensorProtos* allInitializers_byte_stream;

    // Reconstruct onnx model view.
    const auto onnx_model_view = std::string((const char*)onnx_model_bytestream,
                                             onnx_model_bytestream_size);
    if (!onnx_model->ParseFromString(onnx_model_view)) {
      std::string err_msg = "The provided ONNX bytestream to refit could not be parsed.";
      return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    }

    // Extract graph and initializer information.
    auto const& graph = onnx_model->mutable_graph();
    allInitializers_byte_stream = graph->mutable_initializer();
    message = "[TensorRT EP] Initializers that were found " + std::to_string(allInitializers_byte_stream->size());
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

    // Loop through all initializers
    int missing_initializer_data = 0;
    for (int initializer_idx = 0; initializer_idx < allInitializers_byte_stream->size(); ++initializer_idx) {
      auto& proto = allInitializers_byte_stream->at(initializer_idx);
      auto& proto_name = proto.name();
      bool weight_is_refittable = std::find(refit_names.begin(), refit_names.end(), proto_name) != refit_names.end();
      if (weight_is_refittable) {
        if (proto.has_data_location()) {
          if (proto.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
            // Default values for reading into external_data blob.
            int64_t offset = 0;
            size_t length = 0;
            auto external_data = proto.mutable_external_data();
            const std::string kOffset = "offset", kLength = "length";
            for (int entry_idx = 0; entry_idx < external_data->size(); ++entry_idx) {
              auto current_key = external_data->at(entry_idx).mutable_key();
              auto current_value = external_data->at(entry_idx).mutable_value();
              if (*current_key == kOffset && !current_value->empty()) {
                offset = std::stoll(*current_value);
              } else if (*current_key == kLength && !current_value->empty()) {
                length = std::stoul(*current_value);
              }
            }
            names.push_back(proto.name());
            bytes.push_back(static_cast<const char*>(onnx_external_data_bytestream) + offset);
            sizes.push_back(length);
          } else {
            std::string err_msg = "[TensorRT EP] Proto: " + proto_name + " expected to have external datalocation, but default datalocation was provided instead.";
            return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
          }
        } else if (proto.has_raw_data()) {
          auto& raw_data = proto.raw_data();
          names.push_back(proto.name());
          bytes.push_back(raw_data.c_str());
          sizes.push_back(raw_data.size());
        } else {
          message = "[TensorRT EP] Proto: " + proto_name + " has no raw nor external data.";
          Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                      OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                      message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
          ++missing_initializer_data;
        }
      } else {
        message = "[TensorRT EP] Initializer with name: " + proto_name + " was not marked as refittable";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      }
    }
    if (missing_initializer_data) {
      std::string err_msg = "[TensorRT EP] RefitEngine is missing " + std::to_string(missing_initializer_data) + " initializers.";
      return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    }

    // Load extracted initializers into the parser
    if (!names.empty()) {
      message = "[TensorRT EP] Number of initializers submitted to refitter " + std::to_string(names.size());
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      for (size_t i = 0; i < names.size(); i++) {
        bool refloadInit = parser_refitter->loadInitializer(names[i].c_str(), bytes[i], sizes[i]);
        if (!refloadInit) {
          std::string err_msg = "TensorRT EP's IParserRefitter could not refit deserialized weight-stripped engine with weights contained in the provided bytestream";
          return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
        }
      }
    }
    // Perform refit.
    if (!parser_refitter->refitModelProto()) {
      std::string err_msg = "TensorRT EP's IParserRefitter refitModelProto() failed with the provided external data bytestream.";
      return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    }
    refit_complete = true;
  }
#else
  // Refitting with external data is not supported prior to TensorRT 10.13. Log a warning in this case for the user.
  if (refit_with_external_data) {
    message = "[TensorRT EP] Refitting with an onnx_external_data_bytestream is only supported on TensorRT versions >= 10.13! This parameter will be ignored for refitting, and the resulting refitted engine may be incorrect.";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
#endif  // (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR > 12) || NV_TENSORRT_MAJOR > 10
  // If new refit flow was not completed, then fallback to refit_from_file.
  if (!refit_complete) {
    if (refit_from_file) {
      std::string message = "[TensorRT EP] Refitting from file on disk: " + onnx_model_path.string();
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      if (!parser_refitter->refitFromFile(onnx_model_path.string().c_str())) {
        std::string err_msg = "TensorRT EP's IParserRefitter could not refit deserialized weight-stripped engine with weights contained in: " + onnx_model_path.string();
        return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
    } else {
      std::string message = "[TensorRT EP] Refitting from byte array";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      if (!parser_refitter->refitFromBytes(onnx_model_bytestream, onnx_model_bytestream_size)) {
        std::string err_msg = "TensorRT EP's IParserRefitter could not refit deserialized weight-stripped engine with weights contained in the provided bytestream";
        return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
    }
  }

  if (refitter->refitCudaEngine()) {
    std::string message = "[TensorRT EP] Successfully refitted the weight-stripped engine.";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  } else {
    std::string err_msg =
        "TensorRT EP's IRefitter could not refit deserialized weight-stripped engine with weights contained in: " +
        onnx_model_path.string();
    return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
  }

  // serialize the refitted engine to disk
  if (serialize_refitted_engine) {
    std::string refitted_engine_cache = GetWeightRefittedEnginePath(weight_stripped_engine_cath_path);
    nvinfer1::IHostMemory* serialized_engine = trt_engine->serialize();
    std::ofstream engine_file(refitted_engine_cache, std::ios::binary | std::ios::out);
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    std::string message = "[TensorRT EP] Serialize the refitted engine to " + refitted_engine_cache;
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
  return nullptr;
#else
  std::string err_msg = "TensorRT EP's IParserRefitter can only be used on TRT 10.0 onwards.";
  return ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
#endif
}

TensorrtExecutionProvider::~TensorrtExecutionProvider() {
  if (alloc_ != nullptr) {
    ort_api.ReleaseAllocator(alloc_);
  }
}

/// <summary>
///
/// Plugin TensorRT EP implementing OrtEp
///
/// </summary>
TensorrtExecutionProvider::TensorrtExecutionProvider(TensorrtExecutionProviderFactory& factory,
                                                     const std::string& name,
                                                     const OrtSessionOptions& session_options,
                                                     const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_(factory),
      name_{name},
      session_options_{session_options},
      logger_{logger} {
  // Implementation of OrtEp interfaces
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  // Initialize the execution provider.

  Ort::Status ort_status(ort_api.Logger_LogMessage(&logger_,
                                                   OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                   ("Plugin EP has been created with name " + name_).c_str(),
                                                   ORT_FILE, __LINE__, __FUNCTION__));

  // populate apis as global for utility functions
  g_ort_api = &ort_api;
  g_ep_api = &ep_api;
  g_model_editor_api = &model_editor_api;

  // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
  // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
  // We extract those EP options to create a new "provider options" key-value map.
  std::string lowercase_ep_name = name_.c_str();
  std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
  // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
  std::string key_prefix = "ep." + lowercase_ep_name + ".";

  // Get all the provider options as session config from sesson
  ProviderOptions provider_options;

  // Get the provider options from all the config entries in session option
  OrtKeyValuePairs* key_value_pairs = nullptr;
  ort_api.GetSessionOptionsConfigEntries(&session_options, &key_value_pairs);

  const char* const* keys = nullptr;
  const char* const* values = nullptr;
  size_t num_entries = 0;
  ort_api.GetKeyValuePairs(key_value_pairs, &keys, &values, &num_entries);

  for (size_t i = 0; i < num_entries; ++i) {
    const char* key = keys[i];

    // only gets ep provider options
    if (strncmp(key, key_prefix.c_str(), key_prefix.size()) == 0) {
      std::string key_str = key;
      const char* value = values[i];
      provider_options[key_str.substr(key_prefix.size())] = value;
    }
  }

  ort_api.ReleaseKeyValuePairs(key_value_pairs);

  // Provider options to TensorrtExecutionProviderInfo
  info_ = TensorrtExecutionProviderInfo::FromProviderOptions(provider_options);
  info_.has_trt_options = true;
  device_id_ = info_.device_id;

  std::string profile_min_shapes, profile_max_shapes, profile_opt_shapes;

  // incase the EP context is dumped the engine cache has to be enabled
  auto enable_engine_cache_for_ep_context_model = [this]() {
    if (dump_ep_context_model_ && ep_context_embed_mode_ == 0) {
      engine_cache_enable_ = true;
    }
  };

  // get provider options
  if (info_.has_trt_options) {
    max_partition_iterations_ = info_.max_partition_iterations;
    min_subgraph_size_ = info_.min_subgraph_size;
    max_workspace_size_ = info_.max_workspace_size;
    fp16_enable_ = info_.fp16_enable;
    int8_enable_ = info_.int8_enable;
    if (int8_enable_) {
      int8_calibration_cache_name_ = info_.int8_calibration_table_name;
      int8_use_native_tensorrt_calibration_table_ = info_.int8_use_native_calibration_table;
    }
    if (fp16_enable_ || int8_enable_) {  // DLA can only be enabled with FP16 or INT8
      dla_enable_ = info_.dla_enable;
      dla_core_ = info_.dla_core;
    }
    dump_subgraphs_ = info_.dump_subgraphs;
    engine_cache_enable_ = info_.engine_cache_enable;
    weight_stripped_engine_enable_ = info_.weight_stripped_engine_enable;
    onnx_model_folder_path_ = info_.onnx_model_folder_path;
    onnx_model_bytestream_ = info_.onnx_bytestream;
    onnx_model_bytestream_size_ = info_.onnx_bytestream_size;
    onnx_external_data_bytestream_ = info_.external_data_bytestream;
    onnx_external_data_bytestream_size_ = info_.external_data_bytestream_size;
    timing_cache_enable_ = info_.timing_cache_enable;
    force_timing_cache_match_ = info_.force_timing_cache;
    detailed_build_log_ = info_.detailed_build_log;
    dump_ep_context_model_ = info_.dump_ep_context_model;
    ep_context_file_path_ = info_.ep_context_file_path;
    ep_context_embed_mode_ = info_.ep_context_embed_mode;
    enable_engine_cache_for_ep_context_model();
    if (engine_cache_enable_ || int8_enable_ || timing_cache_enable_) {
      cache_path_ = info_.engine_cache_path;
      cache_prefix_ = info_.engine_cache_prefix;
    }
    // use a more global cache if given
    if (timing_cache_enable_) {
      if (!info_.timing_cache_path.empty()) {
        global_cache_path_ = info_.timing_cache_path;
      } else {
        global_cache_path_ = cache_path_;
      }
    }
    engine_decryption_enable_ = info_.engine_decryption_enable;
    if (engine_decryption_enable_) {
      engine_decryption_lib_path_ = info_.engine_decryption_lib_path;
    }
    force_sequential_engine_build_ = info_.force_sequential_engine_build;
    context_memory_sharing_enable_ = info_.context_memory_sharing_enable;
    if (fp16_enable_) {
      layer_norm_fp32_fallback_ = info_.layer_norm_fp32_fallback;
    }
    build_heuristics_enable_ = info_.build_heuristics_enable;
    sparsity_enable_ = info_.sparsity_enable;
    builder_optimization_level_ = info_.builder_optimization_level;
    auxiliary_streams_ = info_.auxiliary_streams;
    tactic_sources_ = info_.tactic_sources;
    profile_min_shapes = info_.profile_min_shapes;
    profile_max_shapes = info_.profile_max_shapes;
    profile_opt_shapes = info_.profile_opt_shapes;
    cuda_graph_enable_ = info_.cuda_graph_enable;
    engine_hw_compatible_ = info_.engine_hw_compatible;
    op_types_to_exclude_ = info_.op_types_to_exclude;
  } else {
    // deprecate env provider option
  }

  // Validate setting
  if (max_partition_iterations_ <= 0) {
    std::string message = "[TensorRT EP] TensorRT option trt_max_partition_iterations must be a positive integer value. Set it to 1000";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    max_partition_iterations_ = 1000;
  }
  if (min_subgraph_size_ <= 0) {
    std::string message = "[TensorRT EP] TensorRT option trt_min_subgraph_size must be a positive integer value. Set it to 1";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    min_subgraph_size_ = 1;
  }
  if (max_workspace_size_ <= 0) {
    std::string message = "[TensorRT EP] TensorRT option trt_max_workspace_size must be a positive integer value. Set it to 1073741824 (1GB)";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    max_workspace_size_ = 1 << 30;
  }
  if (dla_core_ < 0) {
    std::string message = "[TensorRT EP] TensorRT option trt_dla_core must be a non-negative integer value. Set it to 0";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    dla_core_ = 0;
  }

  // If ep_context_file_path_ is provided as a directory, create it if it's not existed
  if (dump_ep_context_model_ && !ep_context_file_path_.empty() && std::filesystem::path(ep_context_file_path_).extension().empty() && !std::filesystem::is_directory(ep_context_file_path_)) {
    if (!std::filesystem::create_directory(ep_context_file_path_)) {
      throw std::runtime_error("Failed to create directory " + ep_context_file_path_);
    }
  }

  // If dump_ep_context_model_ is enable, TRT EP forces cache_path_ to be the relative path of ep_context_file_path_.
  // For example,
  //    - original cache path = "engine_cache_dir" -> new cache path = "./context_model_dir/engine_cache_dir"
  //    - original cache path = ""                 -> new cache path = "./context_model_dir"
  // The new cache path will be saved as the "ep_cache_context" node attritue of the EP context node.
  // For security reason, it needs to make sure the engine cache is saved inside context model directory.
  if (dump_ep_context_model_ && engine_cache_enable_) {
    if (IsAbsolutePath(cache_path_)) {
      std::string message = "In the case of dumping context model and for security purpose, the trt_engine_cache_path should be set with a relative path, but it is an absolute path:  " + cache_path_;
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
    if (IsRelativePathToParentPath(cache_path_)) {
      std::string message = "In the case of dumping context model and for security purpose, The trt_engine_cache_path has '..', it's not allowed to point outside the directory.";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    // Engine cache relative path to context model directory.
    // It's used when dumping the "ep_cache_context" node attribute.
    engine_cache_relative_path_to_context_model_dir_ = cache_path_;

    // Make cache_path_ to be the relative path of ep_context_file_path_
    cache_path_ = GetPathOrParentPathOfCtxModel(ep_context_file_path_).append(cache_path_).string();
  }

  // Hardware compatibility: pre-check on environment
  if (engine_cache_enable_ && engine_hw_compatible_) {
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
    if (std::stoi(compute_capability_) < 80) {
      std::string message = "Engine hardware compatibility cannot be enabled as GPU arch < 80. ";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      engine_hw_compatible_ = false;
    } else if (std::stoi(compute_capability_) == 87) {
      std::string message = "Engine hardware compatibility cannot be enabled on Jetson Orin. ";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      engine_hw_compatible_ = false;
    }
#else
    std::string message = "Engine hardware compatibility cannot be enabled as TRT < 8.6. ";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    engine_hw_compatible_ = false;
#endif
  }

  if (engine_cache_enable_ || int8_enable_ || timing_cache_enable_) {
    if (!cache_path_.empty() && !fs::is_directory(cache_path_)) {
      if (!fs::create_directory(cache_path_)) {
        throw std::runtime_error("Failed to create directory " + cache_path_);
      }
    }
    if (!global_cache_path_.empty() && !fs::is_directory(global_cache_path_)) {
      if (!fs::create_directory(global_cache_path_)) {
        throw std::runtime_error("Failed to create directory " + global_cache_path_);
      }
    }
  }

  if (engine_decryption_enable_) {
    LIBTYPE handle = OPENLIB(engine_decryption_lib_path_.c_str());
    if (handle == nullptr) {
      std::string message = "TensorRT EP could not open shared library from " + engine_decryption_lib_path_;
      THROW_IF_ERROR(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
    }
    engine_decryption_ = (int (*)(const char*, char*, size_t*))LIBFUNC(handle, "decrypt");
    engine_encryption_ = (int (*)(const char*, char*, size_t))LIBFUNC(handle, "encrypt");
    if (engine_decryption_ == nullptr) {
      std::string message = "TensorRT EP could not find decryption function in shared library from " + engine_decryption_lib_path_;
      THROW_IF_ERROR(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
    }
  }

  if (int8_enable_) {
    int8_calibration_cache_available_ = !int8_calibration_cache_name_.empty();
  }

  /*
   * Parse explicit min/max/opt profile shapes from provider options.
   *
   * The format of min/max/opt profile shapes is defined as below:
   * "input1:dim1xdim2...,input2:dim1xdim2...,...,input1:dim3xdim4...,input2:dim3xdim4...,..."
   *
   * (Note: if multiple shapes with same input name are specified, TRT EP will consider them as multiple profiles.
   *  Please refer to ParserProfileShapes() for more details)
   *
   */
  bool status = true;
  if (status) {
    status = ParseProfileShapes(profile_min_shapes, profile_min_shapes_);
    if (!status) {
      profile_min_shapes_.clear();
      std::string message = "[TensorRT EP] The format of provider option 'trt_profile_min_shapes' is wrong, please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
  }

  if (status) {
    status = ParseProfileShapes(profile_max_shapes, profile_max_shapes_);
    if (!status) {
      profile_max_shapes_.clear();
      std::string message = "[TensorRT EP] The format of provider option 'trt_profile_max_shapes' is wrong, please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
  }

  if (status) {
    status = ParseProfileShapes(profile_opt_shapes, profile_opt_shapes_);
    if (!status) {
      profile_opt_shapes_.clear();
      std::string message = "[TensorRT EP] The format of provider option 'trt_profile_opt_shapes' is wrong, please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
  }

  if (status) {
    status = ValidateProfileShapes(profile_min_shapes_, profile_max_shapes_, profile_opt_shapes_);
    if (!status) {
      std::string message = "[TensorRT EP] Profile shapes validation failed. Make sure the provider options 'trt_profile_min_shapes', 'trt_profile_max_shapes' and 'trt_profile_opt_shapes' have same input name and number of profile.";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      message = "[TensorRT EP] TRT EP will implicitly create optimization profiles based on input tensor for you.";
      Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                  OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      profile_min_shapes_.clear();
      profile_max_shapes_.clear();
      profile_opt_shapes_.clear();
    }
  }

  // cuda graph:
  // cudaStreamSynchronize() is not allowed in cuda graph capture.
  //
  // external stream:
  // If user provides "external" cuda stream, only this cuda stream will be used even if multiple threads are running InferenceSession.Run() concurrently.
  // So, no need to synchronize different streams after enqueueV3.
  if (cuda_graph_enable_ || external_stream_) {
    sync_stream_after_enqueue_ = false;
  }

  {
    auto lock = GetApiLock();
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(GetTensorrtLogger(detailed_build_log_, logger_, &ort_api)));
  }

  // EP Context setting
  if (dump_ep_context_model_) {
    extra_attr_keys_.push_back(k_ep_ctx_hardware_architecture.c_str());
    extra_attr_keys_.push_back(k_ep_ctx_onnx_model_filename.c_str());

    if (engine_cache_enable_ && engine_hw_compatible_) {
      extra_attr_values_.push_back(k_cc_hw_compatible.c_str());
    } else {
      extra_attr_values_.push_back(compute_capability_.c_str());
    }
    extra_attr_values_.push_back(model_path_);
  }
}

void ORT_API_CALL TensorrtExecutionProvider::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr, OrtNodeComputeInfo** node_compute_infos,
                                                                         size_t num_node_compute_infos) noexcept {
  (void)this_ptr;
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    delete node_compute_infos[i];
  }
}

//
// Implementation of TRTEpNodeComputeInfo
//
TRTEpNodeComputeInfo::TRTEpNodeComputeInfo(TensorrtExecutionProvider& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* TRTEpNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr, OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
  auto* node_compute_info = static_cast<TRTEpNodeComputeInfo*>(this_ptr);
  TensorrtExecutionProvider& ep = node_compute_info->ep;

  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  auto state_it = ep.compute_states_.find(fused_node_name);
  if (state_it == ep.compute_states_.end()) {
    std::string message = "Unable to TensorRT EP's compute state for fused node with name " + fused_node_name;
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  TensorrtComputeState& trt_ep_compute_state = *state_it->second;
  *compute_state = &trt_ep_compute_state;
  return nullptr;
}

OrtStatus* TRTEpNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context) {
  auto* node_compute_info = static_cast<TRTEpNodeComputeInfo*>(this_ptr);
  TensorrtExecutionProvider& ep = node_compute_info->ep;

  TensorrtComputeState* trt_state = reinterpret_cast<TensorrtComputeState*>(compute_state);
  Ort::KernelContext ctx(kernel_context);

  // The whole compute_function should be considered the critical section where multiple threads may update kernel
  // function state, access one builder, create/serialize/save engine, save profile and serialize/save timing cache.
  // Therefore, those operations should be synchronized across different threads when ORT is using multithreading.
  // More details here, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
  std::lock_guard<std::mutex> lock(*(trt_state->tensorrt_mu_ptr));
  const std::unordered_map<std::string, size_t>& input_indexes = (trt_state->input_info)[0];
  const std::unordered_map<std::string, size_t>& output_indexes = (trt_state->output_info)[0];
  const std::unordered_map<std::string, size_t>& output_types = (trt_state->output_info)[1];
  auto fused_node_name = trt_state->fused_node_name;
  // This map "shape_ranges" contains the shape range info for setting TRT optimization profiles.
  // The info is used for both shape tensor and execution tensor:
  // tensor name->(dimension->[min, max, opt])
  auto& shape_ranges = trt_state->input_shape_ranges;
  std::unordered_map<std::string, std::vector<int32_t>>
      shape_tensor_values;  // This map holds "shape tensor -> shape values" for the shape tensor input across this
                            // inference run
  std::unordered_map<std::string, std::vector<int64_t>>
      shape_tensor_values_int64;  // same as above but for int64 shape tensor input

  uint16_t device_id = trt_state->device_id;
  auto max_workspace_size = trt_state->max_workspace_size;
  auto trt_builder = trt_state->builder;
  auto trt_engine = trt_state->engine->get();
  auto trt_context = trt_state->context->get();
  auto trt_profiles = trt_state->profiles;
  auto context_memory = trt_state->context_memory;
  auto max_context_mem_size_ptr = trt_state->max_context_mem_size_ptr;
  auto cache_prefix = trt_state->cache_prefix;
  auto compute_capability = trt_state->compute_capability;
  auto engine_cache_enable = trt_state->engine_cache_enable;
  auto engine_hw_compatible = trt_state->engine_hw_compatible;
  auto timing_cache_enable = trt_state->timing_cache_enable;
  auto force_timing_cache_match = trt_state->force_timing_cache;
  auto global_cache_path = trt_state->timing_cache_path;
  auto detailed_build_log = trt_state->detailed_build_log;

  auto weight_stripped_engine_enable = trt_state->weight_stripped_engine_enable;
  auto weight_stripped_engine_refit = trt_state->weight_stripped_engine_refit;
  auto model_path = trt_state->model_path;
  auto onnx_model_folder_path = trt_state->onnx_model_folder_path;
  auto onnx_model_bytestream = trt_state->onnx_model_bytestream;
  auto onnx_model_bytestream_size = trt_state->onnx_model_bytestream_size;
  auto onnx_external_data_bytestream = trt_state->onnx_external_data_bytestream;
  auto onnx_external_data_bytestream_size = trt_state->onnx_external_data_bytestream_size;

  auto sync_stream_after_enqueue = trt_state->sync_stream_after_enqueue;

  int num_inputs = static_cast<int>(input_indexes.size());
  int num_outputs = static_cast<int>(output_indexes.size());
  bool engine_update = false;
  bool context_update = false;
  std::unordered_set<std::string> input_names;

  std::unordered_map<std::string, DDSOutputAllocatorMap>& dds_output_allocator_maps = ep.GetDDSOutputAllocators();
  auto& dds_output_allocator_map = dds_output_allocator_maps[fused_node_name];

  // Get default OrtMemoryInfo from factory
  const OrtMemoryInfo* mem_info = nullptr;
  if (ep.factory_.cuda_gpu_memory_infos.find(device_id) !=
      ep.factory_.cuda_gpu_memory_infos.end()) {
    mem_info = ep.factory_.cuda_gpu_memory_infos[device_id].get();
  }

  // Get allocator from OrtKernelContext
  if (ep.alloc_ == nullptr) {
    Ort::ThrowOnError(ep.ort_api.KernelContext_GetAllocator(kernel_context, mem_info, &ep.alloc_));
  }
  OrtAllocator* alloc = ep.alloc_;

  void* cuda_stream;
  Ort::ThrowOnError(ep.ort_api.KernelContext_GetGPUComputeStream(kernel_context, &cuda_stream));
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  // Name the engine cache based on GPU compute capacity and reduce the chance of loading an incompatible cache
  // Note: Engine cache generated on a GPU with large memory might not be loadable on a GPU with smaller memory, even
  // if they share the same compute capacity Prepare cache name
  std::string cache_path = "";
  // Customize cache prefix if assigned
  if (!cache_prefix.empty()) {
    cache_path = GetCachePath(trt_state->engine_cache_path, trt_state->cache_prefix) + trt_state->cache_suffix;
  } else {
    cache_path = GetCachePath(trt_state->engine_cache_path, trt_state->trt_node_name_with_precision);
  }

  // Enable hardware compatility mode if assigned
  std::string cache_hw_compat = "_sm" + compute_capability;
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
  if (engine_cache_enable && engine_hw_compatible) {
    cache_hw_compat = "_sm80+";
    std::string message = "[TensorRT EP] Hardware compatibility is enabled when loading and capturing engine cache.";
    Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                   OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                   message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
  }
#endif

  // Name the engine cache based on GPU compute capacity and reduce the chance of loading an incompatible cache
  // Note: Engine cache generated on a GPU with large memory might not be loadable on a GPU with smaller memory, even
  // if they share the same compute capacity
  const std::string cache_path_prefix = cache_path + cache_hw_compat;
  std::string engine_cache_path = cache_path_prefix + ".engine";
  const std::string encrypted_engine_cache_path = engine_cache_path + ".encrypted";
  const std::string profile_cache_path = cache_path_prefix + ".profile";
  std::string timing_cache_path = "";
  if (timing_cache_enable) {
    timing_cache_path = GetTimingCachePath(global_cache_path, compute_capability);
  }

  // If weight-stripped engine is enabled and refitted engine cache is not present,
  // TRT EP will use the engine cache with ".stripped.engine" appended to the end.
  const std::filesystem::path engine_cache_fs_path = engine_cache_path;
  if (weight_stripped_engine_enable && !std::filesystem::exists(engine_cache_fs_path)) {
    engine_cache_path = cache_path_prefix + ".stripped.engine";
    weight_stripped_engine_refit = true;
  }

  // Load serialized engine
  if (trt_state->engine_cache_enable && trt_engine == nullptr) {
    std::ifstream engine_file(engine_cache_path, std::ios::binary | std::ios::in);
    std::ifstream profile_file(profile_cache_path, std::ios::binary | std::ios::in);
    if (engine_file && !trt_state->engine_decryption_enable && profile_file) {
      // Deserialize profile
      shape_ranges = DeserializeProfileV2(profile_file);
      std::string message = "[TensorRT EP] DeSerialized " + profile_cache_path;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

      // Prepare buffer
      engine_file.seekg(0, std::ios::end);
      size_t engine_size = engine_file.tellg();
      engine_file.seekg(0, std::ios::beg);
      std::unique_ptr<char[]> engine_buf{new char[engine_size]};
      engine_file.read((char*)engine_buf.get(), engine_size);

      // Deserialize engine
      // Note: Deserializing an engine from a TensorRT runtime is thread safe per TRT doc
      // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
      trt_state->engine->reset();
      *(trt_state->engine) = std::unique_ptr<nvinfer1::ICudaEngine>(
          trt_state->runtime->deserializeCudaEngine(engine_buf.get(), engine_size));
      if (!(*(trt_state->engine))) {
        std::string err_msg = "TensorRT EP Failed to Build Engine.";
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      message = "[TensorRT EP] DeSerialized " + engine_cache_path;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      trt_engine = trt_state->engine->get();
      context_update = true;

    } else if (trt_state->engine_decryption_enable && std::filesystem::exists(encrypted_engine_cache_path) &&
               profile_file) {
      shape_ranges = DeserializeProfileV2(profile_file);
      std::string message = "[TensorRT EP] DeSerialized " + profile_cache_path;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      //  Decrypt engine
      size_t engine_size = 0;
      if (!trt_state->engine_decryption(encrypted_engine_cache_path.c_str(), nullptr, &engine_size)) {
        std::string err_msg = "TensorRT EP could not get engine buffer size";
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      std::unique_ptr<char[]> engine_buf{new char[engine_size]};
      if (!trt_state->engine_decryption(encrypted_engine_cache_path.c_str(), &engine_buf[0], &engine_size)) {
        std::string err_msg = "TensorRT EP could not call engine decryption function decrypt";
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      // Deserialize engine
      // Note: Deserializing an engine from a TensorRT runtime is thread safe per TRT doc
      // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
      trt_state->engine->reset();
      *(trt_state->engine) = std::unique_ptr<nvinfer1::ICudaEngine>(
          trt_state->runtime->deserializeCudaEngine(engine_buf.get(), engine_size));
      if (!(*(trt_state->engine))) {
        std::string err_msg = "TensorRT EP could not deserialize engine from encrypted cache: " + encrypted_engine_cache_path;
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      message = "[TensorRT EP] Decrypted and DeSerialized " + encrypted_engine_cache_path;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      trt_engine = trt_state->engine->get();
      context_update = true;
    }
  }

  // Check and update shape ranges for dynamic shape inputs.
  for (int i = 0, end = num_inputs; i < end; ++i) {
    auto input = trt_state->network->get()->getInput(i);
    const std::string& input_name = input->getName();
    input_names.insert(input_name);

    // If there is any input tensor in shape_ranges, it means this input tensor has dynamic shape and its profile
    // shape values have not yet resolved. TRT EP will help determine the min/max/opt profile values based on current
    // input tensor value.
    if (shape_ranges.find(input_name) != shape_ranges.end()) {
      auto status = ApplyProfileShapesFromInputTensorValue(trt_profiles, ctx, input, shape_ranges, input_indexes,
                                                           shape_tensor_values, shape_tensor_values_int64, stream,
                                                           &engine_update);
      if (status != nullptr) {
        std::string err_msg = "TensorRT EP failed to parse input tensor and generate optimization profiles.";
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
    }
  }

  // Regenerate engine
  if (engine_update) {
    // Destroy the IExecutionContext objects before destroying an engine object, otherwise it will lead to undefined
    // behavior.
    trt_state->context->reset();
    trt_state->engine->reset();
    auto trt_config = std::unique_ptr<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
    if (max_workspace_size > 0) {
      trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size);
    }
    for (auto trt_profile : trt_profiles) {
      trt_config->addOptimizationProfile(trt_profile);
    }
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    // Set INT8 Per Tensor Dynamic range
    if (trt_state->int8_enable && trt_builder->platformHasFastInt8() && trt_state->int8_calibration_cache_available) {
      trt_config->setInt8Calibrator(nullptr);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      if (!SetDynamicRange(*trt_state->network->get(), trt_state->dynamic_range_map)) {
        std::string err_msg = "TensorRT EP failed to set INT8 dynamic range.";
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
    }
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    // Set precision
    if (trt_state->int8_enable) {
      trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
      std::string message = "[TensorRT EP] INT8 mode is enabled";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
    if (trt_state->fp16_enable) {
      trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
      std::string message = "[TensorRT EP] FP16 mode is enabled";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    // Set DLA (DLA can only run with FP16 or INT8)
    if ((trt_state->fp16_enable || trt_state->int8_enable) && trt_state->dla_enable) {
      std::string message = "[TensorRT EP] use DLA core " + trt_state->dla_core;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      trt_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      trt_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
      trt_config->setDLACore(trt_state->dla_core);
    }

    // enable sparse weights
    if (trt_state->sparsity_enable) {
      trt_config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
      std::string message = "[TensorRT EP] Sparse weights are allowed";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR == 5
    // enable builder heuristics
    if (trt_state->build_heuristics_enable) {
      trt_config->setFlag(nvinfer1::BuilderFlag::kENABLE_TACTIC_HEURISTIC);
      std::string message = "[TensorRT EP] Builder heuristics are enabled";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
#elif NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
    // switch optimizaion level
    if (trt_state->builder_optimization_level != 3) {
      trt_config->setBuilderOptimizationLevel(trt_state->builder_optimization_level);
      std::string message = "[TensorRT EP] Builder optimization level is set to " + trt_state->builder_optimization_level;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    // limit auxiliary streams
    if (trt_state->auxiliary_streams >= 0) {
      trt_config->setMaxAuxStreams(trt_state->auxiliary_streams);
      std::string message = "[TensorRT EP] Auxiliary streams are se to " + trt_state->auxiliary_streams;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
#else
    if (trt_state->builder_optimization_level != 3) {
      std::string message = "[TensorRT EP] Builder optimization level can only be used on TRT 8.6 onwards!";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
    if (trt_state->auxiliary_streams >= 0) {
      std::string message = "[TensorRT EP] Auxiliary streams can only be set on TRT 8.6 onwards!";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
#endif
    if (weight_stripped_engine_enable) {
#if NV_TENSORRT_MAJOR >= 10
      trt_config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
      std::string message = "[TensorRT EP] STRIP_PLAN is enabled";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      trt_config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
      message = "[TensorRT EP] REFIT_IDENTICAL is enabled";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
#else
      std::string message = "[TensorRT EP] weight-stripped engines can only be used on TRT 10.0 onwards!";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
#endif
    }
    // limit used tactic sources
    if (trt_state->filter_tactic_sources) {
      nvinfer1::TacticSources tactics = trt_config->getTacticSources();
      tactics |= trt_state->tactic_sources;
      trt_config->setTacticSources(tactics);
      std::string message = "[TensorRT EP] Tactic sources are limited using bitmask " + tactics;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    // Load timing cache from file. Create a fresh cache if the file doesn't exist
    std::unique_ptr<nvinfer1::ITimingCache> timing_cache = nullptr;
    if (trt_state->timing_cache_enable) {
      std::vector<char> loaded_timing_cache = loadTimingCacheFile(timing_cache_path);
      timing_cache.reset(trt_config->createTimingCache(static_cast<const void*>(loaded_timing_cache.data()),
                                                       loaded_timing_cache.size()));
      if (timing_cache == nullptr) {
        std::string err_msg = "TensorRT EP could not create timing cache: " + timing_cache_path;
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      trt_config->setTimingCache(*timing_cache, force_timing_cache_match);
      if (detailed_build_log) {
        std::string message = "[TensorRT EP] Deserialized timing cache from " + timing_cache_path;
        Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                       message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      }
    }

#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 5 || NV_TENSORRT_MAJOR > 8
    // Enable hardware compatility mode if assigned
    if (trt_state->engine_hw_compatible) {
      trt_config->setHardwareCompatibilityLevel(nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS);
      std::string message = "[TensorRT EP] Re-generate engine with hardware compatibility enabled.";
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
#endif

    // Build engine
    std::unique_ptr<nvinfer1::IHostMemory> serialized_engine;
    {
      auto lock = ep.GetApiLock();
      std::chrono::steady_clock::time_point engine_build_start;
      if (detailed_build_log) {
        engine_build_start = std::chrono::steady_clock::now();
      }
      serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
          trt_builder->buildSerializedNetwork(*trt_state->network->get(), *trt_config));
      if (!serialized_engine) {
        std::string err_msg = "TensorRT EP failed to create engine from network.";
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      *(trt_state->engine) = std::unique_ptr<nvinfer1::ICudaEngine>(
          trt_state->runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
      if (!(*(trt_state->engine))) {
        std::string err_msg = "TensorRT EP failed to deserialize engine.";
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      if (detailed_build_log) {
        auto engine_build_stop = std::chrono::steady_clock::now();
        std::string message = "TensorRT engine build for " + trt_state->trt_node_name_with_precision + " took: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(engine_build_stop - engine_build_start).count()) + "ms";
        Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      }
    }
    if (!(*(trt_state->engine))) {
      std::string err_msg = "TensorRT EP Failed to Build Engine.";
      return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    }
    trt_engine = trt_state->engine->get();
    if (trt_state->engine_cache_enable) {
      // Serialize engine profile
      SerializeProfileV2(profile_cache_path, shape_ranges);
      std::string message = "[TensorRT EP] Serialized " + profile_cache_path;
      Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

      // Serialize engine
      if (trt_state->engine_decryption_enable) {
        // Encrypt engine. The library is not always deployed with the encrypt function, so check if it is available
        // first.
        if (trt_state->engine_encryption != nullptr) {
          if (!trt_state->engine_encryption(encrypted_engine_cache_path.c_str(),
                                            reinterpret_cast<char*>(serialized_engine->data()),
                                            serialized_engine->size())) {
            std::string err_msg = "TensorRT EP could not call engine encryption function encrypt";
            return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
          }
          std::string message = "[TensorRT EP] Serialized and encrypted engine " + encrypted_engine_cache_path;
          Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                         OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                         message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        } else {
          std::string message = "[TensorRT EP] Engine cache encryption function is not found. No cache is written to disk";
          Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                         OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                         message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }
      } else {
        std::ofstream file(engine_cache_path, std::ios::binary | std::ios::out);
        file.write(reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
        std::string message = "[TensorRT EP] Serialized " + engine_cache_path;
        Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                       message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      }
    }

    // serialize and save timing cache
    if (trt_state->timing_cache_enable) {
      auto timing_cache = trt_config->getTimingCache();
      std::unique_ptr<nvinfer1::IHostMemory> timingCacheHostData{timing_cache->serialize()};
      if (timingCacheHostData == nullptr) {
        std::string err_msg = "TensorRT EP could not serialize timing cache: " + timing_cache_path;
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
      }
      saveTimingCacheFile(timing_cache_path, timingCacheHostData.get());
      if (detailed_build_log) {
        std::string message = "[TensorRT EP] Serialized timing cache " + timing_cache_path;
        Ort::ThrowOnError(ep.ort_api.Logger_LogMessage(&ep.logger_,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                       message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
      }
    }

    // TODO: In current ORT's EPContext design, there is no way TRT EP can update the engine cache binary in EPContext node with the rebuilt engine.
    //       The hacky way is to directly modify the EPContext model that graph_partitioner generates in session initialization.

    context_update = true;

    if (weight_stripped_engine_refit) {
      auto status =
          ep.RefitEngine(model_path,
                         onnx_model_folder_path,
                         engine_cache_path,
                         false /* path check for security */,
                         onnx_model_bytestream,
                         onnx_model_bytestream_size,
                         onnx_external_data_bytestream,
                         onnx_external_data_bytestream_size,
                         trt_engine,
                         true /* serialize refitted engine to disk */, detailed_build_log);
      if (status != nullptr) {
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, "RefitEngine failed.");
      }
    }
  }

  if (context_update) {
    if (trt_state->context_memory_sharing_enable) {
#if NV_TENSORRT_MAJOR < 10
      *(trt_state->context) = std::unique_ptr<nvinfer1::IExecutionContext>(
          trt_state->engine->get()->createExecutionContextWithoutDeviceMemory());
#else
      *(trt_state->context) =
          std::unique_ptr<nvinfer1::IExecutionContext>(trt_state->engine->get()->createExecutionContext(
              nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
#endif
    } else {
      *(trt_state->context) =
          std::unique_ptr<nvinfer1::IExecutionContext>(trt_state->engine->get()->createExecutionContext());
    }
    if (!(*(trt_state->context))) {
      std::string err_msg = "TensorRT EP failed to create context.";
      return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
    }
    trt_context = trt_state->context->get();
  }

  // Check before using trt_engine
  if (trt_engine == nullptr) {
    std::string err_msg = "No engine is found.";
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
  }

  // Get input and output binding names
  int total_bindings = trt_engine->getNbIOTensors();
  std::vector<char const*> input_binding_names, output_binding_names;
  for (int i = 0, end = total_bindings; i < end; ++i) {
    auto const& name = trt_engine->getIOTensorName(i);
    auto const& mode = trt_engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_binding_names.push_back(name);
    } else {
      output_binding_names.push_back(name);
    }
  }

  /*
   * Set input shapes and bind input buffers
   */
  std::vector<AllocatorUniquePtr<void>> scratch_buffers;
  for (size_t i = 0, end = input_binding_names.size(); i < end; ++i) {
    char const* input_name = input_binding_names[i];

    size_t input_index = 0;
    const auto iter = input_indexes.find(input_name);
    if (iter != input_indexes.end()) {
      input_index = iter->second;
    }
    auto input_tensor = ctx.GetInput(input_index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shapes = tensor_info.GetShape();

    auto status = BindContextInput(ctx, trt_engine, trt_context, input_name, input_index, shape_tensor_values,
                                   shape_tensor_values_int64, scratch_buffers, alloc, stream);
    if (status != nullptr) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL, "BindContextInput failed.");
    }
  }

  /*
   * Set output shapes and bind output buffers
   */
  std::unordered_map<char const*, void*> buffers;
  buffers.reserve(num_outputs);
  using OutputOrtValue = Ort::UnownedValue;
  std::unordered_map<size_t, OutputOrtValue> output_tensors;
  output_tensors.reserve(num_outputs);
  std::unordered_map<size_t, int> output_dim_sizes;
  output_dim_sizes.reserve(num_outputs);

  for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
    char const* output_name = output_binding_names[i];

    size_t output_index = 0;
    const auto& index_iter = output_indexes.find(output_name);
    if (index_iter != output_indexes.end()) {
      output_index = index_iter->second;
    }

    size_t output_type = 0;
    const auto type_iter = output_types.find(output_name);
    if (type_iter != output_types.end()) {
      output_type = type_iter->second;
    }

    auto status = BindContextOutput(ctx, trt_context, output_name, output_index, output_type, i, output_tensors,
                                    output_dim_sizes, dds_output_allocator_map, scratch_buffers, alloc, buffers);
    if (status != nullptr) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL, "BindContextOutput failed.");
    }
  }

  // Set execution context memory
  if (trt_state->context_memory_sharing_enable) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    size_t mem_size = trt_engine->getDeviceMemorySize();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (mem_size > *max_context_mem_size_ptr) {
      *max_context_mem_size_ptr = mem_size;
      *context_memory = MakeUniquePtrFromOrtAllocator<void>(alloc, *max_context_mem_size_ptr, true);
    }
    trt_context->setDeviceMemory((*context_memory).get());
  }

  // TODO: Add support for CUDA graph for plugin ep.
  /*
  // Start CUDA graph capture.
  // Note: The reason we don't put graph capture in OnRunStart() like CUDA EP does is because
  // current ORT TRT doesn't get cuda stream until compute time and graph capture requires cuda stream.
  if (cuda_graph_enable_ && IsGraphCaptureAllowed() && !IsGraphCaptured(0)) {
    // LOGS_DEFAULT(INFO) << "Capturing the cuda graph for this model";
    cuda_graph_.SetStream(stream);
    CaptureBegin(0);
  }
  */

  // Run TRT inference
  if (!trt_context->enqueueV3(stream)) {
    std::string err_msg = "TensorRT EP execution context enqueue failed.";
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
  }

  /*
   * Given that InferenceSession::Run() is guaranteed to be thread-safe meaning multiple threads can call this
   * function concurrently, TRT EP needs to carefully take care of concurrency here, if not, following concurrent
   * issue might happen:
   *
   * It's suggested that to perform inference concurrently in multiple streams, use one trt execution context per
   * stream. In the design of TRT EP (Not apply per-thread context implementation) and if multiple threads are calling
   * InferenceSession::Run() concurrently, the trt execution context instance is shared by all the threads and each
   * thread aquires different stream from ORT. So TRT EP will end up having one trt execution context using multiple
   * streams which is not suggested. But, since the whole compute_func() is protected by the lock and if
   * cudaStreamSynchronize() is enforced here, one trt execution context per stream is guaranteed.
   *
   * Therefore, TRT EP needs to call cudaStreamSynchronize() which means to wait until stream has completed all
   * operations to prevent the concurrent issue mentioned above. However, if cuda graph is enabled, TRT EP won't call
   * cudaStreamSynchronize() since it's not allowed during graph capture.
   */
  if (sync_stream_after_enqueue) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  }

  // Assign TRT output back to ORT output
  // (1) Bind TRT DDS output to ORT kernel context output. (It needs to wait until enqueueV3 is finished)
  // (2) Cast TRT INT32 output to ORT INT64 output or TRT double output to float output
  for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
    char const* output_name = output_binding_names[i];

    size_t output_type = 0;
    const auto& iter = output_types.find(output_name);
    if (iter != output_types.end()) {
      output_type = iter->second;
    }

    if (dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end()) {
      size_t output_index = 0;
      const auto& index_iter = output_indexes.find(output_name);
      if (index_iter != output_indexes.end()) {
        output_index = index_iter->second;
      }
      auto status = BindKernelOutput(ctx, mem_info, dds_output_allocator_map, output_name, output_index, output_type, stream);
      if (status != nullptr) {
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, "BindKernelOutput failed.");
      }
    } else {
      auto& output_tensor = output_tensors[i];
#if NV_TENSORRT_MAJOR < 10
      if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        auto output_tensor_ptr = output_tensor.GetTensorMutableData<int64_t>();
        if (output_tensor_ptr != nullptr) {
          cuda::Impl_Cast<int32_t, int64_t>(stream, reinterpret_cast<int32_t*>(buffers[output_name]), output_tensor_ptr,
                                            output_dim_sizes[i]);
        }
      }
#endif
      if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        auto output_tensor_ptr = output_tensor.GetTensorMutableData<double>();
        if (output_tensor_ptr != nullptr) {
          cuda::Impl_Cast<float, double>(stream, reinterpret_cast<float*>(buffers[output_name]), output_tensor_ptr,
                                         output_dim_sizes[i]);
        }
      }
    }
  }

  // TODO: Add support for CUDA graph for plugin ep.
  /*
  // End CUDA graph capture.
  // Note: One reason we don't put end of graph capture in OnRunEnd() like CUDA EP does is because of cuda stream
  // mentioned in graph capture above, another reason is because OnRunEnd() is not synchronized with OnRunStart() and
  // ExecuteGraph() per inference_session.cc. It's safe to start/end CUDA graph capture in compute_func() here since
  // cuda graph object is maintained by a per thread basis.
  if (cuda_graph_enable_ && !IsGraphCaptured(0)) {
    if (IsGraphCaptureAllowed()) {
      CaptureEnd(0);
      // CUDA work issued to a capturing stream doesn't actually run on the GPU,
      // so run the captured graph here to actually execute the work.
      ORT_RETURN_IF_ERROR(ReplayGraph(0));
    } else {
      IncrementRegularRunCountBeforeGraphCapture();
    }
  }
  */

  return nullptr;
}

void TRTEpNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  TensorrtComputeState& trt_ep_compute_state = *reinterpret_cast<TensorrtComputeState*>(compute_state);
  (void)trt_ep_compute_state;
  // Do nothing for here.
}

//
// Implementation of TRTEpEpContextNodeComputeInfo
//
TRTEpEpContextNodeComputeInfo::TRTEpEpContextNodeComputeInfo(TensorrtExecutionProvider& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* TRTEpEpContextNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr, OrtNodeComputeContext* compute_context,
                                                          void** compute_state) {
  auto* node_compute_info = static_cast<TRTEpEpContextNodeComputeInfo*>(this_ptr);
  TensorrtExecutionProvider& ep = node_compute_info->ep;

  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  auto state_it = ep.compute_states_for_ep_context_.find(fused_node_name);
  if (state_it == ep.compute_states_for_ep_context_.end()) {
    std::string message = "Unable to TensorRT EP's compute state for fused node with name " + fused_node_name;
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  TensorrtComputeStateForEPContext& trt_ep_compute_state = *state_it->second;
  *compute_state = &trt_ep_compute_state;
  return nullptr;
}

OrtStatus* TRTEpEpContextNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                                      OrtKernelContext* kernel_context) {
  auto* node_compute_info = static_cast<TRTEpEpContextNodeComputeInfo*>(this_ptr);
  TensorrtExecutionProvider& ep = node_compute_info->ep;

  TensorrtComputeStateForEPContext* trt_state = reinterpret_cast<TensorrtComputeStateForEPContext*>(compute_state);
  Ort::KernelContext ctx(kernel_context);

  // The whole compute_function should be considered the critical section.
  // More details here, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
  std::lock_guard<std::mutex> lock(*(trt_state->tensorrt_mu_ptr));

  const std::unordered_map<std::string, size_t>& input_indexes = (trt_state->input_info)[0];
  const std::unordered_map<std::string, size_t>& output_indexes = (trt_state->output_info)[0];
  const std::unordered_map<std::string, size_t>& output_types = (trt_state->output_info)[1];
  uint16_t device_id = trt_state->device_id;
  auto fused_node_name = trt_state->fused_node_name;
  std::unordered_map<std::string, DDSOutputAllocatorMap>& dds_output_allocator_maps = ep.GetDDSOutputAllocators();
  auto& dds_output_allocator_map = dds_output_allocator_maps[fused_node_name];
  auto trt_engine = trt_state->engine->get();
  auto trt_context = trt_state->context->get();
  auto max_context_mem_size_ptr = trt_state->max_context_mem_size_ptr;
  auto context_memory = trt_state->context_memory;
  auto sync_stream_after_enqueue = trt_state->sync_stream_after_enqueue;
  int num_outputs = static_cast<int>(output_indexes.size());
  std::unordered_map<std::string, std::vector<int32_t>> shape_tensor_values;        // This map holds "shape tensor -> shape values" for the shape tensor input across this inference run
  std::unordered_map<std::string, std::vector<int64_t>> shape_tensor_values_int64;  // same as above but for int64 shape tensor input

  // Get default OrtMemoryInfo from factory
  const OrtMemoryInfo* mem_info = nullptr;
  if (ep.factory_.cuda_gpu_memory_infos.find(device_id) !=
      ep.factory_.cuda_gpu_memory_infos.end()) {
    mem_info = ep.factory_.cuda_gpu_memory_infos[device_id].get();
  }

  // Get allocator from OrtKernelContext
  if (ep.alloc_ == nullptr) {
    Ort::ThrowOnError(ep.ort_api.KernelContext_GetAllocator(kernel_context, mem_info, &ep.alloc_));
  }
  OrtAllocator* alloc = ep.alloc_;

  void* cuda_stream;
  Ort::ThrowOnError(ep.ort_api.KernelContext_GetGPUComputeStream(kernel_context, &cuda_stream));
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  // cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Check before using trt_engine
  if (trt_engine == nullptr) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, "No engine is found.");
  }

  // Get input and output binding names
  int total_bindings = trt_engine->getNbIOTensors();
  std::vector<char const*> input_binding_names, output_binding_names;
  for (int i = 0, end = total_bindings; i < end; ++i) {
    auto const& name = trt_engine->getIOTensorName(i);
    auto const& mode = trt_engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_binding_names.push_back(name);
    } else {
      output_binding_names.push_back(name);
    }
  }

  /*
   * Set input shapes and bind input buffers
   */
  std::vector<AllocatorUniquePtr<void>> scratch_buffers;
  for (size_t i = 0, end = input_binding_names.size(); i < end; ++i) {
    char const* input_name = input_binding_names[i];

    size_t input_index = 0;
    const auto iter = input_indexes.find(input_name);
    if (iter != input_indexes.end()) {
      input_index = iter->second;
    }
    auto input_tensor = ctx.GetInput(input_index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shapes = tensor_info.GetShape();

    auto status = BindContextInput(ctx, trt_engine, trt_context, input_name, input_index, shape_tensor_values,
                                   shape_tensor_values_int64, scratch_buffers, alloc, stream);
    if (status != nullptr) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL, "BindContextInput failed.");
    }
  }

  /*
   * Set output shapes and bind output buffers
   */
  std::unordered_map<char const*, void*> buffers;
  buffers.reserve(num_outputs);
  using OutputOrtValue = Ort::UnownedValue;
  std::unordered_map<size_t, OutputOrtValue> output_tensors;
  output_tensors.reserve(num_outputs);
  std::unordered_map<size_t, int> output_dim_sizes;
  output_dim_sizes.reserve(num_outputs);

  for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
    char const* output_name = output_binding_names[i];

    size_t output_index = 0;
    const auto& index_iter = output_indexes.find(output_name);
    if (index_iter != output_indexes.end()) {
      output_index = index_iter->second;
    }

    size_t output_type = 0;
    const auto type_iter = output_types.find(output_name);
    if (type_iter != output_types.end()) {
      output_type = type_iter->second;
    }

    auto status = BindContextOutput(ctx, trt_context, output_name, output_index, output_type, i, output_tensors,
                                    output_dim_sizes, dds_output_allocator_map, scratch_buffers, alloc, buffers);
    if (status != nullptr) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL, "BindContextOutput failed.");
    }
  }

  // Set execution context memory
  if (trt_state->context_memory_sharing_enable) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    size_t mem_size = trt_engine->getDeviceMemorySize();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (mem_size > *max_context_mem_size_ptr) {
      *max_context_mem_size_ptr = mem_size;
      *context_memory = MakeUniquePtrFromOrtAllocator<void>(alloc, *max_context_mem_size_ptr, true);
    }
    trt_context->setDeviceMemory((*context_memory).get());
  }

  // TODO: Add support for CUDA graph for plugin ep.
  /*
  // Start CUDA graph capture.
  // Note: The reason we don't put graph capture in OnRunStart() like CUDA EP does is because
  // current ORT TRT doesn't get cuda stream until compute time and graph capture requires cuda stream.
  if (cuda_graph_enable_ && IsGraphCaptureAllowed() && !IsGraphCaptured(0)) {
    // LOGS_DEFAULT(INFO) << "Capturing the cuda graph for this model";
    cuda_graph_.SetStream(stream);
    CaptureBegin(0);
  }
  */

  // Run TRT inference
  if (!trt_context->enqueueV3(stream)) {
    std::string err_msg = "TensorRT EP execution context enqueue failed.";
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, err_msg.c_str());
  }

  /*
   * Given that InferenceSession::Run() is guaranteed to be thread-safe meaning multiple threads can call this
   * function concurrently, TRT EP needs to carefully take care of concurrency here, if not, following concurrent
   * issue might happen:
   *
   * It's suggested that to perform inference concurrently in multiple streams, use one trt execution context per
   * stream. In the design of TRT EP (Not apply per-thread context implementation) and if multiple threads are calling
   * InferenceSession::Run() concurrently, the trt execution context instance is shared by all the threads and each
   * thread aquires different stream from ORT. So TRT EP will end up having one trt execution context using multiple
   * streams which is not suggested. But, since the whole compute_func() is protected by the lock and if
   * cudaStreamSynchronize() is enforced here, one trt execution context per stream is guaranteed.
   *
   * Therefore, TRT EP needs to call cudaStreamSynchronize() which means to wait until stream has completed all
   * operations to prevent the concurrent issue mentioned above. However, if cuda graph is enabled, TRT EP won't call
   * cudaStreamSynchronize() since it's not allowed during graph capture.
   */
  if (sync_stream_after_enqueue) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  }

  // Assign TRT output back to ORT output
  // (1) Bind TRT DDS output to ORT kernel context output. (It needs to wait until enqueueV3 is finished)
  // (2) Cast TRT INT32 output to ORT INT64 output or TRT double output to float output
  for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
    char const* output_name = output_binding_names[i];

    size_t output_type = 0;
    const auto& iter = output_types.find(output_name);
    if (iter != output_types.end()) {
      output_type = iter->second;
    }

    if (dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end()) {
      size_t output_index = 0;
      const auto& index_iter = output_indexes.find(output_name);
      if (index_iter != output_indexes.end()) {
        output_index = index_iter->second;
      }
      auto status = BindKernelOutput(ctx, mem_info, dds_output_allocator_map, output_name, output_index, output_type, stream);
      if (status != nullptr) {
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, "BindKernelOutput failed.");
      }
    } else {
      auto& output_tensor = output_tensors[i];
#if NV_TENSORRT_MAJOR < 10
      if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        auto output_tensor_ptr = output_tensor.GetTensorMutableData<int64_t>();
        if (output_tensor_ptr != nullptr) {
          cuda::Impl_Cast<int32_t, int64_t>(stream, reinterpret_cast<int32_t*>(buffers[output_name]), output_tensor_ptr,
                                            output_dim_sizes[i]);
        }
      }
#endif
      if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        auto output_tensor_ptr = output_tensor.GetTensorMutableData<double>();
        if (output_tensor_ptr != nullptr) {
          cuda::Impl_Cast<float, double>(stream, reinterpret_cast<float*>(buffers[output_name]), output_tensor_ptr,
                                         output_dim_sizes[i]);
        }
      }
    }
  }

  // TODO: Add support for CUDA graph for plugin ep.
  /*
  // End CUDA graph capture.
  // Note: One reason we don't put end of graph capture in OnRunEnd() like CUDA EP does is because of cuda stream
  // mentioned in graph capture above, another reason is because OnRunEnd() is not synchronized with OnRunStart() and
  // ExecuteGraph() per inference_session.cc. It's safe to start/end CUDA graph capture in compute_func() here since
  // cuda graph object is maintained by a per thread basis.
  if (cuda_graph_enable_ && !IsGraphCaptured(0)) {
    if (IsGraphCaptureAllowed()) {
      CaptureEnd(0);
      // CUDA work issued to a capturing stream doesn't actually run on the GPU,
      // so run the captured graph here to actually execute the work.
      ORT_RETURN_IF_ERROR(ReplayGraph(0));
    } else {
      IncrementRegularRunCountBeforeGraphCapture();
    }
  }
  */

  return nullptr;
}

void TRTEpEpContextNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  TensorrtComputeStateForEPContext& trt_ep_compute_state = *reinterpret_cast<TensorrtComputeStateForEPContext*>(compute_state);
  (void)trt_ep_compute_state;
  // Do nothing for here.
}
}  // namespace trt_ep
