#pragma once

#include "onnxruntime_cxx_api.h"

// #include "flatbuffers/idl.h"
// #include "ort_trt_int8_cal_table.fbs.h"
#include "make_string.h"
// #include "core/providers/cuda/cuda_pch.h"
// #include "core/common/path_string.h"
// #include "core/framework/murmurhash3.h"

// #include"nv_includes.h"
#include "gsl/narrow"

#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <filesystem>

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

namespace trt_ep {

#define ENFORCE(condition, ...)                          \
  do {                                                   \
    if (!(condition)) {                                  \
      throw std::runtime_error(MakeString(__VA_ARGS__)); \
    }                                                    \
  } while (false)

#define THROW(...) \
  throw std::runtime_error(MakeString(__VA_ARGS__));

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_IF(cond, ...)                                                           \
  do {                                                                                 \
    if ((cond)) {                                                                      \
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MakeString(__VA_ARGS__).c_str()); \
    }                                                                                  \
  } while (0)

#define RETURN_IF_NOT(condition, ...) RETURN_IF(!(condition), __VA_ARGS__)

#define MAKE_STATUS(error_code, msg) \
  Ort::GetApi().CreateStatus(error_code, (msg));

#define THROW_IF_ERROR(expr)                         \
  do {                                               \
    auto _status = (expr);                           \
    if (_status != nullptr) {                        \
      std::ostringstream oss;                        \
      oss << Ort::GetApi().GetErrorMessage(_status); \
      Ort::GetApi().ReleaseStatus(_status);          \
      throw std::runtime_error(oss.str());           \
    }                                                \
  } while (0)

#define RETURN_FALSE_AND_PRINT_IF_ERROR(fn)                            \
  do {                                                                 \
    OrtStatus* status = (fn);                                          \
    if (status != nullptr) {                                           \
      std::cerr << Ort::GetApi().GetErrorMessage(status) << std::endl; \
      return false;                                                    \
    }                                                                  \
  } while (0)

// Helper to release Ort one or more objects obtained from the public C API at the end of their scope.
template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** object_ptr, std::function<void(T*)> release_func)
      : objects_(object_ptr), count_(1), release_func_(release_func) {}

  DeferOrtRelease(T** objects, size_t count, std::function<void(T*)> release_func)
      : objects_(objects), count_(count), release_func_(release_func) {}

  ~DeferOrtRelease() {
    if (objects_ != nullptr && count_ > 0) {
      for (size_t i = 0; i < count_; ++i) {
        if (objects_[i] != nullptr) {
          release_func_(objects_[i]);
          objects_[i] = nullptr;
        }
      }
    }
  }
  T** objects_ = nullptr;
  size_t count_ = 0;
  std::function<void(T*)> release_func_ = nullptr;
};

template <typename T>
using AllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

}  // namespace trt_ep