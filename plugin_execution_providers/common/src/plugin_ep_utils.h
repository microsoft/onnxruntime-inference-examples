// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#define RETURN_IF_ERROR(status_expr)   \
  do {                                 \
    Ort::Status status{(status_expr)}; \
    if (!status.IsOK()) {              \
      return status.release();         \
    }                                  \
  } while (0)

#define RETURN_IF(cond, msg)                \
  do {                                      \
    if ((cond)) {                           \
      const char* msg_cstr = (msg);         \
      Ort::Status status{msg, ORT_EP_FAIL}; \
      return status.release();              \
    }                                       \
  } while (0)

#define RETURN_IF_NOT(cond, msg) \
  RETURN_IF(!(cond), msg)

// Ignores an OrtStatus* while taking ownership of it so that it does not get leaked.
#define IGNORE_ORTSTATUS(status_expr)   \
  do {                                  \
    OrtStatus* _status = (status_expr); \
    Ort::Status _ignored{_status};      \
  } while (false)

#ifdef _WIN32
#define EP_WSTR(x) L##x
#define EP_FILE_INTERNAL(x) EP_WSTR(x)
#define EP_FILE EP_FILE_INTERNAL(__FILE__)
#else
#define EP_FILE __FILE__
#endif

#define LOG(ort_api, ort_logger_ptr, level, ...)                                                                \
  do {                                                                                                          \
    std::ostringstream ss;                                                                                      \
    ss << __VA_ARGS__;                                                                                          \
    IGNORE_ORTSTATUS((ort_api).Logger_LogMessage((ort_logger_ptr), ORT_LOGGING_LEVEL_##level, ss.str().c_str(), \
                                                 EP_FILE, __LINE__, __FUNCTION__));                             \
  } while (false)

#define RETURN_ERROR(code, ...)                       \
  do {                                                \
    std::ostringstream ss;                            \
    ss << __VA_ARGS__;                                \
    OrtErrorCode error_code = (code);                 \
    Ort::Status status(ss.str().c_str(), error_code); \
    return status.release();                          \
  } while (false)

#define EP_API_IMPL_BEGIN \
  try {
#define EP_API_IMPL_END                                           \
  }                                                               \
  catch (const Ort::Exception& ex) {                              \
    Ort::Status status(ex);                                       \
    return status.release();                                      \
  }                                                               \
  catch (const std::exception& ex) {                              \
    Ort::Status status(ex.what(), ORT_EP_FAIL);                   \
    return status.release();                                      \
  }                                                               \
  catch (...) {                                                   \
    Ort::Status status("Caught unknown exception.", ORT_EP_FAIL); \
    return status.release();                                      \
  }

// Returns true (via output parameter) if the given OrtValueInfo represents a float tensor.
inline void IsFloatTensor(Ort::ConstValueInfo value_info, bool& result) {
  result = false;

  auto type_info = value_info.TypeInfo();
  ONNXType onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = type_shape.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return;
  }
  result = true;
}

// Gets the tensor shape from `value_info`. Returns std::nullopt if `value_info` is not a tensor.
inline std::optional<std::vector<int64_t>> GetTensorShape(Ort::ConstValueInfo value_info) {
  const auto type_info = value_info.TypeInfo();
  const auto onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return std::nullopt;
  }

  const auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetShape();
}
