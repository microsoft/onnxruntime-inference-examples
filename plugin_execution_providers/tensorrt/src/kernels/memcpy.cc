// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include "memcpy.h"
#include <cuda_runtime.h>

namespace trt_ep {

template <typename T>
OrtStatus* MemcpyKernelBase::CreateImpl(const OrtKernelInfo* info, void* state,
                                        /*out*/ OrtKernelImpl*& kernel) noexcept {
  try {
    auto p = std::make_unique<T>(info, state, typename T::PrivateTag{});
    kernel = p.release();
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  } catch (...) {
    Ort::Status status("Unknown exception in MemcpyKernelBase::Create", ORT_EP_FAIL);
    return status.release();
  }
}

template <typename T>
static void MemcpyKernelBase::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<T*>(this_ptr);
}

OrtStatus* MemcpyFromHost::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  try {
    const OrtApi& ort_api = Ort::GetApi();
    const OrtValue* input_tensor = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_ctx, 0, &input_tensor));

    // Get tensor shape and type
    OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input_tensor, &tensor_info));

    size_t element_count = 0;
    RETURN_IF_ERROR(ort_api.GetTensorShapeElementCount(tensor_info, &element_count));

    ONNXTensorElementDataType element_type;
    RETURN_IF_ERROR(ort_api.GetTensorElementType(tensor_info, &element_type));

    size_t num_dims = 0;
    RETURN_IF_ERROR(ort_api.GetDimensionsCount(tensor_info, &num_dims));

    std::vector<int64_t> dims(num_dims);
    RETURN_IF_ERROR(ort_api.GetDimensions(tensor_info, dims.data(), num_dims));
    ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);

    // Get output tensor
    OrtValue* output_tensor = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetOutput(kernel_ctx, 0, dims.data(), num_dims, &output_tensor));

    // Get data pointers
    const void* input_data = nullptr;
    void* output_data = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorData(input_tensor, &input_data));
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(output_tensor, &output_data));

    // Calculate size in bytes
    size_t bytes = 0;
    RETURN_IF_ERROR(ort_api.GetTensorSizeInBytes(input_tensor, &bytes));

    // Get CUDA stream from kernel context
    void* cuda_stream = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetGPUComputeStream(kernel_ctx, &cuda_stream));
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

    // Copy from host (CPU) to device (GPU) asynchronously
    cudaError_t cuda_err = cudaMemcpyAsync(output_data, input_data, bytes, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess) {
      return ort_api.CreateStatus(ORT_EP_FAIL, cudaGetErrorString(cuda_err));
    }

    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}

OrtStatus* MemcpyToHost::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  try {
    const OrtApi& ort_api = Ort::GetApi();
    const OrtValue* input_tensor = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_ctx, 0, &input_tensor));

    // Get tensor shape and type
    OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input_tensor, &tensor_info));

    size_t num_dims = 0;
    RETURN_IF_ERROR(ort_api.GetDimensionsCount(tensor_info, &num_dims));

    std::vector<int64_t> dims(num_dims);
    RETURN_IF_ERROR(ort_api.GetDimensions(tensor_info, dims.data(), num_dims));
    ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);

    // Get output tensor
    OrtValue* output_tensor = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetOutput(kernel_ctx, 0, dims.data(), num_dims, &output_tensor));

    // Get data pointers
    const void* input_data = nullptr;
    void* output_data = nullptr;
    RETURN_IF_ERROR(ort_api.GetTensorData(input_tensor, &input_data));
    RETURN_IF_ERROR(ort_api.GetTensorMutableData(output_tensor, &output_data));

    // Calculate size in bytes
    size_t bytes = 0;
    RETURN_IF_ERROR(ort_api.GetTensorSizeInBytes(input_tensor, &bytes));

    // Get CUDA stream from kernel context
    void* cuda_stream = nullptr;
    RETURN_IF_ERROR(ort_api.KernelContext_GetGPUComputeStream(kernel_ctx, &cuda_stream));
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

    // Copy from device (GPU) to host (CPU) asynchronously
    cudaError_t cuda_err = cudaMemcpyAsync(output_data, input_data, bytes, cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess) {
      return ort_api.CreateStatus(ORT_EP_FAIL, cudaGetErrorString(cuda_err));
    }

    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    /*version*/ 1,  // Equivalent to start_version: 14, end_version: 14 (inclusive)
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemType::OrtMemTypeCPUInput)
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    MemcpyFromHost)

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    /*version*/ 1,  // Equivalent to start_version: 14, end_version: 14 (inclusive)
    (Ort::KernelDefBuilder()
         .SetOutputMemType(0, OrtMemType::OrtMemTypeCPUOutput)
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    MemcpyToHost)

}  // namespace trt_ep