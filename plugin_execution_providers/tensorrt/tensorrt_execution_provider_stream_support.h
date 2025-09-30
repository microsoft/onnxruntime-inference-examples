// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "tensorrt_provider_factory.h"
#include "ep_utils.h"

#include <cuda_runtime_api.h>

namespace trt_ep {
//
// Class implementing Stream support for synchronization.
//
struct TrtSyncStreamImpl : public OrtSyncStreamImpl, public ApiPtrs {
  TrtSyncStreamImpl(TensorrtExecutionProviderFactory& factory,
                    const OrtEp* ep,
                    uint32_t device_id,
                    const OrtKeyValuePairs* /*stream_options*/);

 private:
  static OrtStatus* ORT_API_CALL CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                                        _Outptr_ OrtSyncNotificationImpl** sync_notification) noexcept;
  static void* ORT_API_CALL GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

  // EP instance if the stream is being created internally for inferencing.
  // nullptr when the stream is created outside of an inference session for data copies.
  const OrtEp* ep_;
  TensorrtExecutionProviderFactory* factory_{nullptr};

  cudaStream_t stream_{nullptr};
  bool own_stream_{true};
};

//
// Class implementing synchronization notification support.
//
struct TrtSyncNotificationImpl : public OrtSyncNotificationImpl, public ApiPtrs {
  static OrtStatus* Create(cudaStream_t stream, const ApiPtrs& apis,
                           std::unique_ptr<TrtSyncNotificationImpl>& notification);

  TrtSyncNotificationImpl(cudaStream_t stream, const ApiPtrs& apis) : stream_(stream), ApiPtrs(apis) {
    ort_version_supported = ORT_API_VERSION;
    Activate = ActivateImpl;
    Release = ReleaseImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                  _In_ OrtSyncStream* stream) noexcept;
  static OrtStatus* ORT_API_CALL WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

  cudaStream_t& stream_;
  cudaEvent_t event_;
};
}  // namespace trt_ep
