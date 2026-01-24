// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorrt_execution_provider_stream_support.h"
#include "tensorrt_provider_factory.h"
#include "tensorrt_execution_provider.h"

#include "cuda/cuda_common.h"
#include "cuda/cuda_call.h"

namespace trt_ep {

//
// TrtSyncStreamImpl implementation
//

TrtSyncStreamImpl::TrtSyncStreamImpl(TensorrtExecutionProviderFactory& factory, const OrtEp* ep, uint32_t device_id, const OrtKeyValuePairs* /*stream_options*/)
    : ApiPtrs(factory), ep_{ep}, factory_{&factory} {
  ort_version_supported = ORT_API_VERSION;
  CreateNotification = CreateNotificationImpl;
  GetHandle = GetHandleImpl;
  Flush = FlushImpl;
  OnSessionRunEnd = OnSessionRunEndImpl;
  Release = ReleaseImpl;

  const TensorrtExecutionProvider* trt_ep = static_cast<const TensorrtExecutionProvider*>(ep_);
  if (trt_ep->external_stream_) {
    stream_ = trt_ep->stream_;
    own_stream_ = false;
  } else {
    CUDA_CALL_THROW(cudaSetDevice(static_cast<int>(device_id)));
    cudaStream_t stream = nullptr;
    CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    stream_ = stream;
    own_stream_ = true;
  }
}

/*static*/
OrtStatus* ORT_API_CALL TrtSyncStreamImpl::CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                                                  _Outptr_ OrtSyncNotificationImpl** notification) noexcept {
  auto& impl = *static_cast<TrtSyncStreamImpl*>(this_ptr);

  std::unique_ptr<TrtSyncNotificationImpl> trt_sync_notification;
  RETURN_IF_ERROR(TrtSyncNotificationImpl::Create(impl.stream_, impl, trt_sync_notification));

  *notification = trt_sync_notification.release();
  return nullptr;
}

/*static*/
void* ORT_API_CALL TrtSyncStreamImpl::GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<TrtSyncStreamImpl*>(this_ptr);
  return static_cast<void*>(impl.stream_);
}

/*static*/
OrtStatus* ORT_API_CALL TrtSyncStreamImpl::FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<TrtSyncStreamImpl*>(this_ptr);

  // only flush when we own the stream, not external
  if (impl.own_stream_) CUDA_CALL_THROW(cudaStreamSynchronize(static_cast<cudaStream_t>(impl.stream_)));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TrtSyncStreamImpl::OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  return nullptr;
}

// callback for EP library to release any internal state
/*static*/
void ORT_API_CALL TrtSyncStreamImpl::ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  delete static_cast<TrtSyncStreamImpl*>(this_ptr);
}

//
// Notification support
//

/*static*/
OrtStatus* TrtSyncNotificationImpl::Create(cudaStream_t stream, const ApiPtrs& apis,
                                           std::unique_ptr<TrtSyncNotificationImpl>& notification) {
  auto trt_sync_notification = std::make_unique<TrtSyncNotificationImpl>(stream, apis);
  CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&trt_sync_notification->event_, cudaEventDisableTiming));

  notification = std::move(trt_sync_notification);

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TrtSyncNotificationImpl::ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  auto& impl = *static_cast<TrtSyncNotificationImpl*>(this_ptr);
  CUDA_RETURN_IF_ERROR(cudaEventRecord(impl.event_, impl.stream_));

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TrtSyncNotificationImpl::WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                                  _In_ OrtSyncStream* stream) noexcept {
  auto& impl = *static_cast<TrtSyncNotificationImpl*>(this_ptr);
  void* handle = impl.ort_api.SyncStream_GetHandle(stream);
  CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(static_cast<cudaStream_t>(handle), impl.event_));

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TrtSyncNotificationImpl::WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  auto& impl = *static_cast<TrtSyncNotificationImpl*>(this_ptr);
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(impl.event_));

  return nullptr;
}

/*static*/
void ORT_API_CALL TrtSyncNotificationImpl::ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  delete static_cast<TrtSyncNotificationImpl*>(this_ptr);
}
}  // namespace trt_ep
