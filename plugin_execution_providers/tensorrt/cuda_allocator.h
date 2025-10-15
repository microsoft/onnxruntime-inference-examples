// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include "onnxruntime_c_api.h"

using DeviceId = int16_t;

namespace trt_ep {

struct CUDAAllocator : OrtAllocator {
  CUDAAllocator(const OrtMemoryInfo* mem_info, DeviceId device_id) : mem_info_(mem_info), device_id_(device_id) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<CUDAAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<CUDAAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const CUDAAllocator*>(this_)->Info(); };
    OrtAllocator::Reserve = nullptr;
    OrtAllocator::GetStats = nullptr;
    OrtAllocator::AllocOnStream = nullptr;  // Allocate memory, handling usage across different Streams. Not used for TRT EP.
  }

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;
  DeviceId GetDeviceId() const { return device_id_; };

 private:
  CUDAAllocator(const CUDAAllocator&) = delete;
  CUDAAllocator& operator=(const CUDAAllocator&) = delete;

  void CheckDevice(bool throw_when_fail) const;
  void SetDevice(bool throw_when_fail) const;

  DeviceId device_id_;
  const OrtMemoryInfo* mem_info_ = nullptr;
};

struct CUDAPinnedAllocator : OrtAllocator {
  CUDAPinnedAllocator(const OrtMemoryInfo* mem_info) : mem_info_(mem_info) {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<CUDAPinnedAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<CUDAPinnedAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const CUDAPinnedAllocator*>(this_)->Info(); };
    OrtAllocator::Reserve = nullptr;
    OrtAllocator::GetStats = nullptr;
    OrtAllocator::AllocOnStream = nullptr;
  }

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;

  DeviceId GetDeviceId() const { return device_id_; };

 private:
  CUDAPinnedAllocator(const CUDAPinnedAllocator&) = delete;
  CUDAPinnedAllocator& operator=(const CUDAPinnedAllocator&) = delete;

  DeviceId device_id_ = 0;
  const OrtMemoryInfo* mem_info_ = nullptr;
};

}  // namespace trt_ep
