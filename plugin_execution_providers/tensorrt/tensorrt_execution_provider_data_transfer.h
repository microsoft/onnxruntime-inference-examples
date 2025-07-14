// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"

struct TRTEpDataTransfer : OrtDataTransferImpl, ApiPtrs {
  TRTEpDataTransfer(ApiPtrs api_ptrs, std::vector<const OrtMemoryDevice*> device_mem_infos,
                      std::vector<const OrtMemoryDevice*> shared_mem_infos)
      : ApiPtrs(api_ptrs), cuda_gpu_mem_devices_{device_mem_infos}, cuda_pinned_mem_devices_{shared_mem_infos} {
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
  }

  static bool ORT_API_CALL CanCopyImpl(void* this_ptr, const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept;

  // function to copy one or more tensors.
  // implementation can optionally use async copy if a stream is available for the input.
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(void* this_ptr, const OrtValue** src_tensors_ptr,
                                                 OrtValue** dst_tensors_ptr, OrtSyncStream** streams_ptr,
                                                 size_t num_tensors) noexcept;
  static void ORT_API_CALL ReleaseImpl(void* this_ptr) noexcept;

 private:
  std::vector<const OrtMemoryDevice*> cuda_gpu_mem_devices_;
  std::vector<const OrtMemoryDevice*> cuda_pinned_mem_devices_;
};