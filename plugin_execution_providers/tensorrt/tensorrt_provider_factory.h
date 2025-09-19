#pragma once

#include "ep_utils.h"
#include "tensorrt_execution_provider_data_transfer.h"
#include "cuda_allocator.h"

using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;

namespace trt_ep {

///
/// Plugin TensorRT EP factory that can create an OrtEp and return information about the supported hardware devices.
///
struct TensorrtExecutionProviderFactory : public OrtEpFactory, public ApiPtrs {
 public:
  TensorrtExecutionProviderFactory(const char* ep_name, const OrtLogger& default_logger, ApiPtrs apis);

  OrtStatus* CreateMemoryInfoForDevices(int num_devices);

  // CUDA gpu memory and CUDA pinned memory are required for allocator and data transfer, these are the OrtMemoryInfo
  // instance required for that.
  // Current TRT EP implementation uses one default OrtMemoryInfo and one host accessible OrtMemoryInfo per ep device.
  std::unordered_map<uint32_t, MemoryInfoUniquePtr> cuda_gpu_memory_infos;  // device id -> memory info
  std::unordered_map<uint32_t, MemoryInfoUniquePtr> cuda_pinned_memory_infos;

  // Keeps allocators per ep device in factory so they can be shared across sessions.
  std::unordered_map<uint32_t, std::unique_ptr<CUDAAllocator>> cuda_gpu_allocators;  // device id -> allocator
  std::unordered_map<uint32_t, std::unique_ptr<CUDAPinnedAllocator>> cuda_pinned_allocators;

  std::vector<const OrtMemoryDevice*> cuda_gpu_mem_devices;
  std::vector<const OrtMemoryDevice*> cuda_pinned_mem_devices;
  std::unique_ptr<TRTEpDataTransfer> data_transfer_impl;  // data transfer implementation for this factory

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices, size_t num_devices,
                                                         OrtEpDevice** ep_devices, size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr, const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*ep_metadata*/, size_t num_devices,
                                              const OrtSessionOptions* session_options, const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept;

  const std::string ep_name_;              // EP name
  const std::string vendor_{"Nvidia"};     // EP vendor name
  const std::string ep_version_{"0.1.0"};  // EP version
  const OrtLogger& default_logger_;
};
}  // namespace trt_ep