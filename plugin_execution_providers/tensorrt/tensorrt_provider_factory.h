#pragma once

#include "tensorrt_execution_provider_utils.h"
#include "tensorrt_execution_provider_data_transfer.h"

///
/// Plugin TensorRT EP factory that can create an OrtEp and return information about the supported hardware devices.
///
struct TensorrtExecutionProviderFactory : public OrtEpFactory, public ApiPtrs {
 public:
  TensorrtExecutionProviderFactory(const char* ep_name, ApiPtrs apis);
  OrtMemoryInfo* GetDefaultMemInfo() const;

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;

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

  const std::string ep_name_;           // EP name
  const std::string vendor_{"Nvidia"};  // EP vendor name

  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
  //MemoryInfoUniquePtr cpu_memory_info_;

  // GPU memory and pinned/shared memory are required for data transfer, these are the
  // OrtMemoryInfo instance required for that.
  MemoryInfoUniquePtr default_gpu_memory_info_;
  MemoryInfoUniquePtr host_accessible_gpu_memory_info_;

  std::unique_ptr<TRTEpDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory
};