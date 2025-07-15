#pragma once

#include "ep_utils.h"
#include "tensorrt_execution_provider_data_transfer.h"

using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;

///
/// Plugin TensorRT EP factory that can create an OrtEp and return information about the supported hardware devices.
///
struct TensorrtExecutionProviderFactory : public OrtEpFactory, public ApiPtrs {
 public:
  TensorrtExecutionProviderFactory(const char* ep_name, ApiPtrs apis);

  const OrtMemoryInfo* GetDefaultGpuMemInfoForDeviceId(uint32_t device_id) const;

  const OrtMemoryInfo* GetHostAccessibleMemInfoForDeviceId(uint32_t device_id) const;

  OrtStatus* CreateMemoryInfoForDevices(int num_devices);

  // CUDA gpu memory and CUDA pinned memory are required for allocator and data transfer, these are the OrtMemoryInfo
  // instance required for that.
  // Current TRT EP implementation uses one default OrtMemoryInfo and one host accessible OrtMemoryInfo per ep device.
  std::vector<MemoryInfoUniquePtr> cuda_gpu_memory_infos;
  std::vector<MemoryInfoUniquePtr> cuda_pinned_memory_infos;
  std::unordered_map<uint32_t, const OrtMemoryInfo*> device_id_to_cuda_gpu_memory_info_map;  // device id -> OrtMemoryInfo

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL TensorrtExecutionProviderFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

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

  void SetGPUDataTransfer(std::unique_ptr<TRTEpDataTransfer> gpu_data_transfer);

  const std::string ep_name_;           // EP name
  const std::string vendor_{"Nvidia"};  // EP vendor name
  const std::string ep_version_{"0.1.0"};  // EP version

  std::unique_ptr<TRTEpDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory
};