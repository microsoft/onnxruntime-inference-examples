#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include "tensorrt_provider_factory.h"
#include "tensorrt_execution_provider.h"
#include "cuda_allocator.h"

#include <gsl/gsl>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

TensorrtExecutionProviderFactory::TensorrtExecutionProviderFactory(const char* ep_name, ApiPtrs apis)
    : ApiPtrs(apis), ep_name_{ep_name} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVersion = GetVersionImpl;

  GetSupportedDevices = GetSupportedDevicesImpl;

  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  CreateDataTransfer = CreateDataTransferImpl;
}

const char* ORT_API_CALL TensorrtExecutionProviderFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const TensorrtExecutionProviderFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

const char* ORT_API_CALL TensorrtExecutionProviderFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const TensorrtExecutionProviderFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

const char* ORT_API_CALL TensorrtExecutionProviderFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const TensorrtExecutionProviderFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

OrtStatus* ORT_API_CALL TensorrtExecutionProviderFactory::GetSupportedDevicesImpl(
                                                       OrtEpFactory* this_ptr,
                                                       const OrtHardwareDevice* const* devices,
                                                       size_t num_devices,
                                                       OrtEpDevice** ep_devices,
                                                       size_t max_ep_devices,
                                                       size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<TensorrtExecutionProviderFactory*>(this_ptr);

  std::vector<const OrtMemoryDevice*> cuda_gpu_mem_devices;
  std::vector<const OrtMemoryDevice*> cuda_pinned_mem_devices;
  int GPU_cnt = 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    // C API
    const OrtHardwareDevice& device = *devices[i];
    if (factory->ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      if (GPU_cnt > 0) {
        continue;
      }
      GPU_cnt++;
      // These can be returned as nullptr if you have nothing to add.
      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api.CreateKeyValuePairs(&ep_options);

      // The ep options can be provided here as default values.
      // Users can also call SessionOptionsAppendExecutionProvider_V2 C API with provided ep options to override.
      factory->ort_api.AddKeyValuePair(ep_metadata, "gpu_type", "data center"); // random example using made up values
      factory->ort_api.AddKeyValuePair(ep_options, "trt_builder_optimization_level", "3");

      // OrtEpDevice copies ep_metadata and ep_options.
      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                 &ep_device);

      factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      uint32_t vendor_id = factory->ort_api.HardwareDevice_VendorId(&device);
      //uint32_t device_id = factory->ort_api.HardwareDevice_DeviceId(&device);
      uint32_t device_id = 0;
      
      // CUDA allocator OrtMemoryInfo
      OrtMemoryInfo* mem_info = nullptr;
      status = factory->ort_api.CreateMemoryInfo_V2("Cuda", OrtMemoryInfoDeviceType_GPU, vendor_id, device_id, OrtDeviceMemoryType_DEFAULT,
                                               /*alignment*/ 0, OrtAllocatorType::OrtDeviceAllocator, &mem_info);

      assert(status == nullptr);  // should never fail.
      MemoryInfoUniquePtr cuda_gpu_memory_info = MemoryInfoUniquePtr(mem_info, factory->ort_api.ReleaseMemoryInfo);

      // CUDA PINNED allocator OrtMemoryInfo
      // HOST_ACCESSIBLE memory should use the non-CPU device type.
      mem_info = nullptr;
      status = factory->ort_api.CreateMemoryInfo_V2("CudaPinned", OrtMemoryInfoDeviceType_GPU, vendor_id, device_id, OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                           /*alignment*/ 0, OrtAllocatorType::OrtDeviceAllocator, &mem_info);

      assert(status == nullptr);  // should never fail.
      MemoryInfoUniquePtr cuda_pinned_memory_info = MemoryInfoUniquePtr(mem_info, factory->ort_api.ReleaseMemoryInfo);

      // Register the allocator info required by TRT EP.
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, cuda_gpu_memory_info.get()));
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, cuda_pinned_memory_info.get()));

      // Get memory device from memory info for gpu data transfer
      cuda_gpu_mem_devices.push_back(factory->ep_api.MemoryInfo_GetMemoryDevice(cuda_gpu_memory_info.get()));
      cuda_pinned_mem_devices.push_back(factory->ep_api.MemoryInfo_GetMemoryDevice(cuda_pinned_memory_info.get()));

      factory->SetDefaultGpuMemInfo(std::move(cuda_gpu_memory_info), device_id);
      factory->SetHostAccessibleMemInfo(std::move(cuda_pinned_memory_info), device_id);

      ep_devices[num_ep_devices++] = ep_device;
    }
  
  // C++ API equivalent. Throws on error.
  //{
  //  Ort::ConstHardwareDevice device(devices[i]);
  //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
  //    Ort::KeyValuePairs ep_metadata;
  //    Ort::KeyValuePairs ep_options;
  //    ep_metadata.Add("version", "0.1");
  //    ep_options.Add("trt_builder_optimization_level", "3");
  //    Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
  //    ep_devices[num_ep_devices++] = ep_device.release();
  //  }
  //}
  }

    // Create gpu data transfer
  auto data_transfer_impl = std::make_unique<TRTEpDataTransfer>(static_cast<const ApiPtrs&>(*factory),
                                                                cuda_gpu_mem_devices,    // device memory
                                                                cuda_pinned_mem_devices  // shared memory
                                                               );

  return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtExecutionProviderFactory::CreateEpImpl(
                                            OrtEpFactory* this_ptr,
                                            _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                            _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                            _In_ size_t num_devices,
                                            _In_ const OrtSessionOptions* session_options,
                                            _In_ const OrtLogger* logger, _Out_ OrtEp** ep) noexcept {
  auto* factory = static_cast<TensorrtExecutionProviderFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for GPU and only expected to be selected for one GPU
    // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
    // the EP has been selected for.
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "TensorRT EP only supports selection for one device.");
  }

  // Create the execution provider
  RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                     "Creating TensorRT EP", ORT_FILE, __LINE__, __FUNCTION__));

  // use properties from the device and ep_metadata if needed
  // const OrtHardwareDevice* device = devices[0];
  // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

  auto trt_ep = std::make_unique<TensorrtExecutionProvider>(*factory, factory->ep_name_, *session_options, *logger);

  *ep = trt_ep.release();
  return nullptr;
}

void ORT_API_CALL TensorrtExecutionProviderFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  TensorrtExecutionProvider* trt_ep = static_cast<TensorrtExecutionProvider*>(ep);
  delete trt_ep;
}

OrtStatus* ORT_API_CALL TensorrtExecutionProviderFactory::CreateAllocatorImpl(
                                                              OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                                                              const OrtKeyValuePairs* /*allocator_options*/,
                                                              OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<TensorrtExecutionProviderFactory*>(this_ptr);
  *allocator = nullptr;

  // NOTE: The factory implementation can return a shared OrtAllocator* instead of creating a new instance on each call.
  //       To do this just make ReleaseAllocatorImpl a no-op.

  // NOTE: If OrtMemoryInfo has allocator type (call MemoryInfoGetType) of OrtArenaAllocator, an ORT BFCArena
  //       will be added to wrap the returned OrtAllocator. The EP is free to implement its own arena, and if it
  //       wants to do this the OrtMemoryInfo MUST be created with an allocator type of OrtDeviceAllocator.

  // NOTE: The OrtMemoryInfo pointer should only ever be coming straight from an OrtEpDevice, and pointer based
  // matching should work.
  
  uint32_t device_id = 0;

  if (factory.GetDeviceIdForDefaultGpuMemInfo(memory_info, &device_id)) {
    // create a CUDA allocator
    auto cuda_allocator = std::make_unique<CUDAAllocator>(memory_info, static_cast<DeviceId>(device_id));
    *allocator = cuda_allocator.release();
  } else if (factory.GetDeviceIdForHostAccessibleMemInfo(memory_info, &device_id)) {
    // create a CUDA PINNED allocator
    auto cuda_pinned_allocator = std::make_unique<CUDAPinnedAllocator>(memory_info);
    *allocator = cuda_pinned_allocator.release();
  } else {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                        "Value did not come directly from an OrtEpDevice returned by this factory.");
  }

  return nullptr;
}

void ORT_API_CALL TensorrtExecutionProviderFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this*/,
                                                                         OrtAllocator* allocator) noexcept {
  delete static_cast<CUDAAllocator*>(allocator);
}

OrtStatus* ORT_API_CALL TensorrtExecutionProviderFactory::CreateDataTransferImpl(
                                                                 OrtEpFactory* this_ptr,
                                                                 OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<TensorrtExecutionProviderFactory*>(this_ptr);
  *data_transfer = factory.data_transfer_impl_.get();

  return nullptr;
}

bool TensorrtExecutionProviderFactory::GetDeviceIdForDefaultGpuMemInfo(const OrtMemoryInfo* mem_info, uint32_t* device_id) const {
  auto iter = cuda_gpu_memory_info_to_device_id_map_.find(mem_info);
  if (iter != cuda_gpu_memory_info_to_device_id_map_.end()) {
    *device_id = iter->second;
    return true;
  }
  return false;
}

const OrtMemoryInfo* TensorrtExecutionProviderFactory::GetDefaultGpuMemInfoForDeviceId(uint32_t device_id) const {
  auto iter = device_id_to_cuda_gpu_memory_info_map_.find(device_id);
  if (iter != device_id_to_cuda_gpu_memory_info_map_.end()) {
    return iter->second;
  }
  return nullptr;
}

void TensorrtExecutionProviderFactory::SetDefaultGpuMemInfo(MemoryInfoUniquePtr mem_info, uint32_t device_id) {
  cuda_gpu_memory_info_to_device_id_map_[mem_info.get()] = device_id;
  device_id_to_cuda_gpu_memory_info_map_[device_id] = mem_info.get();
  cuda_gpu_memory_infos_.push_back(std::move(mem_info));
}

bool TensorrtExecutionProviderFactory::GetDeviceIdForHostAccessibleMemInfo(const OrtMemoryInfo* mem_info, uint32_t* device_id) const {
  auto iter = cuda_pinned_memory_info_to_device_id_map_.find(mem_info);
  if (iter != cuda_pinned_memory_info_to_device_id_map_.end()) {
    *device_id = iter->second;
    return true;
  }
  return false;
}

const OrtMemoryInfo* TensorrtExecutionProviderFactory::GetHostAccessibleMemInfoForDeviceId(uint32_t device_id) const {
  auto iter = device_id_to_cuda_pinned_memory_info_map_.find(device_id);
  if (iter != device_id_to_cuda_pinned_memory_info_map_.end()) {
    return iter->second;
  }
  return nullptr;
}

void TensorrtExecutionProviderFactory::SetHostAccessibleMemInfo(MemoryInfoUniquePtr mem_info, uint32_t device_id) {
  cuda_pinned_memory_info_to_device_id_map_[mem_info.get()] = device_id;
  device_id_to_cuda_pinned_memory_info_map_[device_id] = mem_info.get();
  cuda_pinned_memory_infos_.push_back(std::move(mem_info));
}

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ort_ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<TensorrtExecutionProviderFactory>(registration_name, ApiPtrs{*ort_api, *ort_ep_api, *model_editor_api});

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete factory;
  return nullptr;
}

}  // extern "C"
