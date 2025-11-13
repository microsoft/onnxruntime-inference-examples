// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>

#include "onnxruntime_ep_device_ep_metadata_keys.h"

#include "ep.h"
#include "plugin_ep_utils.h"

BasicPluginEpFactory::BasicPluginEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                                           const OrtModelEditorApi& model_editor_api,
                                           const OrtLogger& /*default_logger*/)
    : OrtEpFactory{}, ort_api_(ort_api), ep_api_(ep_api), model_editor_api_(model_editor_api) {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;

  GetSupportedDevices = GetSupportedDevicesImpl;

  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  CreateDataTransfer = CreateDataTransferImpl;

  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}

BasicPluginEpFactory::~BasicPluginEpFactory() = default;

/*static*/
const char* ORT_API_CALL BasicPluginEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const BasicPluginEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL BasicPluginEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const BasicPluginEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL BasicPluginEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const BasicPluginEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL BasicPluginEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const BasicPluginEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL BasicPluginEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                      const OrtHardwareDevice* const* devices,
                                                                      size_t num_devices, OrtEpDevice** ep_devices,
                                                                      size_t max_ep_devices,
                                                                      size_t* p_num_ep_devices) noexcept {
  EP_API_IMPL_BEGIN

  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<BasicPluginEpFactory*>(this_ptr);

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    Ort::ConstHardwareDevice device(devices[i]);
    if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      Ort::KeyValuePairs ep_metadata;
      // Implementations can add relevant EP metadata here.
      Ort::KeyValuePairs ep_options;
      // Implementations can add relevant EP options here.
      Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
      ep_devices[num_ep_devices++] = ep_device.release();
    }
  }

  return nullptr;

  EP_API_IMPL_END
}

/*static*/
OrtStatus* ORT_API_CALL BasicPluginEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                           const OrtHardwareDevice* const* /*devices*/,
                                                           const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                           size_t num_devices, const OrtSessionOptions* session_options,
                                                           const OrtLogger* logger, OrtEp** ep) noexcept {
  EP_API_IMPL_BEGIN

  auto* factory = static_cast<BasicPluginEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for CPU and only expected to be selected for one device
    return factory->ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "BasicPluginEpFactory only supports selection for one device.");
  }

  BasicPluginEp::Config config = {};
  auto actual_ep = std::make_unique<BasicPluginEp>(*factory, config, *logger);

  *ep = actual_ep.release();
  return nullptr;

  EP_API_IMPL_END
}

/*static*/
void ORT_API_CALL BasicPluginEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  delete static_cast<BasicPluginEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL BasicPluginEpFactory::CreateAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                                                  const OrtMemoryInfo* /*memory_info*/,
                                                                  const OrtKeyValuePairs* /*allocator_options*/,
                                                                  OrtAllocator** allocator) noexcept {
  // Don't support custom allocators in this example for simplicity.
  *allocator = nullptr;
  return nullptr;
}

/*static*/
void ORT_API_CALL BasicPluginEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                                             OrtAllocator* /*allocator*/) noexcept {
  // Do nothing.
}

/*static*/
OrtStatus* ORT_API_CALL BasicPluginEpFactory::CreateDataTransferImpl(OrtEpFactory* /*this_ptr*/,
                                                                     OrtDataTransferImpl** data_transfer) noexcept {
  // Don't support data transfer in this example for simplicity.
  *data_transfer = nullptr;
  return nullptr;
}

/*static*/
bool ORT_API_CALL BasicPluginEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept { return false; }

/*static*/
OrtStatus* ORT_API_CALL BasicPluginEpFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* /*this_ptr*/,
                                                                            const OrtMemoryDevice* /*memory_device*/,
                                                                            const OrtKeyValuePairs* /*stream_options*/,
                                                                            OrtSyncStreamImpl** stream) noexcept {
  // Don't support sync streams in this example.
  *stream = nullptr;
  return nullptr;
}
