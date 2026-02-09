// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_utils.h"

namespace trt_ep {

struct MemcpyFromHost : public OrtKernelImpl {
 private:
  struct PrivateTag {};  // Used to prevent use of public constructor (use static MemcpyFromHost::Create())
                         // Need to make the constructor public for std::make_unique().
 public:
  static OrtStatus* Create(const OrtKernelInfo* info, void* state, /*out*/ OrtKernelImpl*& kernel);

  MemcpyFromHost(const OrtKernelInfo* info, void* state, PrivateTag);

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<MemcpyFromHost*>(this_ptr);
  };

 private:
  const OrtKernelInfo* info_;
  void* state_;  // Custom state passed from OrtEp
};

struct MemcpyToHost : public OrtKernelImpl {
 private:
  struct PrivateTag {};  // Used to prevent use of public constructor (use static MemcpyToHost::Create())
                         // Need to make the constructor public for std::make_unique().
 public:
  static OrtStatus* Create(const OrtKernelInfo* info, void* state, /*out*/ OrtKernelImpl*& kernel);

  MemcpyToHost(const OrtKernelInfo* info, void* state, PrivateTag);

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<MemcpyToHost*>(this_ptr);
  };

 private:
  const OrtKernelInfo* info_;
  void* state_;  // Custom state passed from OrtEp
};

}