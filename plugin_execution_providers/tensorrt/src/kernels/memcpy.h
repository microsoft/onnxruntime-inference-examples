// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

struct OrtKernelImpl;
struct OrtKernelInfo;
struct OrtKernelContext;
struct OrtStatus;

namespace trt_ep {

struct MemcpyKernelBase : public OrtKernelImpl {
  // Base class for MemcpyFromHost and MemcpyToHost to share common code.
 protected:
  MemcpyKernelBase(const OrtKernelInfo* info, void* state) : OrtKernelImpl {}, info_(info), state_(state) {}

  template <typename T>
  static OrtStatus* CreateImpl(const OrtKernelInfo* info, void* state, /*out*/ OrtKernelImpl*& kernel) noexcept;

  template <typename T>
  static void ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

  const OrtKernelInfo* info_;
  void* state_;  // Custom state passed from OrtEp
};

struct MemcpyFromHost : public MemcpyKernelBase {
 private:
  struct PrivateTag {};  // Used to prevent use of public constructor (use static MemcpyFromHost::Create())
                         // Need to make the constructor public for std::make_unique().

  // Allow base template helper to access PrivateTag
  friend struct MemcpyKernelBase;

 public:
  MemcpyFromHost(const OrtKernelInfo* info, void* state, PrivateTag) : MemcpyKernelBase(info, state) {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
  };

  static OrtStatus* Create(const OrtKernelInfo* info, void* state,
                           /*out*/ OrtKernelImpl*& kernel) noexcept {
    return CreateImpl<MemcpyFromHost>(info, state, kernel);
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    MemcpyKernelBase::ReleaseImpl<MemcpyFromHost>(this_ptr);
  };
};

struct MemcpyToHost : public MemcpyKernelBase {
 private:
  struct PrivateTag {};  // Used to prevent use of public constructor (use static MemcpyFromHost::Create())
                         // Need to make the constructor public for std::make_unique().

  // Allow base template helper to access PrivateTag
  friend struct MemcpyKernelBase;

 public:
  MemcpyToHost(const OrtKernelInfo* info, void* state, PrivateTag) : MemcpyKernelBase(info, state) {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
  };

  static OrtStatus* Create(const OrtKernelInfo* info, void* state,
                           /*out*/ OrtKernelImpl*& kernel) noexcept {
    return CreateImpl<MemcpyToHost>(info, state, kernel);
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    MemcpyKernelBase::ReleaseImpl<MemcpyToHost>(this_ptr);
  };
};

}