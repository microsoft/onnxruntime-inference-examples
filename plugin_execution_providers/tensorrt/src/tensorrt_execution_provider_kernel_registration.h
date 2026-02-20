// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"

namespace trt_ep {

size_t GetNumKernels();

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state, OrtKernelRegistry** kernel_registry);

}  // namespace trt_ep
