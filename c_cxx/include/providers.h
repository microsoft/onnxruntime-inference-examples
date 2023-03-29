// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_CUDA
#include "cuda_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "tensorrt_provider_factory.h"
#endif
#ifdef USE_DML
#include "dml_provider_factory.h"
#endif
