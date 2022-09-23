// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cpu/cpu_provider_factory.h"

#ifdef USE_CUDA
#include "cuda/cuda_provider_factory.h"
#endif
#ifdef USE_DNNL
#include "dnnl/dnnl_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "nuphar/nuphar_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "tensorrt/tensorrt_provider_factory.h"
#endif
#ifdef USE_DML
#include "dml/dml_provider_factory.h"
#endif
#ifdef USE_MIGRAPHX
#include "migraphx/migraphx_provider_factory.h"
#endif
