// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cpu_provider_factory.h"

#ifdef USE_CUDA
#include "cuda_provider_factory.h"
#endif
#ifdef USE_DNNL
#include "dnnl_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "nuphar_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "tensorrt_provider_factory.h"
#endif
#ifdef USE_DML
#include "dml_provider_factory.h"
#endif
#ifdef USE_MIGRAPHX
#include "migraphx_provider_factory.h"
#endif
