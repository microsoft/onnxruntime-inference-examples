// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------
//
template <typename ERRTYPE>
const char* CudaErrString(ERRTYPE) {
  THROW();
}

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, OrtStatus*> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg, const char* file, const int line) {
  if (retCode != successCode) {
    try {
      int currentCudaDevice = -1;
      cudaGetDevice(&currentCudaDevice);
      cudaGetLastError();  // clear last CUDA error
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=? ; file=%s ; line=%d ; expr=%s; %s",
               libName, (int)retCode, CudaErrString(retCode), currentCudaDevice,
               // hostname,
               file, line, exprString, msg);
      if constexpr (THRW) {
        // throw an exception with the error info
        THROW(str);
      } else {
        return MAKE_STATUS(ORT_EP_FAIL, str);
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since CUDA code sometimes hangs in destruction, so we'd never get to see the error
      if constexpr (THRW) {
        THROW(e.what());
      } else {
        return MAKE_STATUS(ORT_EP_FAIL, e.what());
      }
    }
  }
  if constexpr (!THRW) {
    return nullptr;
  }
}

#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
#define CUDA_CALL_THROW(expr) (CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
