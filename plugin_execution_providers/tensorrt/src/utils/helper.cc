// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#include <Windows.h>
#include <assert.h>
#endif

#ifdef ORT_NO_EXCEPTIONS
#if defined(__ANDROID__)
#include <android/log.h>
#else
#include <iostream>
#endif
#endif

#include <string>
#include "ep_utils.h"

#ifdef _WIN32
std::string ToUTF8String(std::wstring_view s) {
  if (s.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
    THROW("length overflow");

  const int src_len = static_cast<int>(s.size() + 1);
  const int len = WideCharToMultiByte(CP_UTF8, 0, s.data(), src_len, nullptr, 0, nullptr, nullptr);
  assert(len > 0);
  std::string ret(static_cast<size_t>(len) - 1, '\0');
#pragma warning(disable : 4189)
  const int r = WideCharToMultiByte(CP_UTF8, 0, s.data(), src_len, (char*)ret.data(), len, nullptr, nullptr);
  assert(len == r);
#pragma warning(default : 4189)
  return ret;
}

std::wstring ToWideString(std::string_view s) {
  if (s.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
    THROW("length overflow");

  const int src_len = static_cast<int>(s.size() + 1);
  const int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), src_len, nullptr, 0);
  assert(len > 0);
  std::wstring ret(static_cast<size_t>(len) - 1, '\0');
#pragma warning(disable : 4189)
  const int r = MultiByteToWideChar(CP_UTF8, 0, s.data(), src_len, (wchar_t*)ret.data(), len);
  assert(len == r);
#pragma warning(default : 4189)
  return ret;
}
#endif  // #ifdef _WIN32
