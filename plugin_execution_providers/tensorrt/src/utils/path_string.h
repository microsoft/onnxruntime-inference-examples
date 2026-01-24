// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <type_traits>

// for std::tolower or std::towlower
#ifdef _WIN32
#include <cwctype>
#else
#include <cctype>
#endif

#include "onnxruntime_c_api.h"

// char type for filesystem paths
using PathChar = ORTCHAR_T;
// string type for filesystem paths
using PathString = std::basic_string<PathChar>;

inline std::string ToUTF8String(const std::string& s) { return s; }
#ifdef _WIN32
/**
 * Convert a wide character string to a UTF-8 string
 */
std::string ToUTF8String(std::wstring_view s);
inline std::string ToUTF8String(const wchar_t* s) {
  return ToUTF8String(std::wstring_view{s});
}
inline std::string ToUTF8String(const std::wstring& s) {
  return ToUTF8String(std::wstring_view{s});
}
std::wstring ToWideString(std::string_view s);
inline std::wstring ToWideString(const char* s) {
  return ToWideString(std::string_view{s});
}
inline std::wstring ToWideString(const std::string& s) {
  return ToWideString(std::string_view{s});
}
inline std::wstring ToWideString(const std::wstring& s) { return s; }
inline std::wstring ToWideString(std::wstring_view s) { return std::wstring{s}; }
#else
inline std::string ToWideString(const std::string& s) { return s; }
inline std::string ToWideString(const char* s) { return s; }
inline std::string ToWideString(std::string_view s) { return std::string{s}; }
#endif

inline PathString ToPathString(const PathString& s) {
  return s;
}

#ifdef _WIN32

static_assert(std::is_same<PathString, std::wstring>::value, "PathString is not std::wstring!");

inline PathString ToPathString(std::string_view s) {
  return ToWideString(s);
}
inline PathString ToPathString(const char* s) {
  return ToWideString(s);
}
inline PathString ToPathString(const std::string& s) {
  return ToWideString(s);
}

inline PathChar ToLowerPathChar(PathChar c) {
  return std::towlower(c);
}

inline std::string PathToUTF8String(const PathString& s) {
  return ToUTF8String(s);
}

#else

static_assert(std::is_same<PathString, std::string>::value, "PathString is not std::string!");

inline PathString ToPathString(const char* s) {
  return s;
}

inline PathString ToPathString(std::string_view s) {
  return PathString{s};
}

inline PathChar ToLowerPathChar(PathChar c) {
  return std::tolower(c);
}

inline std::string PathToUTF8String(const PathString& s) {
  return s;
}

#endif
