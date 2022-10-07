// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <string>

#ifdef _WIN32
#include <objbase.h>  // MultiByteToWideChar
#endif

template <size_t N>
int64_t GetShapeSize(const std::array<int64_t, N>& shape) {
  int64_t size = 1;

  for (auto dim : shape) {
    size *= dim;
  }

  return size;
}

#ifdef _WIN32
std::wstring ConvertString(std::string_view str);
#endif

//
// BMP utils:
//

struct BmpInfo {
  explicit BmpInfo(const char* bpm_filepath) noexcept;

  BmpInfo(const BmpInfo& other) = default;
  BmpInfo& operator=(const BmpInfo& other) = default;

  BmpInfo(BmpInfo&& other) = default;
  BmpInfo& operator=(BmpInfo&& other) = default;

  ~BmpInfo() = default;

  enum class LoadStatus {
    Ok,
    NotBMP,
    ReadError,
    UnsupportedBMPType,
    UnsupportedCompression,
  };

  static constexpr const char* LoadStatusString(LoadStatus status) noexcept {
    switch (status) {
      case LoadStatus::Ok:
        return "Success";
      case LoadStatus::NotBMP:
        return "Not a valid BMP file";
      case LoadStatus::ReadError:
        return "Error while reading BMP file";
      case LoadStatus::UnsupportedBMPType:
        return "Unsupported BMP file type; only support BITMAPINFOHEADER";
      case LoadStatus::UnsupportedCompression:
        return "Unsupported BMP compression type; only support uncompressed images";
      default:
        return "UNKNOWN ERROR";
    }
  }

  LoadStatus Load();

  bool IsValid() const;
  size_t Width() const { return width_; }
  size_t Height() const { return height_; }
  uint16_t BytesPerPixel() const { return bpp_ >> 3; }
  size_t NumPixels() const { return width_ * height_; }
  const uint8_t* Data() const;
  size_t Size() const;

 private:
  size_t width_;
  size_t height_;
  uint16_t bpp_;
  std::vector<uint8_t> data_;
  const char* filepath_;
};
