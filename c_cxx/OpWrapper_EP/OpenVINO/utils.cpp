// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <vector>
#include <fstream>
#include <cassert>
#include <cmath>

#ifdef _WIN32
std::wstring ConvertString(std::string_view str) {
  int str_len = static_cast<int>(str.size());
  int size = MultiByteToWideChar(CP_UTF8, 0, str.data(), str_len, NULL, 0);  // Query size.

  std::wstring wide_str(size, 0);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), str_len, &wide_str[0], size);

  return wide_str;
}
#endif

static std::string SlurpFile(const char* path) {
  constexpr size_t READ_AMOUNT = 4096;

  std::ifstream stream(path, std::ios::in | std::ios::binary);
  std::string output;
  std::string buf(READ_AMOUNT, '\0');

  while (stream.read(&buf[0], READ_AMOUNT)) {
    output.append(buf, 0, stream.gcount());
  }

  output.append(buf, 0, stream.gcount());

  return output;
}

//
// Bmp utils:
//

// Disable struct member padding for the following BMP structs.
// This enables direct memcpys to struct instances from file contents.
#pragma pack(push, 1)
struct BmpHdr {
  char magic[2];
  uint32_t size;
  uint32_t _reserved;
  uint32_t data_offset;
};

struct BmpInfoHdr {
  uint32_t hdr_size;
  uint32_t width;
  int32_t height;
  uint16_t num_color_planes;
  uint16_t num_bpp;
  uint32_t compression;
  uint32_t img_size;
  uint32_t hor_res;
  uint32_t ver_res;
  uint32_t num_cols_in_palette;
  uint32_t num_imp_colors;
};
#pragma pack(pop)

static inline bool IsIndexValid(size_t index, size_t size, bool allow_end) {
  return (index < size || (allow_end && index == size));
}

static inline size_t RowColToIndex(size_t row, size_t col, size_t stride) { return row * stride + col; }

template <typename T>
static size_t FillFromBytes(T* dst, std::string_view src, size_t offset) {
  const size_t next_offset = offset + sizeof(T);

  // Check for wrap-around and that copy start & end are within bounds.
  if ((next_offset > offset) && IsIndexValid(next_offset, src.size(), true)) {
    std::memcpy(dst, &src[offset], sizeof(T));
    return next_offset;
  }

  return 0;
}

BmpInfo::BmpInfo(const char* bmp_filepath) noexcept : width_(0), height_(0), bpp_(0), filepath_(bmp_filepath) {
}

BmpInfo::LoadStatus BmpInfo::Load() {
  if (IsValid()) {
    return BmpInfo::LoadStatus::Ok;
  }

  std::string contents = SlurpFile(filepath_);

  size_t offset = 0;  // Track offset into contents.data() from which data is read.

  // Get bitmap header.
  BmpHdr hdr;
  offset = FillFromBytes(&hdr, contents, offset);
  if (offset == 0) {
    return BmpInfo::LoadStatus::ReadError;
  }

  // Check magic bytes: 'BM'
  if (hdr.magic[0] != 'B' || hdr.magic[1] != 'M') {
    return BmpInfo::LoadStatus::NotBMP;
  }
  
  // Get DIB header size without advancing offset into contents_.
  uint32_t dib_hdr_size;
  size_t tmp_offset = FillFromBytes(&dib_hdr_size, contents, offset);
  if (tmp_offset == 0) {
    return BmpInfo::LoadStatus::ReadError;
  }

  constexpr int EXPECTED_DIB_HDR_SIZE = 40;

  // Only support BMP files with 40 byte dib headers (BITMAPINFOHEADER).
  // See: https://en.wikipedia.org/wiki/BMP_file_format
  if (dib_hdr_size != EXPECTED_DIB_HDR_SIZE) {
    return BmpInfo::LoadStatus::UnsupportedBMPType;
  }

  // Get DIB header (AKA information header).
  BmpInfoHdr dib_hdr;
  offset = FillFromBytes(&dib_hdr, contents, offset);
  if (offset == 0) {
    return BmpInfo::LoadStatus::ReadError;
  }

  // Only support uncompressed images
  if (dib_hdr.compression != 0) {
    return BmpInfo::LoadStatus::UnsupportedCompression;
  }

  width_ = dib_hdr.width;
  height_ = abs(dib_hdr.height);
  bpp_ = dib_hdr.num_bpp;

  const bool is_top_down = dib_hdr.height < 0;
  const size_t bytes_per_pixel = bpp_ >> 3;
  const size_t padded_row_size = ((bpp_ * width_ + 31) / 32) * 4;  // Rows are padded to a multiple of 4 bytes
  const size_t unpadded_row_size = width_ * bytes_per_pixel;
  const size_t row_padding = padded_row_size - unpadded_row_size;

  const size_t data_size = height_ * unpadded_row_size;
  data_.resize(data_size);

  // Copy pixels to unpadded linear array. Flip vertically if necessary.
  size_t src_index = hdr.data_offset;

  for (size_t r = 0; r < static_cast<size_t>(height_); ++r) {  // Iterate row by row
    const size_t dst_row_index = is_top_down ? r : (height_ - r - 1);
    const size_t dst_index = unpadded_row_size * dst_row_index;

    assert(dst_index + unpadded_row_size <= data_.size());
    assert(src_index + unpadded_row_size <= contents.size());
    std::memcpy(&data_[dst_index], &contents[src_index], unpadded_row_size);

    src_index += unpadded_row_size + row_padding;
  }

  return BmpInfo::LoadStatus::Ok;
}

bool BmpInfo::IsValid() const { return !data_.empty(); }

const uint8_t* BmpInfo::Data() const { return data_.data(); }

size_t BmpInfo::Size() const { return width_ * height_ * (bpp_ >> 3); }
