
#include <shcore.h>
#include <wincodec.h>
#include <wincodecsdk.h>

#include <filesystem>
#include <vector>
#include <wil/com.h>

#include "image_file.h"


/**
 *  Read the file from `input_file` and auto-scale it to 720x720
 * \param out should be freed by caller after use
 * \param output_count Array length of the `out` param
 */
int read_image_file(_In_z_ const ORTCHAR_T* input_file, _Out_ size_t* height, _Out_ size_t* width, _Outptr_ float** out,
                    _Out_ size_t* output_count) {
  wil::com_ptr_failfast<IWICImagingFactory> piFactory =
      wil::CoCreateInstanceFailFast<IWICImagingFactory>(CLSID_WICImagingFactory);
  wil::com_ptr_failfast<IWICBitmapDecoder> decoder;
  FAIL_FAST_IF_FAILED(
      piFactory->CreateDecoderFromFilename(input_file, NULL, GENERIC_READ,
                                           WICDecodeMetadataCacheOnDemand,  // defer parsing non-critical metadata
                                           &decoder));
  UINT count = 0;
  FAIL_FAST_IF_FAILED(decoder->GetFrameCount(&count));
  if (count != 1) {
    printf("The image has multiple frames, I don't know which to choose.\n");
    return -1;
  }
  wil::com_ptr_failfast<IWICBitmapFrameDecode> piFrameDecode;
  FAIL_FAST_IF_FAILED(decoder->GetFrame(0, &piFrameDecode));
  UINT image_width, image_height;
  FAIL_FAST_IF_FAILED(piFrameDecode->GetSize(&image_width, &image_height));
  wil::com_ptr_failfast<IWICBitmapScaler> scaler;
  IWICBitmapSource* source_to_copy = piFrameDecode.get();
  if (image_width != 720 || image_height != 720) {
    FAIL_FAST_IF_FAILED(piFactory->CreateBitmapScaler(&scaler));
    FAIL_FAST_IF_FAILED(scaler->Initialize(source_to_copy, 720, 720, WICBitmapInterpolationModeFant));
    source_to_copy = scaler.get();
    image_width = 720;
    image_height = 720;
  }
  wil::com_ptr_failfast<IWICFormatConverter> ppIFormatConverter;
  FAIL_FAST_IF_FAILED(piFactory->CreateFormatConverter(&ppIFormatConverter));
  FAIL_FAST_IF_FAILED(ppIFormatConverter->Initialize(source_to_copy, GUID_WICPixelFormat24bppRGB,
                                                     WICBitmapDitherTypeNone, NULL, 0.f, WICBitmapPaletteTypeCustom));
  // output format is 24bpp, which means 24 bits per pixel
  constexpr UINT bytes_per_pixel = 24 / 8;
  UINT stride = image_width * bytes_per_pixel;
  std::vector<uint8_t> data(image_width * image_height * bytes_per_pixel);
  FAIL_FAST_IF_FAILED(ppIFormatConverter->CopyPixels(nullptr, stride, static_cast<UINT>(data.size()), data.data()));

  hwc_to_chw(data.data(), image_height, image_width, out, output_count);
  *height = image_height;
  *width = image_width;
  return 0;
}

int write_image_file(_In_ uint8_t* model_output_bytes, unsigned int height, unsigned int width,
                     _In_z_ const ORTCHAR_T* output_file) {
  std::filesystem::path file_path(output_file);
  if (!file_path.has_extension()) {
    printf("Unrecognized file type!\n");
    return -1;
  }
  auto ext = file_path.extension().wstring();
  GUID container_format = GUID_ContainerFormatJpeg;
  if (_wcsicmp(ext.c_str(), L".jpg") == 0) {
    container_format = GUID_ContainerFormatJpeg;
  } else if (_wcsicmp(ext.c_str(), L".png") == 0) {
    container_format = GUID_ContainerFormatPng;
  } else if (_wcsicmp(ext.c_str(), L".bmp") == 0) {
    container_format = GUID_ContainerFormatBmp;
  } else {
    wprintf(L"Unrecognized file type:%s!\n", ext.c_str());
    return -1;
  }

  wil::com_ptr_failfast<IWICImagingFactory> piFactory =
      wil::CoCreateInstanceFailFast<IWICImagingFactory>(CLSID_WICImagingFactory);
  wil::com_ptr_failfast<IWICStream> output_stream;
  FAIL_FAST_IF_FAILED(piFactory->CreateStream(&output_stream));
  output_stream->InitializeFromFilename(output_file, GENERIC_WRITE);

  wil::com_ptr_failfast<IWICBitmapEncoder> encoder;
  piFactory->CreateEncoder(container_format, nullptr, &encoder);
  FAIL_FAST_IF_FAILED(encoder->Initialize(output_stream.get(), WICBitmapEncoderNoCache));
  wil::com_ptr_failfast<IWICBitmapFrameEncode> frame;
  wil::com_ptr_failfast<IPropertyBag2> bag = nullptr;
  FAIL_FAST_IF_FAILED(encoder->CreateNewFrame(&frame, &bag));
  FAIL_FAST_IF_FAILED(frame->Initialize(bag.get()));
  frame->SetSize(width, height);
  WICPixelFormatGUID targetFormat = GUID_WICPixelFormat24bppRGB;
  FAIL_FAST_IF_FAILED(frame->SetPixelFormat(&targetFormat));
  constexpr UINT bytes_per_pixel = 24 / 8;
  size_t stride = width * bytes_per_pixel;
  // The last parameter of WritePixels is a "BYTE*", not "const BYTE*".
  FAIL_FAST_IF_FAILED(frame->WritePixels(height, static_cast<UINT>(stride),
                                         static_cast<UINT>(height * width * bytes_per_pixel), model_output_bytes));
  FAIL_FAST_IF_FAILED(frame->Commit());
  FAIL_FAST_IF_FAILED(encoder->Commit());
  return 0;
}