#pragma once
#include "onnxruntime_c_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * \param out should be freed by caller after use
 * \param output_count Array length of the `out` param
 */
int read_image_file(_In_z_ const ORTCHAR_T* input_file, _Out_ size_t* height, _Out_ size_t* width, _Outptr_ float** out,
                  _Out_ size_t* output_count);


int write_image_file(_In_ uint8_t* model_output_bytes, unsigned int height,
                     unsigned int width, _In_z_ const ORTCHAR_T* output_file);

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count);
#ifdef __cplusplus
}
#endif