// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <assert.h>
#include <stdio.h>

#include "onnxruntime_c_api.h"
#ifdef _WIN32
#ifdef USE_DML
#include "providers.h"
#endif
#include <objbase.h>
#endif
#include "image_file.h"

#ifdef _WIN32
#define tcscmp wcscmp
#else
#define tcscmp strcmp
#endif

const OrtApi* g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t stride = h * w;
  *output_count = stride * 3;
  float* output_data = (float*)malloc(*output_count * sizeof(float));
  assert(output_data != NULL);
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output_data[c * stride + i] = input[i * 3 + c];
    }
  }
  *output = output_data;
}

/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 */
static void chw_to_hwc(const float* input, size_t h, size_t w, uint8_t** output) {
  size_t stride = h * w;
  uint8_t* output_data = (uint8_t*)malloc(stride * 3);
  assert(output_data != NULL);
  for (size_t c = 0; c != 3; ++c) {
    size_t t = c * stride;
    for (size_t i = 0; i != stride; ++i) {
      float f = input[t + i];
      if (f < 0.f || f > 255.0f) f = 0;
      output_data[i * 3 + c] = (uint8_t)f;
    }
  }
  *output = output_data;
}

static void usage() { printf("usage: <model_path> <input_file> <output_file> [cpu|cuda|dml] \n"); }

int run_inference(OrtSession* session, const ORTCHAR_T* input_file, const ORTCHAR_T* output_file) {
  size_t input_height;
  size_t input_width;
  float* model_input;
  size_t model_input_ele_count;

  if (read_image_file(input_file, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
    return -1;
  }
  if (input_height != 720 || input_width != 720) {
    printf("please resize to image to 720x720\n");
    free(model_input);
    return -1;
  }
  OrtMemoryInfo* memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_shape[] = {1, 3, 720, 720};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count * sizeof(float);

  OrtValue* input_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
                                                           input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &input_tensor));
  assert(input_tensor != NULL);
  int is_tensor;
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);
  g_ort->ReleaseMemoryInfo(memory_info);
  const char* input_names[] = {"inputImage"};
  const char* output_names[] = {"outputImage"};
  OrtValue* output_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1,
                                &output_tensor));
  assert(output_tensor != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
  assert(is_tensor);
  int ret = 0;
  float* output_tensor_data = NULL;
  ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
  uint8_t* output_image_data = NULL;
  chw_to_hwc(output_tensor_data, 720, 720, &output_image_data);
  if (write_image_file(output_image_data, 720, 720, output_file) != 0) {
    ret = -1;
  }
  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);
  free(model_input);
  return ret;
}

void verify_input_output_count(OrtSession* session) {
  size_t count;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  assert(count == 1);
}

int enable_cuda(OrtSessionOptions* session_options) {
  // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
  OrtCUDAProviderOptions o;
  // Here we use memset to initialize every field of the above data struct to zero.
  memset(&o, 0, sizeof(o));
  // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
  // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
  o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  o.gpu_mem_limit = SIZE_MAX;
  OrtStatus* onnx_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
  if (onnx_status != NULL) {
    const char* msg = g_ort->GetErrorMessage(onnx_status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(onnx_status);
    return -1;
  }
  return 0;
}

#ifdef USE_DML
void enable_dml(OrtSessionOptions* session_options) {
  ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
}
#endif

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  if (argc < 4) {
    usage();
    return -1;
  }

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
    return -1;
  }
#ifdef _WIN32
  // CoInitializeEx is only needed if Windows Image Component will be used in this program for image loading/saving.
  HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  if (!SUCCEEDED(hr)) return -1;
#endif
  ORTCHAR_T* model_path = argv[1];
  ORTCHAR_T* input_file = argv[2];
  ORTCHAR_T* output_file = argv[3];
  // By default it will try CUDA first. If CUDA is not available, it will run all the things on CPU.
  // But you can also explicitly set it to DML(directml) or CPU(which means cpu-only).
  ORTCHAR_T* execution_provider = (argc >= 5) ? argv[4] : NULL;
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  assert(env != NULL);
  int ret = 0;
  OrtSessionOptions* session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

  if (execution_provider) {
    if (tcscmp(execution_provider, ORT_TSTR("cpu")) == 0) {
      // Nothing; this is the default
    } else if (tcscmp(execution_provider, ORT_TSTR("dml")) == 0) {
#ifdef USE_DML
      enable_dml(session_options);
#else
      puts("DirectML is not enabled in this build.");
      return -1;
#endif
    } else if (tcscmp(execution_provider, ORT_TSTR("cuda")) == 0) {
      printf("Try to enable CUDA first\n");
      ret = enable_cuda(session_options);
      if (ret) {
        fprintf(stderr, "CUDA is not available\n");
        return -1;
      } else {
        printf("CUDA is enabled\n");
      }
    }
  }

  OrtSession* session;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
  verify_input_output_count(session);
  ret = run_inference(session, input_file, output_file);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);
  if (ret != 0) {
    fprintf(stderr, "fail\n");
  }
#ifdef _WIN32
  CoUninitialize();
#endif
  return ret;
}
