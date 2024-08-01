#include <android/log.h>
#include <jni.h>

#include <array>
#include <sstream>
#include <string>
#include <vector>

#include "ort_genai.h"

namespace {

constexpr auto kLogTag = "genai.demo.native";

void HandleCppException(JNIEnv *env, const std::exception &e) {
  __android_log_print(ANDROID_LOG_ERROR, kLogTag, "HandleCppException - exception: %s", e.what());

  // copy error message from e.what()
  jstring jerr_msg = env->NewStringUTF(e.what());

  static const char *className = "ai/onnxruntime/genai/demo/GenAIException";
  jclass exClazz = env->FindClass(className);
  jmethodID exConstructor = env->GetMethodID(exClazz, "<init>", "(Ljava/lang/String;)V");
  jobject javaException = env->NewObject(exClazz, exConstructor, jerr_msg);
  env->Throw(static_cast<jthrowable>(javaException));
}

// handle conversion/release of jstring to const char*
struct CString {
  CString(JNIEnv *env, jstring str) : env_{env}, str_{str}, cstr{env->GetStringUTFChars(str, /* isCopy */ nullptr)} {}

  const char *cstr;

  operator const char *() const { return cstr; }

  ~CString() { env_->ReleaseStringUTFChars(str_, cstr); }

 private:
  JNIEnv *env_;
  jstring str_;
};
}  // namespace

extern "C" JNIEXPORT jlong JNICALL Java_ai_onnxruntime_genai_demo_GenAIWrapper_loadModel(JNIEnv *env, jobject thiz,
                                                                                         jstring model_path) {
  try {
    CString path{env, model_path};
    std::unique_ptr<OgaModel> model = OgaModel::Create(path);
    return (jlong)model.release();
  } catch (const std::exception &e) {
    HandleCppException(env, e);
    return (jlong) nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL Java_ai_onnxruntime_genai_demo_GenAIWrapper_releaseModel(JNIEnv *env, jobject thiz,
                                                                                           jlong native_model) {
  try {
    std::unique_ptr<OgaModel> model{reinterpret_cast<OgaModel *>(native_model)};
    model.reset();
  } catch (const std::exception &e) {
    HandleCppException(env, e);
  }
}

extern "C" JNIEXPORT jlong JNICALL Java_ai_onnxruntime_genai_demo_GenAIWrapper_createTokenizer(JNIEnv *env,
                                                                                               jobject thiz,
                                                                                               jlong native_model) {
  try {
    const auto *model = reinterpret_cast<const OgaModel *>(native_model);
    std::unique_ptr<OgaTokenizer> tokenizer = OgaTokenizer::Create(*model);
    return (jlong)tokenizer.release();
  } catch (const std::exception &e) {
    HandleCppException(env, e);
    return (jlong) nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL Java_ai_onnxruntime_genai_demo_GenAIWrapper_releaseTokenizer(JNIEnv *env,
                                                                                               jobject thiz,
                                                                                               jlong native_tokenizer) {
  try {
    std::unique_ptr<OgaTokenizer> tokenizer{reinterpret_cast<OgaTokenizer *>(native_tokenizer)};
    tokenizer.reset();
  } catch (const std::exception &e) {
    HandleCppException(env, e);
  }
}

extern "C" JNIEXPORT void JNICALL Java_ai_onnxruntime_genai_demo_GenAIWrapper_run(JNIEnv *env, jobject thiz,
                                                                                  jlong native_model,
                                                                                  jlong native_tokenizer,
                                                                                  jstring jprompt) {
  try {
    auto *model = reinterpret_cast<OgaModel *>(native_model);
    auto *tokenizer = reinterpret_cast<OgaTokenizer *>(native_tokenizer);

    CString prompt{env, jprompt};

    // setup the callback to GenAIWrapper::gotNextToken
    jclass genai_wrapper = env->GetObjectClass(thiz);
    jmethodID callback_id = env->GetMethodID(genai_wrapper, "gotNextToken", "(Ljava/lang/String;)V");
    const auto do_callback = [&](const char *token) {
      jstring jtoken = env->NewStringUTF(token);
      env->CallVoidMethod(thiz, callback_id, jtoken);
      env->DeleteLocalRef(jtoken);
    };

    auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

    auto sequences = OgaSequences::Create();
    constexpr size_t seq_len = 512;
    constexpr int32_t space_token = 220;

    tokenizer->Encode(prompt, *sequences);

    const size_t num_tokens = sequences->SequenceCount(0);
    __android_log_print(ANDROID_LOG_DEBUG, kLogTag, "num prompt tokens: %zu", num_tokens);

    if (num_tokens > seq_len) {
      std::ostringstream s;
      s << "Too many prompt tokens! Maximum allowed is " << seq_len << ", got " << num_tokens << ".";
      throw std::runtime_error(s.str());
    }

    sequences->PadSequence(space_token, seq_len, 0);
    std::array<float, seq_len> attn_mask{};
    for (size_t i = 0; i < std::min(seq_len, num_tokens); ++i) {
      attn_mask[i] = 1.0f;
    }
    for (size_t i = num_tokens; i < seq_len; ++i) {
      attn_mask[i] = 0.0f;
    }

    const std::array<int64_t, 1> shape{seq_len};

    auto attn_mask_tensor = OgaTensor::Create(attn_mask.data(), shape.data(), shape.size(), OgaElementType_float32);

    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 1024);
    params->SetSearchOptionBool("do_sample", true);
    params->SetSearchOption("top_k", 5);
    params->SetSearchOption("top_p", 0.9);
    params->SetSearchOption("temperature", 0.1);
    params->SetInputSequences(*sequences);
    params->SetModelInput("attn_mask", *attn_mask_tensor);

    auto generator = OgaGenerator::Create(*model, *params);

    while (!generator->IsDone()) {
      generator->ComputeLogits();

      generator->GenerateNextToken();

      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
      const char *new_decoded_token = tokenizer_stream->Decode(new_token);

      do_callback(new_decoded_token);
    }
  } catch (const std::exception &e) {
    HandleCppException(env, e);
  }
}
