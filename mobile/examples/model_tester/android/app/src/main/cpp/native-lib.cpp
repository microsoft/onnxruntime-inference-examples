#include <android/log.h>

#include <jni.h>

#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include "model_runner.h"

namespace util {
struct JstringUtfCharDeleter {
  JstringUtfCharDeleter(JNIEnv& env, jstring jstr) : env{env}, jstr{jstr} {}

  void operator()(const char* p) {
    env.ReleaseStringUTFChars(jstr, p);
  }

  JNIEnv& env;
  jstring jstr;
};

auto MakeUniqueJstringUtfCharPtr(JNIEnv& env, jstring jstr) {
  const auto* raw_utf_chars = env.GetStringUTFChars(jstr, nullptr);
  return std::unique_ptr<const char, JstringUtfCharDeleter>{
      raw_utf_chars, JstringUtfCharDeleter{env, jstr}};
}

std::string JstringToStdString(JNIEnv& env, jstring jstr) {
  auto utf_chars = MakeUniqueJstringUtfCharPtr(env, jstr);
  return std::string{utf_chars.get()};
}

std::vector<std::string> JstringArrayToStdStrings(JNIEnv& env, jobjectArray jobjs) {
  std::vector<std::string> strs;
  const auto java_string_class = env.FindClass("java/lang/String");
  const auto length = env.GetArrayLength(jobjs);
  for (jsize i = 0; i < length; ++i) {
    const auto jobj = env.GetObjectArrayElement(jobjs, i);
    if (!env.IsInstanceOf(jobj, java_string_class)) {
      throw std::runtime_error("jobjectArray element is not a string.");
    }
    const auto jstr = static_cast<jstring>(jobj);
    strs.emplace_back(JstringToStdString(env, jstr));
  }
  return strs;
}

struct JbyteArrayElementsDeleter {
  JbyteArrayElementsDeleter(JNIEnv& env, jbyteArray array) : env{env}, array{array} {}

  void operator()(jbyte* p) {
    env.ReleaseByteArrayElements(array, p, 0);
  }

  JNIEnv& env;
  jbyteArray array;
};

auto MakeUniqueJbyteArrayElementsPtr(JNIEnv& env, jbyteArray array) {
  auto* jbytes_raw = env.GetByteArrayElements(array, nullptr);
  return std::unique_ptr<jbyte[], JbyteArrayElementsDeleter>{
      jbytes_raw, JbyteArrayElementsDeleter{env, array}};
}
}  // namespace util

extern "C" JNIEXPORT jstring JNICALL
Java_com_onnxruntime_example_modeltester_MainActivity_run(JNIEnv* env, jobject thiz,
                                                          jbyteArray java_model_bytes,
                                                          jint num_iterations,
                                                          jstring java_execution_provider_type,
                                                          jobjectArray java_execution_provider_option_names,
                                                          jobjectArray java_execution_provider_option_values) {
  try {
    auto model_bytes = util::MakeUniqueJbyteArrayElementsPtr(*env, java_model_bytes);
    const size_t model_bytes_length = env->GetArrayLength(java_model_bytes);
    auto model_bytes_span = std::span{reinterpret_cast<const std::byte*>(model_bytes.get()),
                                      model_bytes_length};

    auto config = model_runner::RunConfig{};
    config.model_path_or_bytes = model_bytes_span;
    config.num_iterations = num_iterations;

    if (java_execution_provider_type != nullptr) {
      config.ep.emplace();
      config.ep->provider_name = util::JstringToStdString(*env, java_execution_provider_type);

      if (java_execution_provider_option_names != nullptr &&
          java_execution_provider_option_values != nullptr) {
        const auto option_names = util::JstringArrayToStdStrings(*env, java_execution_provider_option_names);
        const auto option_values = util::JstringArrayToStdStrings(*env, java_execution_provider_option_values);
        if (option_names.size() != option_values.size()) {
          throw std::runtime_error("Execution provider option names and values must have the same size.");
        }
        for (size_t i = 0; i < option_names.size(); ++i) {
          config.ep->provider_options.emplace(option_names[i], option_values[i]);
        }
      }
    }

    auto result = model_runner::Run(config);

    auto summary = model_runner::GetRunSummary(config, result);

    return env->NewStringUTF(summary.c_str());
  } catch (const std::exception& e) {
    const auto java_exception_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(java_exception_class, e.what());

    __android_log_print(ANDROID_LOG_ERROR, "com.onnxruntime.example.modeltester",
                        "Error: %s", e.what());

    return nullptr;
  }
}