#include <jni.h>
#include <string>
#include <vector>

#include <android/log.h>

#include "ort_genai_c.h"

namespace {
    void ThrowException(JNIEnv *env, OgaResult *result) {
        __android_log_write(ANDROID_LOG_DEBUG, "native", "ThrowException");
        // copy error so we can release the OgaResult
        jstring jerr_msg = env->NewStringUTF(OgaResultGetError(result));
        OgaDestroyResult(result);

        static const char *className = "ai/onnxruntime/genai/demo/GenAIException";
        jclass exClazz = env->FindClass(className);
        jmethodID exConstructor = env->GetMethodID(exClazz, "<init>", "(Ljava/lang/String;)V");
        jobject javaException = env->NewObject(exClazz, exConstructor, jerr_msg);
        env->Throw(static_cast<jthrowable>(javaException));
    }

    void ThrowIfError(JNIEnv *env, OgaResult *result) {
        if (result != nullptr) {
            ThrowException(env, result);
        }
    }

    // handle conversion/release of jstring to const char*
    struct CString {
        CString(JNIEnv *env, jstring str)
                : env_{env}, str_{str}, cstr{env->GetStringUTFChars(str, /* isCopy */ nullptr)} {
        }

        const char *cstr;

        operator const char *() const { return cstr; }

        ~CString() {
            env_->ReleaseStringUTFChars(str_, cstr);
        }

    private:
        JNIEnv *env_;
        jstring str_;
    };
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_demo_GenAIWrapper_loadModel(JNIEnv *env, jobject thiz, jstring model_path) {
    CString path{env, model_path};
    OgaModel *model = nullptr;
    OgaResult *result = OgaCreateModel(path, &model);

    ThrowIfError(env, result);

    return (jlong)model;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_demo_GenAIWrapper_releaseModel(JNIEnv *env, jobject thiz, jlong native_model) {
    auto* model = reinterpret_cast<OgaModel*>(native_model);
    OgaDestroyModel(model);
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_demo_GenAIWrapper_createTokenizer(JNIEnv *env, jobject thiz, jlong native_model) {
    const auto* model = reinterpret_cast<const OgaModel*>(native_model);
    OgaTokenizer *tokenizer = nullptr;
    OgaResult* result = OgaCreateTokenizer(model, &tokenizer);

    ThrowIfError(env, result);

    return (jlong)tokenizer;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_demo_GenAIWrapper_releaseTokenizer(JNIEnv *env, jobject thiz, jlong native_tokenizer) {
    auto* tokenizer = reinterpret_cast<OgaTokenizer*>(native_tokenizer);
    OgaDestroyTokenizer(tokenizer);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_demo_GenAIWrapper_run(JNIEnv *env, jobject thiz, jlong native_model, jlong native_tokenizer,
                                                jstring jprompt, jboolean use_callback) {
    using SequencesPtr = std::unique_ptr<OgaSequences, std::function<void(OgaSequences*)>>;
    using GeneratorParamsPtr = std::unique_ptr<OgaGeneratorParams, std::function<void(OgaGeneratorParams*)>>;
    using TokenizerStreamPtr = std::unique_ptr<OgaTokenizerStream, std::function<void(OgaTokenizerStream*)>>;
    using GeneratorPtr = std::unique_ptr<OgaGenerator, std::function<void(OgaGenerator*)>>;

    auto* model = reinterpret_cast<OgaModel*>(native_model);
    auto* tokenizer = reinterpret_cast<OgaTokenizer*>(native_tokenizer);

    CString prompt{env, jprompt};

    const auto check_result = [env](OgaResult* result) {
        ThrowIfError(env, result);
    };

    OgaSequences* sequences = nullptr;
    check_result(OgaCreateSequences(&sequences));
    SequencesPtr seq_cleanup{sequences, OgaDestroySequences};

    check_result(OgaTokenizerEncode(tokenizer, prompt, sequences));

    OgaGeneratorParams* generator_params = nullptr;
    check_result(OgaCreateGeneratorParams(model, &generator_params));
    GeneratorParamsPtr gp_cleanup{generator_params, OgaDestroyGeneratorParams};

    check_result(OgaGeneratorParamsSetSearchNumber(generator_params, "max_length", 120));
    check_result(OgaGeneratorParamsSetInputSequences(generator_params, sequences));

    __android_log_print(ANDROID_LOG_DEBUG, "native", "starting token generation");

    const auto decode_tokens = [&](const int32_t* tokens, size_t num_tokens){
        const char* output_text = nullptr;
        check_result(OgaTokenizerDecode(tokenizer, tokens, num_tokens, &output_text));
        jstring text = env->NewStringUTF(output_text);
        OgaDestroyString(output_text);
        return text;
    };

    jstring output_text;

    if (!use_callback) {
        OgaSequences *output_sequences = nullptr;
        check_result(OgaGenerate(model, generator_params, &output_sequences));
        SequencesPtr output_seq_cleanup(output_sequences, OgaDestroySequences);

        size_t num_sequences = OgaSequencesCount(output_sequences);
        __android_log_print(ANDROID_LOG_DEBUG, "native", "%zu sequences generated", num_sequences);

        // We don't handle batched requests, so there will only be one sequence and we can hardcode using `0` as the index.
        const int32_t* tokens = OgaSequencesGetSequenceData(output_sequences, 0);
        size_t num_tokens = OgaSequencesGetSequenceCount(output_sequences, 0);

        output_text = decode_tokens(tokens, num_tokens);
    }
    else {
        OgaTokenizerStream* tokenizer_stream = nullptr;
        check_result(OgaCreateTokenizerStream(tokenizer, &tokenizer_stream));
        TokenizerStreamPtr stream_cleanup(tokenizer_stream, OgaDestroyTokenizerStream);

        OgaGenerator *generator = nullptr;
        check_result(OgaCreateGenerator(model, generator_params, &generator));
        GeneratorPtr gen_cleanup(generator, OgaDestroyGenerator);

        // setup the callback to GenAIWrapper::gotNextToken
        jclass genai_wrapper = env->GetObjectClass(thiz);
        jmethodID callback_id = env->GetMethodID(genai_wrapper, "gotNextToken", "(Ljava/lang/String;)V");
        const auto do_callback = [&](const char* token){
            jstring jtoken = env->NewStringUTF(token);
            env->CallVoidMethod(thiz, callback_id, jtoken);
            env->DeleteLocalRef(jtoken);
        };

        while (!OgaGenerator_IsDone(generator)) {
            check_result(OgaGenerator_ComputeLogits(generator));
            check_result(OgaGenerator_GenerateNextToken(generator));

            const int32_t* seq = OgaGenerator_GetSequenceData(generator, 0);
            size_t seq_len = OgaGenerator_GetSequenceCount(generator, 0);  // last token
            const char* token = nullptr;
            check_result(OgaTokenizerStreamDecode(tokenizer_stream, seq[seq_len - 1], &token));
            do_callback(token);
            // Destroy is (assumably) not required for OgaTokenizerStreamDecode based on this which seems to indicate
            // the tokenizer is re-using memory for each call.
            //  `'out' is valid until the next call to OgaTokenizerStreamDecode
            //   or when the OgaTokenizerStream is destroyed`
            // OgaDestroyString(token); This causes 'Scudo ERROR: misaligned pointer when deallocating address'
        }

        // decode overall
        const int32_t* tokens = OgaGenerator_GetSequenceData(generator, 0);
        size_t num_tokens = OgaGenerator_GetSequenceCount(generator, 0);
        output_text = decode_tokens(tokens, num_tokens);
    }

    return output_text;
}
