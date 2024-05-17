// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <stdexcept>

#if __cplusplus >= 202002L
#include <span>
#endif

#include "ort_genai_c.h"

// GenAI C++ API
//
// This is a zero cost wrapper around the C API, and provides for a set of C++ classes with automatic resource management

/* A simple end to end example of how to generate an answer from a prompt:
 *
 * auto model = OgaModel::Create("phi-2");
 * auto tokenizer = OgaTokenizer::Create(*model);
 *
 * auto sequences = OgaSequences::Create();
 * tokenizer->Encode("A great recipe for Kung Pao chicken is ", *sequences);
 *
 * auto params = OgaGeneratorParams::Create(*model);
 * params->SetInputSequences(*sequences);
 * params->SetSearchOption("max_length", 200);
 *
 * auto output_sequences = model->Generate(*params);
 * auto out_string = tokenizer->Decode(output_sequences->Get(0));
 *
 * std::cout << "Output: " << std::endl << out_string << std::endl;
 */

// The types defined in this file are to give us zero overhead C++ style interfaces around an opaque C pointer.
// For example, there is no actual 'OgaModel' type defined anywhere, so we create a fake definition here
// that lets users have a C++ style OgaModel type that can be held in a std::unique_ptr.
//
// This OgaAbstract struct is to prevent accidentally trying to use them by value.
struct OgaAbstract {
  OgaAbstract() = delete;
  OgaAbstract(const OgaAbstract&) = delete;
  void operator=(const OgaAbstract&) = delete;
};

struct OgaResult : OgaAbstract {
  const char* GetError() const { return OgaResultGetError(this); }
  static void operator delete(void* p) { OgaDestroyResult(reinterpret_cast<OgaResult*>(p)); }
};

// This is used to turn OgaResult return values from the C API into std::runtime_error exceptions
inline void OgaCheckResult(OgaResult* result) {
  if (result) {
    std::unique_ptr<OgaResult> p_result{result};  // Take ownership so it's destroyed properly
    throw std::runtime_error(p_result->GetError());
  }
}

struct OgaLog {
  void SetBool(const char* name, bool value) {
    OgaCheckResult(OgaSetLogBool(name, value));
  }

  void SetString(const char* name, const char* value) {
    OgaCheckResult(OgaSetLogString(name, value));
  }
};

struct OgaModel : OgaAbstract {
  static std::unique_ptr<OgaModel> Create(const char* config_path) {
    OgaModel* p;
    OgaCheckResult(OgaCreateModel(config_path, &p));
    return std::unique_ptr<OgaModel>(p);
  }

  std::unique_ptr<OgaSequences> Generate(const OgaGeneratorParams& params) {
    OgaSequences* p;
    OgaCheckResult(OgaGenerate(this, &params, &p));
    return std::unique_ptr<OgaSequences>(p);
  }

  static void operator delete(void* p) { OgaDestroyModel(reinterpret_cast<OgaModel*>(p)); }
};

struct OgaString {
  OgaString(const char* p) : p_{p} {}
  ~OgaString() { OgaDestroyString(p_); }

  operator const char*() const { return p_; }

  const char* p_;
};

struct OgaSequences : OgaAbstract {
  static std::unique_ptr<OgaSequences> Create() {
    OgaSequences* p;
    OgaCheckResult(OgaCreateSequences(&p));
    return std::unique_ptr<OgaSequences>(p);
  }

  size_t Count() const {
    return OgaSequencesCount(this);
  }

  size_t SequenceCount(size_t index) const {
    return OgaSequencesGetSequenceCount(this, index);
  }

  const int32_t* SequenceData(size_t index) const {
    return OgaSequencesGetSequenceData(this, index);
  }

#if __cplusplus >= 202002L
  std::span<const int32_t> Get(size_t index) const {
    return {SequenceData(index), SequenceCount(index)};
  }
#endif

  static void operator delete(void* p) { OgaDestroySequences(reinterpret_cast<OgaSequences*>(p)); }
};

struct OgaTokenizer : OgaAbstract {
  static std::unique_ptr<OgaTokenizer> Create(const OgaModel& model) {
    OgaTokenizer* p;
    OgaCheckResult(OgaCreateTokenizer(&model, &p));
    return std::unique_ptr<OgaTokenizer>(p);
  }

  void Encode(const char* str, OgaSequences& sequences) const {
    OgaCheckResult(OgaTokenizerEncode(this, str, &sequences));
  }

  OgaString Decode(const int32_t* tokens_data, size_t tokens_length) const {
    const char* p;
    OgaCheckResult(OgaTokenizerDecode(this, tokens_data, tokens_length, &p));
    return p;
  }

#if __cplusplus >= 202002L
  OgaString Decode(std::span<const int32_t> tokens) const {
    const char* p;
    OgaCheckResult(OgaTokenizerDecode(this, tokens.data(), tokens.size(), &p));
    return p;
  }
#endif

  static void operator delete(void* p) { OgaDestroyTokenizer(reinterpret_cast<OgaTokenizer*>(p)); }
};

struct OgaTokenizerStream : OgaAbstract {
  static std::unique_ptr<OgaTokenizerStream> Create(const OgaTokenizer& tokenizer) {
    OgaTokenizerStream* p;
    OgaCheckResult(OgaCreateTokenizerStream(&tokenizer, &p));
    return std::unique_ptr<OgaTokenizerStream>(p);
  }

  /*
   * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
   * The caller is responsible for concatenating each chunk together to generate the complete result.
   * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
   */
  const char* Decode(int32_t token) {
    const char* out;
    OgaCheckResult(OgaTokenizerStreamDecode(this, token, &out));
    return out;
  }

  static void operator delete(void* p) { OgaDestroyTokenizerStream(reinterpret_cast<OgaTokenizerStream*>(p)); }
};

struct OgaGeneratorParams : OgaAbstract {
  static std::unique_ptr<OgaGeneratorParams> Create(const OgaModel& model) {
    OgaGeneratorParams* p;
    OgaCheckResult(OgaCreateGeneratorParams(&model, &p));
    return std::unique_ptr<OgaGeneratorParams>(p);
  }

  void SetSearchOption(const char* name, double value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchNumber(this, name, value));
  }

  void SetSearchOptionBool(const char* name, bool value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchBool(this, name, value));
  }

  void SetInputIDs(const int32_t* input_ids, size_t input_ids_count, size_t sequence_length, size_t batch_size) {
    OgaCheckResult(OgaGeneratorParamsSetInputIDs(this, input_ids, input_ids_count, sequence_length, batch_size));
  }

  void SetInputSequences(const OgaSequences& sequences) {
    OgaCheckResult(OgaGeneratorParamsSetInputSequences(this, &sequences));
  }

  void TryGraphCaptureWithMaxBatchSize(int max_batch_size) {
    OgaCheckResult(OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(this, max_batch_size));
  }

  static void operator delete(void* p) { OgaDestroyGeneratorParams(reinterpret_cast<OgaGeneratorParams*>(p)); }
};

struct OgaGenerator : OgaAbstract {
  static std::unique_ptr<OgaGenerator> Create(OgaModel& model, const OgaGeneratorParams& params) {
    OgaGenerator* p;
    OgaCheckResult(OgaCreateGenerator(&model, &params, &p));
    return std::unique_ptr<OgaGenerator>(p);
  }

  bool IsDone() const {
    return OgaGenerator_IsDone(this);
  }

  void ComputeLogits() {
    OgaCheckResult(OgaGenerator_ComputeLogits(this));
  }

  void GenerateNextToken() {
    OgaCheckResult(OgaGenerator_GenerateNextToken(this));
  }

  size_t GetSequenceCount(size_t index) const {
    return OgaGenerator_GetSequenceCount(this, index);
  }

  const int32_t* GetSequenceData(size_t index) const {
    return OgaGenerator_GetSequenceData(this, index);
  }

#if __cplusplus >= 202002L
  std::span<const int32_t> GetSequence(size_t index) const {
    return {GetSequenceData(index), GetSequenceCount(index)};
  }
#endif

  static void operator delete(void* p) { OgaDestroyGenerator(reinterpret_cast<OgaGenerator*>(p)); }
};
