// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef BUILDING_ORT_GENAI_C
#define OGA_EXPORT __declspec(dllexport)
#else
#define OGA_EXPORT __declspec(dllimport)
#endif
#define OGA_API_CALL _stdcall
#else
// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define OGA_EXPORT __attribute__((visibility("default")))
#else
#define OGA_EXPORT
#endif
#define OGA_API_CALL
#endif

// ONNX Runtime Generative AI C API
// This API is not thread safe.

typedef struct OgaResult OgaResult;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaModel OgaModel;
// OgaSequences is an array of token arrays where the number of token arrays can be obtained using
// OgaSequencesCount and the number of tokens in each token array can be obtained using OgaSequencesGetSequenceCount.
typedef struct OgaSequences OgaSequences;
typedef struct OgaTokenizer OgaTokenizer;
typedef struct OgaTokenizerStream OgaTokenizerStream;

/* \brief Call this on process exit to cleanly shutdown the genai library & its onnxruntime usage
 * \return Error message contained in the OgaResult. The const char* is owned by the OgaResult
 *         and can will be freed when the OgaResult is destroyed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaShutdown();

/*
 * \param[in] result OgaResult that contains the error message.
 * \return Error message contained in the OgaResult. The const char* is owned by the OgaResult
 *         and can will be freed when the OgaResult is destroyed.
 */
OGA_EXPORT const char* OGA_API_CALL OgaResultGetError(const OgaResult* result);

/*
 * \param[in] Set logging options, see logging.h 'struct LogItems' for the list of available options
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetLogBool(const char* name, bool value);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetLogString(const char* name, const char* value);

/*
 * \param[in] result OgaResult to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyResult(OgaResult*);
OGA_EXPORT void OGA_API_CALL OgaDestroyString(const char*);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateSequences(OgaSequences** out);

/*
 * \param[in] sequences OgaSequences to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroySequences(OgaSequences* sequences);

/*
 * \brief Returns the number of sequences in the OgaSequences
 * \param[in] sequences
 * \return The number of sequences in the OgaSequences
 */
OGA_EXPORT size_t OGA_API_CALL OgaSequencesCount(const OgaSequences* sequences);

/*
 * \brief Returns the number of tokens in the sequence at the given index
 * \param[in] sequences
 * \return The number of tokens in the sequence at the given index
 */
OGA_EXPORT size_t OGA_API_CALL OgaSequencesGetSequenceCount(const OgaSequences* sequences, size_t sequence_index);

/*
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaSequencesGetSequenceCount
 * \param[in] sequences
 * \return The pointer to the sequence data at the given index. The pointer is valid until the OgaSequences is destroyed.
 */
OGA_EXPORT const int32_t* OGA_API_CALL OgaSequencesGetSequenceData(const OgaSequences* sequences, size_t sequence_index);

/*
 * \brief Creates a model from the given configuration directory and device type.
 * \param[in] config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * \param[in] device_type The device type to use for the model.
 * \param[out] out The created model.
 * \return OgaResult containing the error message if the model creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out);

/*
 * \brief Destroys the given model.
 * \param[in] model The model to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyModel(OgaModel* model);

/*
 * \brief Generates an array of token arrays from the model execution based on the given generator params.
 * \param[in] model The model to use for generation.
 * \param[in] generator_params The parameters to use for generation.
 * \param[out] out The generated sequences of tokens. The caller is responsible for freeing the sequences using OgaDestroySequences
 *             after it is done using the sequences.
 * \return OgaResult containing the error message if the generation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerate(OgaModel* model, const OgaGeneratorParams* generator_params, OgaSequences** out);

/*
 * \brief Creates a OgaGeneratorParams from the given model.
 * \param[in] model The model to use for generation.
 * \param[out] out The created generator params.
 * \return OgaResult containing the error message if the generator params creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out);

/*
 * \brief Destroys the given generator params.
 * \param[in] generator_params The generator params to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* generator_params);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchNumber(OgaGeneratorParams* generator_params, const char* name, double value);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchBool(OgaGeneratorParams* generator_params, const char* name, bool value);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(OgaGeneratorParams* generator_params, int32_t max_batch_size);

/*
 * \brief Sets the input ids for the generator params. The input ids are used to seed the generation.
 * \param[in] generator_params The generator params to set the input ids on.
 * \param[in] input_ids The input ids array of size input_ids_count = batch_size * sequence_length.
 * \param[in] input_ids_count The total number of input ids.
 * \param[in] sequence_length The sequence length of the input ids.
 * \param[in] batch_size The batch size of the input ids.
 * \return OgaResult containing the error message if the setting of the input ids failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputIDs(OgaGeneratorParams* generator_params, const int32_t* input_ids,
                                                                 size_t input_ids_count, size_t sequence_length, size_t batch_size);

/*
 * \brief Sets the input id sequences for the generator params. The input id sequences are used to seed the generation.
 * \param[in] generator_params The generator params to set the input ids on.
 * \param[in] sequences The input id sequences.
 * \return OgaResult containing the error message if the setting of the input id sequences failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputSequences(OgaGeneratorParams* generator_params, const OgaSequences* sequences);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams*, const int32_t* inputs, size_t count);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperDecoderInputIDs(OgaGeneratorParams*, const int32_t* input_ids, size_t input_ids_count);

/*
 * \brief Creates a generator from the given model and generator params.
 * \param[in] model The model to use for generation.
 * \param[in] params The parameters to use for generation.
 * \param[out] out The created generator.
 * \return OgaResult containing the error message if the generator creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGenerator(OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out);

/*
 * \brief Destroys the given generator.
 * \param[in] generator The generator to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyGenerator(OgaGenerator* generator);

/*
 * \brief Returns true if the generator has finished generating all the sequences.
 * \param[in] generator The generator to check if it is done with generating all sequences.
 * \return True if the generator has finished generating all the sequences, false otherwise.
 */
OGA_EXPORT bool OGA_API_CALL OgaGenerator_IsDone(const OgaGenerator* generator);

/*
 * \brief Computes the logits from the model based on the input ids and the past state. The computed logits are stored in the generator.
 * \param[in] generator The generator to compute the logits for.
 * \return OgaResult containing the error message if the computation of the logits failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_ComputeLogits(OgaGenerator* generator);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken(OgaGenerator* generator);

/*
 * \brief Returns the number of tokens in the sequence at the given index.
 * \param[in] generator The generator to get the count of the tokens for the sequence at the given index.
 * \return The number tokens in the sequence at the given index.
 */
OGA_EXPORT size_t OGA_API_CALL OgaGenerator_GetSequenceCount(const OgaGenerator* generator, size_t index);

/*
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaGenerator_GetSequenceCount
 * \param[in] generator The generator to get the sequence data for the sequence at the given index.
 * \return The pointer to the sequence data at the given index. The sequence data is owned by the OgaGenerator
 *         and will be freed when the OgaGenerator is destroyed. The caller must copy the data if it needs to
 *         be used after the OgaGenerator is destroyed.
 */
OGA_EXPORT const int32_t* OGA_API_CALL OgaGenerator_GetSequenceData(const OgaGenerator* generator, size_t index);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer*);

/* Encodes a single string and adds the encoded sequence of tokens to the OgaSequences. The OgaSequences must be freed with OgaDestroySequences
   when it is no longer needed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer*, const char* str, OgaSequences* sequences);

/* Decode a single token sequence and returns a null terminated utf8 string. out_string must be freed with OgaDestroyString
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer*, const int32_t* tokens, size_t token_count, const char** out_string);

/* OgaTokenizerStream is to decoded token strings incrementally, one token at a time.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer*, OgaTokenizerStream** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream*);

/*
 * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
 * The caller is responsible for concatenating each chunk together to generate the complete result.
 * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream*, int32_t token, const char** out);

OGA_EXPORT OgaResult* OGA_API_CALL OgaSetCurrentGpuDeviceId(int device_id);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGetCurrentGpuDeviceId(int* device_id);

#ifdef __cplusplus
}
#endif
