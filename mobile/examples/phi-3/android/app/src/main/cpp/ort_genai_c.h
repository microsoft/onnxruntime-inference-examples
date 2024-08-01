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

typedef enum OgaElementType {
  OgaElementType_undefined,
  OgaElementType_float32,  // maps to c type float
  OgaElementType_uint8,    // maps to c type uint8_t
  OgaElementType_int8,     // maps to c type int8_t
  OgaElementType_uint16,   // maps to c type uint16_t
  OgaElementType_int16,    // maps to c type int16_t
  OgaElementType_int32,    // maps to c type int32_t
  OgaElementType_int64,    // maps to c type int64_t
  OgaElementType_string,   // string type (not currently supported by Oga)
  OgaElementType_bool,     // maps to c type bool
  OgaElementType_float16,  // IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
  OgaElementType_float64,  // maps to c type double
  OgaElementType_uint32,   // maps to c type uint32_t
  OgaElementType_uint64,   // maps to c type uint64_t
} OgaElementType;

typedef struct OgaResult OgaResult;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaModel OgaModel;
// OgaSequences is an array of token arrays where the number of token arrays can be obtained using
// OgaSequencesCount and the number of tokens in each token array can be obtained using OgaSequencesGetSequenceCount.
typedef struct OgaSequences OgaSequences;
typedef struct OgaTokenizer OgaTokenizer;
typedef struct OgaTokenizerStream OgaTokenizerStream;
typedef struct OgaTensor OgaTensor;
typedef struct OgaImages OgaImages;
typedef struct OgaNamedTensors OgaNamedTensors;
typedef struct OgaMultiModalProcessor OgaMultiModalProcessor;

/* \brief Call this on process exit to cleanly shutdown the genai library & its onnxruntime usage
 */
OGA_EXPORT void OGA_API_CALL OgaShutdown();

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
OGA_EXPORT void OGA_API_CALL OgaDestroyNamedTensors(OgaNamedTensors*);

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

OGA_EXPORT OgaResult* OGA_API_CALL OgaSequencesPadSequence(OgaSequences* sequences, int32_t pad_token_id, size_t sequence_length, size_t sequence_index);

OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadImage(const char* image_path, OgaImages** images);

OGA_EXPORT void OGA_API_CALL OgaDestroyImages(OgaImages* images);

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
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerate(const OgaModel* model, const OgaGeneratorParams* generator_params, OgaSequences** out);

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

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputs(OgaGeneratorParams* generator_params, const OgaNamedTensors* named_tensors);

/*
 * \brief For additional model inputs that genai does not handle, this lets the user set their values. For example LoRA models handle
 * fine tuning through model inputs. This lets the user supply the fine tuning inputs, while genai handles the standard inputs.
 * \param[in] generator_params The generator params to set the input on
 * \param[in] name Name of the model input (this must match the model's input name)
 * \param[in] tensor The OgaTensor of the input data
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetModelInput(OgaGeneratorParams* generator_params, const char* name, OgaTensor* tensor);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams*, OgaTensor* tensor);

/*
 * \brief Creates a generator from the given model and generator params.
 * \param[in] model The model to use for generation.
 * \param[in] params The parameters to use for generation.
 * \param[out] out The created generator.
 * \return OgaResult containing the error message if the generator creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out);

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

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateMultiModalProcessor(const OgaModel* model, OgaMultiModalProcessor** out);

OGA_EXPORT void OGA_API_CALL OgaDestroyMultiModalProcessor(OgaMultiModalProcessor* processor);

/* Encodes a single string and adds the encoded sequence of tokens to the OgaSequences. The OgaSequences must be freed with OgaDestroySequences
   when it is no longer needed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer*, const char* str, OgaSequences* sequences);

OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessImages(const OgaMultiModalProcessor*, const char* prompt, const OgaImages* images, OgaNamedTensors** input_tensors);

/* Decode a single token sequence and returns a null terminated utf8 string. out_string must be freed with OgaDestroyString
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer*, const int32_t* tokens, size_t token_count, const char** out_string);
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorDecode(const OgaMultiModalProcessor*, const int32_t* tokens, size_t token_count, const char** out_string);

/* OgaTokenizerStream is to decoded token strings incrementally, one token at a time.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer*, OgaTokenizerStream** out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStreamFromProcessor(const OgaMultiModalProcessor*, OgaTokenizerStream** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream*);

/*
 * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
 * The caller is responsible for concatenating each chunk together to generate the complete result.
 * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream*, int32_t token, const char** out);

/* Create an OgaTensor from a user owned buffer. The OgaTensor does not own the memory (as it has no way to free it) so
 * the 'data' parameter must be valid for the lifetime of the OgaTensor.
 *
 * \param[in] data User supplied memory pointer, must remain valid for lifetime of the OgaTensor
 * \param[in] shape_dims Pointer to array of int64_t values that define the tensor shape, example [1 20 30] would be equivalent to a C array of [1][20][30]
 * \param[in] shape_dims_count Count of elements in the shape_dims array
 * \param[in] element_type The data type that 'data' points to.
 * \param[out] out Writes the newly created OgaTensor into this, must be destroyed with OgaDestroyTensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTensorFromBuffer(void* data, const int64_t* shape_dims, size_t shape_dims_count, OgaElementType element_type, OgaTensor** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTensor(OgaTensor* tensor);

/* Get the OgaElementType of the data stored in the OgaTensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetType(OgaTensor*, OgaElementType* out);

/* Get the number of dimensions of the OgaTensor's shape, typically used to allocate a buffer of this size then calling OgaTensorGetShape with it
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetShapeRank(OgaTensor*, size_t* out);

/* Copies the shape dimensions into the shape_dims parameters. shape_dims_count must match the value returned by OgaTensorGetShapeRank
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetShape(OgaTensor*, int64_t* shape_dims, size_t shape_dims_count);

/* A pointer to the tensor data, it is typically cast into the actual data type of the tensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetData(OgaTensor*, void** out);

OGA_EXPORT OgaResult* OGA_API_CALL OgaSetCurrentGpuDeviceId(int device_id);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGetCurrentGpuDeviceId(int* device_id);

#ifdef __cplusplus
}
#endif
