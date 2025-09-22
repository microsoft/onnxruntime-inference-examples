package com.example.local_llm

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer

class OnnxModel(private val context: Context, private val config: ModelConfig) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = initializeModel()

    companion object {
        const val MAX_TOKENS = 1024
        const val MAX_INPUT_TOKENS = 512
        const val TEMPERATURE = 0.8f
        const val REPETITION_PENALTY = 1.5f
        private const val TAG = "OnnxModel"
    }

    // Initialize ONNX session from asset model path
    private fun initializeModel(): OrtSession {
        val modelFile = loadModelFile(config.modelPath)
        Log.d(TAG, "Loading model from: ${modelFile.absolutePath}")
        val opts = OrtSession.SessionOptions()
        val session = env.createSession(modelFile.absolutePath, opts)
        Log.d(TAG, "Model loaded and session initialized")
        return session
    }

    // Copy model file from assets to internal storage (required by ONNX runtime)
    private fun loadModelFile(filename: String): File {
        val assetManager = context.assets
        val inputStream = assetManager.open(filename)
        val file = File(context.filesDir, filename)
        val outputStream = FileOutputStream(file)
        inputStream.copyTo(outputStream)
        inputStream.close()
        outputStream.close()
        return file
    }

    // Temperature scaling for logits
    private fun applyTemperature(logits: FloatArray, temperature: Float): FloatArray {
        if (temperature == 1.0f) return logits
        Log.d(TAG, "Applying temperature: $temperature")
        return FloatArray(logits.size) { i -> logits[i] / temperature }
    }

    // Penalize previously generated tokens to reduce repetition
    private fun applyRepetitionPenalty(logits: FloatArray, generated: List<Int>, penalty: Float): FloatArray {
        if (penalty == 1.0f) return logits
        Log.d(TAG, "Applying repetition penalty: $penalty")
        val adjusted = logits.copyOf()
        for (tokenId in generated) {
            if (tokenId in adjusted.indices) {
                if (adjusted[tokenId] < 0) {
                    adjusted[tokenId] *= penalty
                } else {
                    adjusted[tokenId] /= penalty
                }
            }
        }
        return adjusted
    }

    // Standard (non-streaming) inference that generates full output in one go
    fun runInference(
        inputIds: IntArray,
        maxTokens: Int = MAX_TOKENS,
        endTokenId: Int = 151645
    ): IntArray {
        val generated = inputIds.toMutableList()

        for (i in 0 until maxTokens) {
            val seqLen = generated.size.toLong()
            Log.d(TAG, "Iteration $i | Sequence length: $seqLen")

            val inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(generated.map { it.toLong() }.toLongArray()), longArrayOf(1, seqLen))
            val attnTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(seqLen.toInt()) { 1L }), longArrayOf(1, seqLen))
            val posTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(seqLen.toInt()) { it.toLong() }), longArrayOf(1, seqLen))

            val results = session.run(mapOf(
                "input_ids" to inputTensor,
                "attention_mask" to attnTensor,
                "position_ids" to posTensor
            ))

            val logits = (results[0].value as Array<Array<FloatArray>>)[0].last()
            val nextTokenId = logits.indices.maxByOrNull { logits[it] } ?: 0
            generated.add(nextTokenId)

            Log.d(TAG, "Generated token: $nextTokenId")

            inputTensor.close(); attnTensor.close(); posTensor.close(); results.close()
            if (nextTokenId == endTokenId) break
        }

        return generated.toIntArray()
    }

    // Streaming inference — calls back with each generated token
    fun runInferenceStreaming(
        inputIds: IntArray,
        maxTokens: Int = MAX_TOKENS,
        endTokenIds: Set<Int> = setOf(151645),
        shouldStop: () -> Boolean = { false },
        onTokenGenerated: (Int) -> Unit
    ) {
        val generated = inputIds.toMutableList()

        for (i in 0 until maxTokens) {
            if (shouldStop()) {
                Log.d(TAG, "Generation stopped early at token $i")
                break
            }

            val seqLen = generated.size.toLong()
            val inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(generated.map { it.toLong() }.toLongArray()), longArrayOf(1, seqLen))
            val attnTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(seqLen.toInt()) { 1L }), longArrayOf(1, seqLen))
            val posTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(seqLen.toInt()) { it.toLong() }), longArrayOf(1, seqLen))

            val results = session.run(mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attnTensor,
                "position_ids" to posTensor
            ))

            val rawLogits = (results[0].value as Array<Array<FloatArray>>)[0].last()

            // Apply temperature and/or penalty here if desired
            // val logitsWithTemp = applyTemperature(rawLogits, TEMPERATURE)
            // val logitsWithPenalty = applyRepetitionPenalty(rawLogits, generated, REPETITION_PENALTY)
            // val logitsAdjusted = applyRepetitionPenalty(applyTemperature(rawLogits, TEMPERATURE), generated, REPETITION_PENALTY)

            val nextTokenId = rawLogits.indices.maxByOrNull { rawLogits[it] } ?: 0
            generated.add(nextTokenId)

            Log.d(TAG, "Streaming token: $nextTokenId")
            inputIdsTensor.close(); attnTensor.close(); posTensor.close(); results.close()

            onTokenGenerated(nextTokenId)
            if (nextTokenId in endTokenIds) break
        }
    }

    // Run token-by-token inference using past key-value (KV) caching.
    // This improves performance by avoiding computation over past tokens.
    fun runInferenceStreamingWithPastKV(
        inputIds: IntArray,
        maxTokens: Int = MAX_TOKENS,
        maxInputTokens: Int = MAX_INPUT_TOKENS,
        endTokenIds: Set<Int> = config.eosTokenIds,
        shouldStop: () -> Boolean = { false },
        onTokenGenerated: (Int) -> Unit
    ) {

        val generated: MutableList<Int> = if (inputIds.size > maxInputTokens) {
            Log.w(TAG, "Prompt had ${inputIds.size} tokens; truncated to last $maxInputTokens tokens.")
            inputIds.takeLast(maxInputTokens).toMutableList()
        } else {
            inputIds.toMutableList()
        }

        val isQwen3 = config.modelName.contains("qwen3", ignoreCase = true)

        // Initialize empty past key/value cache for all layers
        val pastKeyValues = mutableMapOf<String, OnnxTensor>()
        repeat(config.numLayers) { layer ->
            listOf("key", "value").forEach { kv ->
                val name = "past_key_values.$layer.$kv"
                val shape = longArrayOf(config.batchSize.toLong(), config.numKvHeads.toLong(), 0, config.headDim.toLong())
                val emptyKV = FloatArray(0)
                pastKeyValues[name] = if (config.dtype == "float16") {
                    createFloat16Tensor(env, emptyKV, shape)
                } else {
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(emptyKV), shape)
                }
            }
        }

        var totalPosition = inputIds.size.toLong()  // Running position counter for position_ids and attention mask

        for (i in 0 until maxTokens) {
            if (shouldStop()) {
                Log.d(TAG, "Stopped externally at token $i")
                break
            }

            // Use full prompt on first step, then only the last generated token
            val currentInput = if (i == 0) inputIds else intArrayOf(generated.last())
            val seqLen = currentInput.size.toLong()

            // Create input_ids tensor
            val inputTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(currentInput.map { it.toLong() }.toLongArray()),
                longArrayOf(1, seqLen)
            )

            // Attention mask: Qwen3 requires full attention mask up to current position
            val attentionTensor = if (isQwen3) {
                val attn = LongArray(totalPosition.toInt()) { 1L }
                OnnxTensor.createTensor(env, LongBuffer.wrap(attn), longArrayOf(1, totalPosition))
            } else {
                val attn = LongArray(seqLen.toInt()) { 1L }
                OnnxTensor.createTensor(env, LongBuffer.wrap(attn), longArrayOf(1, seqLen))
            }

            // Position IDs: increment from where the last token ended
            val startPos = totalPosition - seqLen
            val posArray = LongArray(seqLen.toInt()) { j -> startPos + j }
            val posTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(posArray), longArrayOf(1, seqLen))

            // Merge standard inputs with cached past key-values
            val inputs = mutableMapOf(
                "input_ids" to inputTensor,
                "attention_mask" to attentionTensor,
                "position_ids" to posTensor
            ).apply { putAll(pastKeyValues) }

            // Run the ONNX model
            val results = session.run(inputs)
            val rawLogits = (results[0].value as Array<Array<FloatArray>>)[0].last()

            // Apply temperature or repetition penalty if desired:
            // val logitsWithTemp = applyTemperature(rawLogits, TEMPERATURE)
            // val logitsWithPenalty = applyRepetitionPenalty(rawLogits, generated, REPETITION_PENALTY)
            // val logitsAdjusted = applyRepetitionPenalty(applyTemperature(rawLogits, TEMPERATURE), generated, REPETITION_PENALTY)

            // Select highest-probability token (greedy decoding)
            val nextTokenId = rawLogits.indices.maxByOrNull { rawLogits[it] } ?: break
            // Log.d(TAG, "Step $i - Token $nextTokenId")

            // Stop if generated token is in end-of-sequence set
            if (nextTokenId in endTokenIds) break

            // Return token to UI or callback
            onTokenGenerated(nextTokenId)
            generated.add(nextTokenId)
            totalPosition += 1

            // Update KV cache with present key/values from model output
            results.drop(1).forEachIndexed { index, result ->
                val layer = index / 2
                val kv = if (index % 2 == 0) "key" else "value"
                val name = "past_key_values.$layer.$kv"
                val ortValue = result.value as? OnnxTensor
                ortValue?.let {
                    pastKeyValues[name]?.close()  // Free old tensor
                    pastKeyValues[name] = it
                }
            }

            // Clean up tensors
            inputTensor.close()
            attentionTensor.close()
            posTensor.close()
        }

        // Release all cached key/value tensors after generation
        pastKeyValues.values.forEach { it.close() }
    }
}
