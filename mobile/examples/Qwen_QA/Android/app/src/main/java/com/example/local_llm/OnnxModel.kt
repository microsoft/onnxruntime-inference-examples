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
        const val TEMPERATURE = 0.8f
        const val REPETITION_PENALTY = 1.5f
        // val END_TOKEN_IDS = setOf(151643, 151645, 2687, 11255) // <|endoftext|>, <|im_end|>, /s, /p
    }

    private fun initializeModel(): OrtSession {
        val modelFile = loadModelFile(config.modelPath)
        Log.d("ONNX", "Loading model from: ${config.modelPath}")
        val opts = OrtSession.SessionOptions()
        Log.d("ONNX", "Model loaded")
        return env.createSession(modelFile.absolutePath, opts)
    }

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

    // Normal inference all iterations at once
    fun runInference(
        inputIds: IntArray,
        maxTokens: Int = 1024,
        endTokenId: Int = 151645
    ): IntArray {
        val generated = inputIds.toMutableList()

        for (i in 0 until maxTokens) {
            val seqLen = generated.size.toLong()
            // val inputNameMap = session.inputNames.associateBy { it }

            // Create input_ids tensor
            val inputIdsArray = generated.map { it.toLong() }.toLongArray()
            val inputIdsBuffer = LongBuffer.wrap(inputIdsArray)
            val inputTensor = OnnxTensor.createTensor(env, inputIdsBuffer, longArrayOf(1, seqLen))

            // Create attention_mask tensor
            val attnMaskArray = LongArray(seqLen.toInt()) { 1L }
            val attnMaskBuffer = LongBuffer.wrap(attnMaskArray)
            val attnTensor = OnnxTensor.createTensor(env, attnMaskBuffer, longArrayOf(1, seqLen))

            // Create position_ids tensor
            val posIdsArray = LongArray(seqLen.toInt()) { it.toLong() }
            val posIdsBuffer = LongBuffer.wrap(posIdsArray)
            val posTensor = OnnxTensor.createTensor(env, posIdsBuffer, longArrayOf(1, seqLen))

            val inputs: Map<String, OnnxTensor> = mapOf(
                "input_ids" to inputTensor,
                "attention_mask" to attnTensor,
                "position_ids" to posTensor
            )

            val results = session.run(inputs)
            val output = results[0].value as Array<Array<FloatArray>>
            val logits = output[0].last()  // last token's logits
            val nextTokenId = logits.indices.maxByOrNull { logits[it] } ?: 0
            generated.add(nextTokenId)

            // Close tensors
            inputTensor.close()
            attnTensor.close()
            posTensor.close()
            results.close()

            // Break if end token
            if (nextTokenId == endTokenId) break
        }

        return generated.toIntArray()
    }

    // Run Inference streaming the output
    fun runInferenceStreaming(
        inputIds: IntArray,
        maxTokens: Int = 1024,
        endTokenIds: Set<Int> = setOf(151645),
        shouldStop: () -> Boolean = { false },
        onTokenGenerated: (Int) -> Unit
    ) {
        val generated = inputIds.toMutableList()

        for (i in 0 until maxTokens) {
            if (shouldStop()) break

            val seqLen = generated.size.toLong()

            val inputIdsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(generated.map { it.toLong() }.toLongArray()),
                longArrayOf(1, seqLen)
            )
            val attnTensor = OnnxTensor.createTensor(
                env, LongBuffer.wrap(LongArray(seqLen.toInt()) { 1L }), longArrayOf(1, seqLen)
            )
            val posTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(seqLen.toInt()) { it.toLong() }),
                longArrayOf(1, seqLen)
            )

            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attnTensor,
                "position_ids" to posTensor
            )

            val results = session.run(inputs)
            val output = results[0].value as Array<Array<FloatArray>>
            val logits = output[0].last()
            val nextTokenId = logits.indices.maxByOrNull { logits[it] } ?: 0
            generated.add(nextTokenId)

            inputIdsTensor.close()
            attnTensor.close()
            posTensor.close()
            results.close()

            onTokenGenerated(nextTokenId)
            if (nextTokenId in endTokenIds) break
        }
    }

    // Run Inference with streaming and past key values
    fun runInferenceStreamingWithPastKV(
        inputIds: IntArray,
        maxTokens: Int = MAX_TOKENS,
        endTokenIds: Set<Int> = config.eosTokenIds,
        shouldStop: () -> Boolean = { false },
        onTokenGenerated: (Int) -> Unit
    ) {
        val generated = inputIds.toMutableList()
        val numLayers = config.numLayers
        val numKvHeads = config.numKvHeads
        val headDim = config.headDim
        val batchSize = config.batchSize
        val isQwen3 = config.modelName.contains("qwen3", ignoreCase = true)

        val pastKeyValues = mutableMapOf<String, OnnxTensor>()
        repeat(numLayers) { layer ->
            listOf("key", "value").forEach { kv ->
                val name = "past_key_values.$layer.$kv"
                val shape = longArrayOf(batchSize.toLong(), numKvHeads.toLong(), 0, headDim.toLong())
                val emptyKV = FloatArray(0)
                pastKeyValues[name] = if (config.dtype == "float16") {
                    createFloat16Tensor(env, emptyKV, shape)
                } else {
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(emptyKV), shape)
                }
            }
        }

        var totalPosition: Long = inputIds.size.toLong()

        for (i in 0 until maxTokens) {
            if (shouldStop()) break

            val currentInput = if (i == 0) inputIds else intArrayOf(generated.last())
            val seqLen = currentInput.size.toLong()

            val inputIdsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(currentInput.map { it.toLong() }.toLongArray()),
                longArrayOf(1, seqLen)
            )

            val attentionTensor = if (isQwen3) {
                val attn = LongArray(totalPosition.toInt()) { 1L }
                OnnxTensor.createTensor(env, LongBuffer.wrap(attn), longArrayOf(1, totalPosition))
            } else {
                val attn = LongArray(seqLen.toInt()) { 1L }
                OnnxTensor.createTensor(env, LongBuffer.wrap(attn), longArrayOf(1, seqLen))
            }
            val startPos = totalPosition - seqLen
            val posArray = LongArray(seqLen.toInt()) { j -> startPos + j }
            val positionTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(posArray), longArrayOf(1, seqLen))

            val inputs = mutableMapOf<String, OnnxTensor>(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionTensor,
                "position_ids" to positionTensor
            ).apply { putAll(pastKeyValues) }

            val results = session.run(inputs)
            val logits = (results[0].value as Array<Array<FloatArray>>)[0].last()
            val nextTokenId = logits.indices.maxByOrNull { logits[it] } ?: break
            if (nextTokenId in endTokenIds) break

            onTokenGenerated(nextTokenId)
            generated.add(nextTokenId)
            totalPosition += 1

            results.toList().drop(1).forEachIndexed { index, result ->
                val layer = index / 2
                val kv = if (index % 2 == 0) "key" else "value"
                val name = "past_key_values.$layer.$kv"
                val ortValue = result.value
                if (ortValue is OnnxTensor) {
                    pastKeyValues[name]?.close()
                    pastKeyValues[name] = ortValue
                }
            }

            inputIdsTensor.close()
            attentionTensor.close()
            positionTensor.close()
        }

        pastKeyValues.values.forEach { it.close() }
    }
}
