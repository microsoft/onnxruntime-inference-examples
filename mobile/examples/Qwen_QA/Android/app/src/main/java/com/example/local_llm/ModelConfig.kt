package com.example.local_llm

enum class PromptStyle {
    QWEN2_5,
    QWEN3
}

data class RoleTokenIds(
    val systemStart: List<Int>,
    val userStart: List<Int>,
    val assistantStart: List<Int>,
    val endToken: Int
)

data class ModelConfig(
    val modelName: String,
    val modelPath: String = "Qwen2_5_0_5B.onnx",
    val promptStyle: PromptStyle,
    val eosTokenIds: Set<Int>,
    val numLayers: Int,
    val numKvHeads: Int,
    val headDim: Int,
    val batchSize: Int,
    val defaultSystemPrompt: String,
    val roleTokenIds: RoleTokenIds,
    val scalarPosId: Boolean = false,
    val dtype: String = "float32",
    val IsThinkingModeAvailable: Boolean = false
)
