package com.example.local_llm

// Enum for selecting how prompts are formatted based on the model's expected structure
enum class PromptStyle {
    QWEN2_5,  // Qwen2.5-style prompt formatting
    QWEN3     // Qwen3-style prompt formatting with scalar position IDs and optional "thinking mode"
}

// Token IDs used to mark system/user/assistant roles and boundaries in prompts
data class RoleTokenIds(
    val systemStart: List<Int>,     // Tokens prepended before system prompt
    val userStart: List<Int>,       // Tokens prepended before user message
    val assistantStart: List<Int>,  // Tokens prepended before model/assistant response
    val endToken: Int               // Token appended at the end of each role block
)

// Main configuration class for defining model behavior, structure, and prompting format
data class ModelConfig(
    val modelName: String,                       // Display name of the model (e.g., "Qwen2_5")
    val modelPath: String = "model.onnx",        // File path to the ONNX model inside assets
    val promptStyle: PromptStyle,                // How to construct the input prompt
    val eosTokenIds: Set<Int>,                   // Set of token IDs that signal end-of-sequence
    val numLayers: Int,                          // Number of transformer layers
    val numKvHeads: Int,                         // Number of key/value heads (can be < attention heads)
    val headDim: Int,                            // Dimensionality of each attention head
    val batchSize: Int,                          // Number of inputs processed together (usually 1)
    val defaultSystemPrompt: String,             // Fallback system prompt if none is provided
    val roleTokenIds: RoleTokenIds,              // Tokens for prompt structure and role separation
    val scalarPosId: Boolean = false,            // Enables scalar-style position IDs (used by Qwen3)
    val dtype: String = "float32",               // Data type for tensors: "float32" or "float16"
    val IsThinkingModeAvailable: Boolean = false // Enables toggle for "thinking mode" (Qwen3-specific)
)
