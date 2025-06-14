package com.example.local_llm

// Constructs token sequences (prompts) for different tasks based on model configuration
class PromptBuilder(
    private val tokenizer: BpeTokenizer,   // Tokenizer used to convert strings into token IDs
    private val config: ModelConfig        // Model-specific configuration (prompt style, role tokens, etc.)
) {
    // Public method for simple QA prompts using default system prompt
    fun buildPromptTokens(userInput: String): IntArray {
        return buildPromptTokens(userInput, PromptIntent.QA())
    }

    // Builds prompt tokens based on intent (e.g., QA) and user input
    fun buildPromptTokens(userInput: String, intent: PromptIntent): IntArray {
        return when (config.promptStyle) {
            // Currently supports Qwen2.5 and Qwen3 formatting
            PromptStyle.QWEN2_5, PromptStyle.QWEN3 -> when (intent) {
                is PromptIntent.QA -> buildQwenQA(userInput, intent.systemPrompt)
            }
        }
    }

    // Builds Qwen-style prompt for QA interaction
    private fun buildQwenQA(userInput: String, systemPromptOverride: String?): IntArray {
        val systemPrompt = systemPromptOverride ?: config.defaultSystemPrompt
        val userPrompt = "Q: $userInput\nA:"

        // Tokenize system and user inputs
        val systemTokens = tokenizer.tokenize(systemPrompt)
        val userTokens = tokenizer.tokenize(userPrompt)

        // Build the final prompt as:
        return buildList {
            addAll(config.roleTokenIds.systemStart)
            addAll(systemTokens.toList())
            add(config.roleTokenIds.endToken)

            addAll(config.roleTokenIds.userStart)
            addAll(userTokens.toList())
            add(config.roleTokenIds.endToken)

            addAll(config.roleTokenIds.assistantStart)
        }.toIntArray()
    }
}
