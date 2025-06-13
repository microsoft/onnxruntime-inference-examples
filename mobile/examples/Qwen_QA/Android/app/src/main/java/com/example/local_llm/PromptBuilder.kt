package com.example.local_llm

class PromptBuilder(
    private val tokenizer: BpeTokenizer,
    private val config: ModelConfig
) {
    fun buildPromptTokens(userInput: String): IntArray {
        return buildPromptTokens(userInput, PromptIntent.QA())
    }

    fun buildPromptTokens(userInput: String, intent: PromptIntent): IntArray {
        return when (config.promptStyle) {
            PromptStyle.QWEN2_5, PromptStyle.QWEN3-> when (intent) {
                is PromptIntent.QA -> buildQwenQA(userInput, intent.systemPrompt)
            }
        }
    }

    private fun buildQwenQA(userInput: String, systemPromptOverride: String?): IntArray {
        val systemPrompt = systemPromptOverride ?: config.defaultSystemPrompt
        val userPrompt = "Q: $userInput\nA:"

        val systemTokens = tokenizer.tokenize(systemPrompt)
        val userTokens = tokenizer.tokenize(userPrompt)

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
