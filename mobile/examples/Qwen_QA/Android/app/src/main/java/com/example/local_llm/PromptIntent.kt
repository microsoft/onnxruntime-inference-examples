package com.example.local_llm

sealed class PromptIntent {
    // If systemPrompt is null, PromptBuilder will fallback to ModelConfig.defaultSystemPrompt
    data class QA(val systemPrompt: String? = null) : PromptIntent()
}