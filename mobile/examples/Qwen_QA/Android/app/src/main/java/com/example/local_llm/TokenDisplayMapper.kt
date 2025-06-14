package com.example.local_llm

import android.content.Context
import org.json.JSONObject

// Maps token IDs to human-readable strings using a Qwen-specific display mapping
class TokenDisplayMapper(context: Context, modelName: String) {

    // Load display map only for Qwen-family models
    private val tokenToDisplay: Map<Int, String> = if (modelName.startsWith("Qwen", ignoreCase = true)) {
        try {
            // Read mapping from qwen_token_display_mapping.json in assets
            val inputStream = context.assets.open("qwen_token_display_mapping.json")
            val jsonStr = inputStream.bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonStr)
            val mapJson = jsonObject.getJSONObject("token_to_display")

            // Parse: {"151645": "<|im_end|>", ...} → Map<Int, String>
            mapJson.keys().asSequence().associate { key ->
                key.toInt() to mapJson.getString(key)
            }
        } catch (e: Exception) {
            // Fallback to empty map if file missing or malformed
            emptyMap()
        }
    } else {
        emptyMap()
    }

    // Returns readable token representation if available, otherwise raw ID wrapped in <>
    fun map(tokenId: Int): String {
        return tokenToDisplay[tokenId] ?: "<$tokenId>"
    }
}
