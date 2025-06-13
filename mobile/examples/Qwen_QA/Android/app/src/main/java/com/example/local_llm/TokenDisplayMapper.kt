package com.example.local_llm

import android.content.Context
import org.json.JSONObject

class TokenDisplayMapper(context: Context, modelName: String) {

    private val tokenToDisplay: Map<Int, String>

    init {
        tokenToDisplay = if (modelName.startsWith("Qwen", ignoreCase = true)) {
            try {
                val inputStream = context.assets.open("qwen_token_display_mapping.json")
                val jsonStr = inputStream.bufferedReader().use { it.readText() }
                val jsonObject = JSONObject(jsonStr)
                val mapJson = jsonObject.getJSONObject("token_to_display")

                mapJson.keys().asSequence().associate { key ->
                    key.toInt() to mapJson.getString(key)
                }
            } catch (e: Exception) {
                emptyMap()
            }
        } else {
            emptyMap()
        }
    }

    fun map(tokenId: Int): String {
        return tokenToDisplay[tokenId] ?: "<$tokenId>"
    }
}
