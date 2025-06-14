package com.example.local_llm

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.InputStream
import java.text.Normalizer

class BpeTokenizer(context: Context) {

    private val vocab: Map<String, Int>
    private val idToToken: Map<Int, String>
    private val merges: List<Pair<String, String>>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val specialTokens: Map<String, Int>
    private val nfcNormalize: Boolean

    companion object {
        private const val TAG = "BpeTokenizer"
    }

    init {
        val tokenizerJson = loadTokenizerJson(context)

        // Load base vocabulary
        vocab = tokenizerJson.getJSONObject("model").getJSONObject("vocab").toIntMap()
        idToToken = vocab.entries.associate { (k, v) -> v to k }

        // Load BPE merge rules
        val mergeList = tokenizerJson.getJSONObject("model").getJSONArray("merges")
        merges = (0 until mergeList.length()).map { i ->
            when (val entry = mergeList.get(i)) {
                is String -> {
                    val parts = entry.split(" ")
                    require(parts.size == 2) { "Invalid merge string: $entry" }
                    parts[0] to parts[1]
                }
                is org.json.JSONArray -> {
                    require(entry.length() == 2) { "Invalid merge array: $entry" }
                    entry.getString(0) to entry.getString(1)
                }
                else -> throw IllegalArgumentException("Unsupported merge entry type: ${entry::class.java}")
            }
        }
        bpeRanks = merges.withIndex().associate { it.value to it.index }

        // Load special tokens like <|im_start|> and <|im_end|>
        val addedTokens = tokenizerJson.optJSONArray("added_tokens")
        specialTokens = if (addedTokens != null) {
            (0 until addedTokens.length()).associate {
                val obj = addedTokens.getJSONObject(it)
                obj.getString("content") to obj.getInt("id")
            }
        } else emptyMap()

        // Check if NFC normalization is enabled
        nfcNormalize = tokenizerJson.optJSONObject("normalizer")
            ?.optString("type") == "NFC"

        // Log tokenizer summary
        Log.d(TAG, "Tokenizer loaded successfully: vocab=${vocab.size}, merges=${merges.size}, specialTokens=${specialTokens.size}, NFC=$nfcNormalize")
    }

    // Converts input text into a list of token IDs using BPE and optional special tokens
    fun tokenize(text: String, addSpecialTokens: Boolean = false): IntArray {
        val tokens = mutableListOf<Int>()

        if (addSpecialTokens) {
            specialTokens["<|im_start|>"]?.let { tokens.add(it) }
        }

        val processed = if (nfcNormalize) Normalizer.normalize(text, Normalizer.Form.NFC) else text

        val bpeTokens = bpe(processed)
        bpeTokens.forEach { bpeToken ->
            vocab[bpeToken]?.let { tokens.add(it) }
        }

        if (addSpecialTokens) {
            specialTokens["<|im_end|>"]?.let { tokens.add(it) }
        }

        Log.d(TAG, "Tokenized: \"$text\" → $tokens")
        return tokens.toIntArray()
    }

    // Decodes a list of token IDs back into a readable string
    fun decode(tokenIds: IntArray): String {
        val builder = StringBuilder()
        for (id in tokenIds) {
            val token = idToToken[id]
            if (token != null) {
                builder.append(token)
            } else {
                val special = specialTokens.entries.find { it.value == id }?.key
                builder.append(special ?: "<unk>")
                if (special == null) Log.w(TAG, "Unknown token ID: $id")
            }
        }

        val raw = builder.toString()
        val cleaned = raw
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("▁", " ")

        return if (nfcNormalize) Normalizer.normalize(cleaned, Normalizer.Form.NFC) else cleaned
    }

    // Decodes a single token ID into a string using cached values
    fun decodeSingleToken(tokenId: Int): String {
        return decodedTokenCache[tokenId] ?: "<unk>"
    }

    // Returns the token ID for a string (special tokens included)
    fun getTokenId(token: String): Int {
        return specialTokens[token]
            ?: vocab[token]
            ?: throw IllegalArgumentException("Token '$token' not found in vocab or special tokens.")
    }

    // Splits input string into space and word chunks before BPE merging
    private fun preTokenize(text: String): List<String> {
        return Regex("""\S+|\s+""").findAll(text).map { it.value }.toList()
    }

    // Applies BPE merge rules to a single token string
    private fun bpe(token: String): List<String> {
        var word = token.toCharArray().map { it.toString() }.toMutableList() // preTokenize
        var pairs = getPairs(word)

        while (true) {
            val best = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (!bpeRanks.containsKey(best)) break

            val (first, second) = best
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i++
                }
            }
            word = newWord
            pairs = getPairs(word)
        }

        return word
    }

    // Returns all adjacent pairs of characters or merged subwords
    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return (0 until word.size - 1).map { word[it] to word[it + 1] }.toSet()
    }

    // Loads the tokenizer.json file from the app's assets folder
    private fun loadTokenizerJson(context: Context): JSONObject {
        val filename = "tokenizer.json"
        Log.d(TAG, "Loading tokenizer from assets/$filename")
        val inputStream: InputStream = context.assets.open(filename)
        val jsonStr = inputStream.bufferedReader().use { it.readText() }
        return JSONObject(jsonStr)
    }

    // Converts a JSON object of string→int mappings into a Kotlin map
    private fun JSONObject.toIntMap(): Map<String, Int> {
        return keys().asSequence().associateWith { this.getInt(it) }
    }

    // Precomputed cache for fast single-token decoding
    private val decodedTokenCache: Map<Int, String> = buildMap {
        idToToken.forEach { (id, token) ->
            val decoded = when {
                token == "Ċ" -> "\n"
                token == "▁" -> " "
                token.startsWith("Ġ") -> " " + token.removePrefix("Ġ")
                token.contains("Ċ") -> token.replace("Ċ", "\n")
                token.contains("▁") -> token.replace("▁", " ")
                else -> token
            }
            put(id, decoded)
        }
        specialTokens.forEach { (key, id) -> putIfAbsent(id, key) }
    }
}
