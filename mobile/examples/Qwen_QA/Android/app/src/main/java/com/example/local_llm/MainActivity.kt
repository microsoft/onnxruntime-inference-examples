package com.example.local_llm

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import io.noties.markwon.Markwon
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var tokenizer: BpeTokenizer
    private lateinit var onnxModel: OnnxModel
    private lateinit var markwon: Markwon
    private val inferenceScope = CoroutineScope(Dispatchers.IO)
    private var inferenceJob: Job? = null

    private val END_TOKEN_IDS = setOf(151643, 151645) // <|endoftext|> and <|im_end|>

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val thinkingToggle: CheckBox = findViewById(R.id.thinkingToggle)
        tokenizer = BpeTokenizer(this)
        markwon = Markwon.create(this)

        val inputEditText: EditText = findViewById(R.id.userInput)
        inputEditText.movementMethod = android.text.method.ScrollingMovementMethod.getInstance()
        val sendButton: Button = findViewById(R.id.sendButton)
        val stopButton: Button = findViewById(R.id.stopButton)
        val clearButton: Button = findViewById(R.id.clearButton)
        val outputText: TextView = findViewById(R.id.outputView)
        val scrollView: ScrollView = findViewById(R.id.outputScroll)
        // Extract role token IDs using tokenizer
        val roleTokens = RoleTokenIds(
            systemStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("system"),
                tokenizer.getTokenId("Ċ")
            ),
            userStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("user"),
                tokenizer.getTokenId("Ċ")
            ),
            assistantStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("assistant"),
                tokenizer.getTokenId("Ċ")
            ),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )
        val modelconfigqwen25 = ModelConfig(
            modelName = "Qwen2_5",
            promptStyle = PromptStyle.QWEN2_5,
            eosTokenIds = setOf(151643, 151645),
            numLayers = 24,
            numKvHeads = 2,
            headDim = 64,
            batchSize = 1,
            defaultSystemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            roleTokenIds = roleTokens,
            scalarPosId = false
        )

        val modelconfigqwen3 = ModelConfig(
            modelName = "Qwen3",
            promptStyle = PromptStyle.QWEN3,
            eosTokenIds = setOf(151643, 151645),
            numLayers = 28,
            numKvHeads = 8,
            headDim = 128,
            batchSize = 1,
            defaultSystemPrompt = "You are Qwen. You are a helpful assistant",
            roleTokenIds = roleTokens,
            scalarPosId = true,
            dtype = "float16",
            IsThinkingModeAvailable = true
        )

        // ---------------------------------------------------------------------
        // ✨ SELECT WHICH MODEL TO RUN
        //
        // Two ModelConfig objects are defined above:
        //
        //     val modelconfigqwen25 = …   // Qwen2.5-0.5B
        //     val modelconfigqwen3  = …   // Qwen3-0.6B
        //
        // Point the `config` reference at the model you want.  All downstream
        // logic (tokenizer style, KV-cache shape, Thinking-Mode toggle, etc.)
        // uses this single variable, so no other code changes are required.
        // ---------------------------------------------------------------------
        val config = modelconfigqwen25      // ← switch to modelconfigqwen3 for Qwen 3

        val skipTokenIdsQwen3 = setOf(151667, 151668)  // e.g., <think>, </think>
        if (config.IsThinkingModeAvailable) {
            thinkingToggle.visibility = View.VISIBLE
        }
        title = "Pocket LLM — ${config.modelName}"
        val promptBuilder = PromptBuilder(tokenizer, config)
        val mapper = TokenDisplayMapper(this@MainActivity, config.modelName)
        markwon.setMarkdown(outputText, "⏳ Please wait, the model is still loading.")
        sendButton.isEnabled = false

        inferenceScope.launch {
            onnxModel = OnnxModel(this@MainActivity, config)
            withContext(Dispatchers.Main) {
                markwon.setMarkdown(outputText, "✅ Model is ready.")
                sendButton.isEnabled = true
            }
        }


        sendButton.setOnClickListener {
            if (!::onnxModel.isInitialized) {
                markwon.setMarkdown(outputText, "⏳ Please wait, the model is still loading.")
                return@setOnClickListener
            }

            val systemPrompt = if (thinkingToggle.isChecked)
                config.defaultSystemPrompt
            else
                "${config.defaultSystemPrompt} /no_think"
            val intent = PromptIntent.QA(systemPrompt)
            val inputIds = promptBuilder.buildPromptTokens(inputEditText.text.toString(), intent)

            sendButton.isEnabled = false
            stopButton.isEnabled = true
            markwon.setMarkdown(outputText, "⏳ Thinking...")

            inferenceJob = inferenceScope.launch {
                try {
                    val tokenIds = inputIds
                    val builder = StringBuilder()
                    var tokenCounter = 0

                    withContext(Dispatchers.Main) {
                        markwon.setMarkdown(outputText, "")
                    }

                    onnxModel.runInferenceStreamingWithPastKV(
                        inputIds = tokenIds,
                        endTokenIds = END_TOKEN_IDS,
                        shouldStop = { inferenceJob?.isActive != true },
                        onTokenGenerated = { tokenId ->
                            val tokenStr = if (config.modelName.startsWith("Qwen", ignoreCase = true)) {
                                mapper.map(tokenId)
                            } else {
                                tokenizer.decodeSingleToken(tokenId)
                            }

                            // Only append if NOT one of the first 4 tokens (for Qwen3)
                            if (!(config.modelName.equals("qwen3", ignoreCase = true) && tokenCounter < 4)) {
                                builder.append(tokenStr)
                                runOnUiThread {
                                    outputText.text = builder.toString()
                                    scrollView.post {
                                        scrollView.fullScroll(ScrollView.FOCUS_DOWN)
                                    }
                                }
                            }
                            tokenCounter++
                        }
                    )

                    withContext(Dispatchers.Main) {
                        markwon.setMarkdown(outputText, builder.toString())
                        scrollView.post {
                            scrollView.fullScroll(ScrollView.FOCUS_DOWN)
                        }
                    }

                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        markwon.setMarkdown(outputText, "❌ Error: ${e.message ?: "Unknown error."}")
                    }
                } finally {
                    withContext(Dispatchers.Main) {
                        sendButton.isEnabled = true
                        stopButton.isEnabled = false
                    }
                }
            }
        }

        stopButton.setOnClickListener {
            inferenceJob?.cancel()
            val current = outputText.text.toString()
            markwon.setMarkdown(outputText, "$current\n⛔ Generation stopped.")
            scrollView.post {
                scrollView.fullScroll(ScrollView.FOCUS_DOWN)
            }
            sendButton.isEnabled = true
            stopButton.isEnabled = false
        }

        clearButton.setOnClickListener {
            inputEditText.text.clear()
            markwon.setMarkdown(outputText, "")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()
    }
}
