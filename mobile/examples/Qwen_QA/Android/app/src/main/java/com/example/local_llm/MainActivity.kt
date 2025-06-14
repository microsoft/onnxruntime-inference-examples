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

        // === Initialize UI ===
        val thinkingToggle: CheckBox = findViewById(R.id.thinkingToggle)
        val inputEditText: EditText = findViewById(R.id.userInput)
        val sendButton: Button = findViewById(R.id.sendButton)
        val stopButton: Button = findViewById(R.id.stopButton)
        val clearButton: Button = findViewById(R.id.clearButton)
        val outputText: TextView = findViewById(R.id.outputView)
        val scrollView: ScrollView = findViewById(R.id.outputScroll)

        tokenizer = BpeTokenizer(this)
        markwon = Markwon.create(this)
        inputEditText.movementMethod = android.text.method.ScrollingMovementMethod.getInstance()

        // === Define token IDs for prompt formatting ===
        val roleTokens = RoleTokenIds(
            systemStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("system"), tokenizer.getTokenId("Ċ")),
            userStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("user"), tokenizer.getTokenId("Ċ")),
            assistantStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("assistant"), tokenizer.getTokenId("Ċ")),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )

        // === Model configurations ===
        val modelconfigqwen25 = ModelConfig(
            modelName = "Qwen2_5",
            promptStyle = PromptStyle.QWEN2_5,
            modelPath = "model.onnx",
            eosTokenIds = END_TOKEN_IDS,
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
            modelPath = "model.onnx",
            eosTokenIds = END_TOKEN_IDS,
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
        // SELECT WHICH MODEL TO RUN
        // ---------------------------------------------------------------------
        val config = modelconfigqwen25  // ← Switch to modelconfigqwen3 to run Qwen 3

        // === Conditional thinking mode toggle ===
        if (config.IsThinkingModeAvailable) {
            thinkingToggle.visibility = View.VISIBLE
        }

        title = "Pocket LLM — ${config.modelName}"

        val promptBuilder = PromptBuilder(tokenizer, config)
        val mapper = TokenDisplayMapper(this, config.modelName)

        // Inform the user that the model is loading
        markwon.setMarkdown(outputText, "⏳ Please wait, the model is still loading.")
        sendButton.isEnabled = false

        // === Async model load ===
        inferenceScope.launch {
            Log.d("MainActivity", "Loading ONNX model...")
            onnxModel = OnnxModel(this@MainActivity, config)
            withContext(Dispatchers.Main) {
                markwon.setMarkdown(outputText, "✅ Model is ready.")
                sendButton.isEnabled = true
            }
        }

        // === Send Button Listener ===
        sendButton.setOnClickListener {
            if (!::onnxModel.isInitialized) {
                markwon.setMarkdown(outputText, "⏳ Please wait, the model is still loading.")
                return@setOnClickListener
            }

            // Use different system prompt based on Thinking Mode toggle
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
                    val builder = StringBuilder()
                    var tokenCounter = 0

                    withContext(Dispatchers.Main) {
                        markwon.setMarkdown(outputText, "")
                    }

                    // Stream tokens one-by-one using past KV cache
                    onnxModel.runInferenceStreamingWithPastKV(
                        inputIds = inputIds,
                        endTokenIds = END_TOKEN_IDS,
                        shouldStop = { inferenceJob?.isActive != true },
                        onTokenGenerated = { tokenId ->
                            val tokenStr = if (config.modelName.startsWith("Qwen", ignoreCase = true)) {
                                mapper.map(tokenId)
                            } else {
                                tokenizer.decodeSingleToken(tokenId)
                            }

                            // Skip first few tokens (typically structural in Qwen3)
                            if (!(config.modelName.equals("qwen3", ignoreCase = true) && tokenCounter < 4)) {
                                builder.append(tokenStr)
                                runOnUiThread {
                                    outputText.text = builder.toString()
                                    scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
                                }
                            }
                            tokenCounter++
                        }
                    )

                    // Finalize output after generation ends
                    withContext(Dispatchers.Main) {
                        markwon.setMarkdown(outputText, builder.toString())
                        scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
                    }

                } catch (e: Exception) {
                    Log.e("MainActivity", "Generation error", e)
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

        // === Stop Button Listener ===
        stopButton.setOnClickListener {
            inferenceJob?.cancel()
            val current = outputText.text.toString()
            markwon.setMarkdown(outputText, "$current\n⛔ Generation stopped.")
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
            sendButton.isEnabled = true
            stopButton.isEnabled = false
        }

        // === Clear Button Listener ===
        clearButton.setOnClickListener {
            inputEditText.text.clear()
            markwon.setMarkdown(outputText, "")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()  // Cancel background inference when activity is destroyed
    }
}
