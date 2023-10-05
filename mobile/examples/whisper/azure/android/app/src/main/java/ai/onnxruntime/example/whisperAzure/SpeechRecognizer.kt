package ai.onnxruntime.example.whisperAzure

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import android.os.SystemClock

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.lang.Exception

@Serializable
data class Error(
    val message: String,
    val type: String
)

@Serializable
data class Response(
    val text: String,
)

@Serializable
data class ErrorResponse(
    val error: Error
)

class SpeechRecognizer(modelBytes: ByteArray) : AutoCloseable {
    private val session: OrtSession
    private val baseInputs: Map<String, OnnxTensor>

    init {
        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())

        session = env.createSession(modelBytes, sessionOptions)

        // NOTE!! Modify the following line and input your own OpenAI AUTH_TOKEN for successful API request calls.
        // By default, it will return an error message (given no correct authToken provided.)
        // It's suggested to store your secret OpenAI API key in a secure place, e.g. https://developer.android.com/reference/androidx/security/crypto/EncryptedSharedPreferences
        // instead of hardcoding one here.
        // See `User Settings` section in your OpenAI account for more detailed info about finding your API Key.
        val authToken = "Set this to your auth token (OpenAI API Key);"
        val authTokenInput = OnnxTensor.createTensor(env, arrayOf(authToken), tensorShape(1.toLong()))
        baseInputs = mapOf(
            "auth_token" to authTokenInput
        )
    }

    data class Result(val text: String, val inferenceTimeInMs: Long, val successful: Boolean)

    fun run(audioTensor: OnnxTensor): Result {
        val inputs = mutableMapOf<String, OnnxTensor>()
        baseInputs.toMap(inputs)
        inputs["transcribe0/file"] = audioTensor
        val startTimeInMs = SystemClock.elapsedRealtime()
        val outputs = session.run(inputs)
        val elapsedTimeInMs = SystemClock.elapsedRealtime() - startTimeInMs
        val recognizedText = outputs.use {
            @Suppress("UNCHECKED_CAST")
            (outputs[0].value as Array<String>)[0]
        }

        // Parse the response from OpenAI Whisper Endpoint
        val json = Json { ignoreUnknownKeys = true; coerceInputValues = true}
        try {
            val responseText = json.decodeFromString<Response>(recognizedText)
            return Result(responseText.text, elapsedTimeInMs, true)
        } catch (e: Exception) {
            println("Error: ${e.message}")
        }

        val errorMsg = json.decodeFromString<ErrorResponse>(recognizedText)
        return Result(errorMsg.error.message, elapsedTimeInMs, false)
    }

    override fun close() {
        baseInputs.values.forEach {
            it.close()
        }
        session.close()
    }
}