package ai.onnxruntime.example.speechrecognition

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
    val type: String,
    val code: String
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
        // By default, the behavior of the app will fail to build if no correct authToken provided.
        val authToken = "Set this to your auth token and add a closing double quote;
        val authTokenInput = OnnxTensor.createTensor(env, arrayOf(authToken), tensorShape(1.toLong()))
        baseInputs = mapOf(
            "auth_token" to authTokenInput
        )
    }

    data class Result(val text: String, val inferenceTimeInMs: Long)

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
        val json = Json { ignoreUnknownKeys = true }
        try {
            val responseText = json.decodeFromString<Response>(recognizedText)
            return Result(responseText.text, elapsedTimeInMs)
        } catch (e: Exception) {
            println("Error: ${e.message}")
        }


        val errorMsg = json.decodeFromString<ErrorResponse>(recognizedText)
        return Result(errorMsg.error.message, elapsedTimeInMs)
    }

    override fun close() {
        baseInputs.values.forEach {
            it.close()
        }
        session.close()
    }
}