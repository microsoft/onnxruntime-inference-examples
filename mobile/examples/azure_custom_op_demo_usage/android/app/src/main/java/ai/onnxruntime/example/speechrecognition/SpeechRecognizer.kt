package ai.onnxruntime.example.speechrecognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import android.os.SystemClock

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

// TODO: Configure for an error response
@Serializable
data class Response(
    val text: String
)

class SpeechRecognizer(modelBytes: ByteArray) : AutoCloseable {
    private val session: OrtSession
    private val baseInputs: Map<String, OnnxTensor>

    init {
        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())

        session = env.createSession(modelBytes, sessionOptions)

        // TODO: ADD USER INPUT AUTH TOKEN
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
        val responseText = Json.decodeFromString<Response>(recognizedText)

        return Result(responseText.text, elapsedTimeInMs)
    }

    override fun close() {
        baseInputs.values.forEach {
            it.close()
        }
        session.close()
    }
}