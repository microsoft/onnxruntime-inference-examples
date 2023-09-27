package ai.onnxruntime.example.whisperLocal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import android.os.SystemClock

class SpeechRecognizer(modelBytes: ByteArray) : AutoCloseable {
    private val session: OrtSession
    private val baseInputs: Map<String, OnnxTensor>

    init {
        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())

        session = env.createSession(modelBytes, sessionOptions)

        val nMels: Long = 80
        val nFrames: Long = 3000

        baseInputs = mapOf(
            "min_length" to createIntTensor(env, intArrayOf(1), tensorShape(1)),
            "max_length" to createIntTensor(env, intArrayOf(200), tensorShape(1)),
            "num_beams" to createIntTensor(env, intArrayOf(1), tensorShape(1)),
            "num_return_sequences" to createIntTensor(env, intArrayOf(1), tensorShape(1)),
            "length_penalty" to createFloatTensor(env, floatArrayOf(1.0f), tensorShape(1)),
            "repetition_penalty" to createFloatTensor(env, floatArrayOf(1.0f), tensorShape(1)),
        )
    }

    data class Result(val text: String, val inferenceTimeInMs: Long)

    fun run(audioTensor: OnnxTensor): Result {
        val inputs = mutableMapOf<String, OnnxTensor>()
        baseInputs.toMap(inputs)
        inputs["audio_pcm"] = audioTensor
        val startTimeInMs = SystemClock.elapsedRealtime()
        val outputs = session.run(inputs)
        val elapsedTimeInMs = SystemClock.elapsedRealtime() - startTimeInMs
        val recognizedText = outputs.use {
            @Suppress("UNCHECKED_CAST")
            (outputs[0].value as Array<Array<String>>)[0][0]
        }
        return Result(recognizedText, elapsedTimeInMs)
    }

    override fun close() {
        baseInputs.values.forEach {
            it.close()
        }
        session.close()
    }
}