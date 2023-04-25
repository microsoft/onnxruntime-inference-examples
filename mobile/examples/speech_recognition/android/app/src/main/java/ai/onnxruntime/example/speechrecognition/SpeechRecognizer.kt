package ai.onnxruntime.example.speechrecognition

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage

class SpeechRecognizer() {
    private val session: OrtSession

    init {
        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        session = env.createSession("", sessionOptions)
    }

    fun recognize() : String {
        return "hello"
    }
}