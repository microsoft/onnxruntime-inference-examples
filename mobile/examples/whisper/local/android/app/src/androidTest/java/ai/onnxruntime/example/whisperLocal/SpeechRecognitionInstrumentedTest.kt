package ai.onnxruntime.example.whisperLocal

import ai.onnxruntime.example.whisperLocal.AudioTensorSource
import ai.onnxruntime.example.whisperLocal.SpeechRecognizer
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class SpeechRecognitionInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("ai.onnxruntime.example.whisperLocal", appContext.packageName)
    }

    @Test
    fun runModelWithPrerecordedAudio() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        val modelBytes: ByteArray =
            appContext.resources.openRawResource(R.raw.whisper_cpu_int8_model).use {
                it.readBytes()
            }

        SpeechRecognizer(modelBytes).use { speechRecognizer ->
            val audioTensor =
                appContext.resources.openRawResource(R.raw.audio_mono_16khz_f32le).use {
                    AudioTensorSource.fromRawPcmBytes(it.readBytes())
                }

            val result = audioTensor.use { speechRecognizer.run(audioTensor) }
            assertTrue(
                result.text.contains(
                    "welcome to the speech recognition example application",
                    ignoreCase = true
                )
            )
        }
    }
}