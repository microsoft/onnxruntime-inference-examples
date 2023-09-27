package ai.onnxruntime.example.whisperLocal

import android.Manifest
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private val usePrerecordedAudioButton: Button by lazy { findViewById(R.id.use_prerecorded_audio_button) }
    private val recordAudioButton: Button by lazy { findViewById(R.id.record_audio_button) }
    private val stopRecordingAudioButton: Button by lazy { findViewById(R.id.stop_recording_audio_button) }

    private val resultText: TextView by lazy { findViewById(R.id.result_text) }
    private val statusText: TextView by lazy { findViewById(R.id.status_text) }

    private val speechRecognizer: SpeechRecognizer by lazy {
        resources.openRawResource(R.raw.whisper_cpu_int8_model).use {
            val modelBytes = it.readBytes()
            SpeechRecognizer(modelBytes)
        }
    }

    private val stopRecordingFlag = AtomicBoolean(false)

    private val workerThreadExecutor = Executors.newSingleThreadExecutor()

    private fun setSuccessfulResult(result: SpeechRecognizer.Result) {
        runOnUiThread {
            statusText.text = "Successful speech recognition (${result.inferenceTimeInMs} ms)"
            resultText.text = result.text.ifEmpty { "<No speech detected.>" }
        }
    }

    private fun setError(exception: Exception) {
        Log.e(TAG, "Error: ${exception.localizedMessage}", exception)
        runOnUiThread {
            statusText.text = "Error"
            resultText.text = exception.localizedMessage
        }
    }

    private fun hasRecordAudioPermission(): Boolean =
        ActivityCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == RECORD_AUDIO_PERMISSION_REQUEST_CODE) {
            if (!hasRecordAudioPermission()) {
                Toast.makeText(
                    this,
                    "Permission to record audio was not granted.",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun resetDefaultAudioButtonState() {
        runOnUiThread {
            usePrerecordedAudioButton.isEnabled = true
            recordAudioButton.isEnabled = true
            stopRecordingAudioButton.isEnabled = false
        }
    }

    private fun disableAudioButtons() {
        runOnUiThread {
            usePrerecordedAudioButton.isEnabled = false
            recordAudioButton.isEnabled = false
            stopRecordingAudioButton.isEnabled = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        usePrerecordedAudioButton.setOnClickListener {
            // Disable audio buttons first.
            // The audio button state will be reset at the end of the use prerecorded audio task.
            disableAudioButtons()

            workerThreadExecutor.submit {
                try {
                    val audioTensor = resources.openRawResource(R.raw.audio_mono_16khz_f32le).use {
                        AudioTensorSource.fromRawPcmBytes(it.readBytes())
                    }
                    val result = audioTensor.use { speechRecognizer.run(audioTensor) }
                    setSuccessfulResult(result)
                } catch (e: Exception) {
                    setError(e)
                } finally {
                    resetDefaultAudioButtonState()
                }
            }
        }

        recordAudioButton.setOnClickListener {
            if (!hasRecordAudioPermission()) {
                requestPermissions(
                    arrayOf(Manifest.permission.RECORD_AUDIO),
                    RECORD_AUDIO_PERMISSION_REQUEST_CODE
                )
                return@setOnClickListener
            }

            // Disable audio buttons first.
            // The stop button will be enabled by the recording task.
            disableAudioButtons()

            workerThreadExecutor.submit {
                try {
                    stopRecordingFlag.set(false)
                    runOnUiThread {
                        stopRecordingAudioButton.isEnabled = true
                    }

                    val audioTensor = AudioTensorSource.fromRecording(stopRecordingFlag)
                    val result = audioTensor.use { speechRecognizer.run(audioTensor) }
                    setSuccessfulResult(result)
                } catch (e: Exception) {
                    setError(e)
                } finally {
                    resetDefaultAudioButtonState()
                }
            }
        }

        stopRecordingAudioButton.setOnClickListener {
            // Disable audio buttons first.
            // The audio button state will be reset at the end of the record audio task.
            disableAudioButtons()

            stopRecordingFlag.set(true)
        }

        resetDefaultAudioButtonState()
    }

    override fun onPause() {
        super.onPause()
        stopRecordingFlag.set(true)
    }

    override fun onDestroy() {
        super.onDestroy()
        workerThreadExecutor.shutdown()
        speechRecognizer.close()
    }

    companion object {
        const val TAG = "ORTSpeechRecognizer"
        private const val RECORD_AUDIO_PERMISSION_REQUEST_CODE = 1
    }
}