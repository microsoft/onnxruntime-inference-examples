package ai.onnxruntime.example.speechrecognition

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
    private val useCannedAudioButton: Button by lazy { findViewById(R.id.use_canned_audio_button) }
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
            resultText.text = result.text.ifEmpty { "<no speech detected>" }
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
            Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

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

    private fun updateAudioButtons(isRecording: Boolean) {
        runOnUiThread {
            useCannedAudioButton.isEnabled = !isRecording
            recordAudioButton.isEnabled = !isRecording
            stopRecordingAudioButton.isEnabled = isRecording
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        useCannedAudioButton.setOnClickListener {
            try {
                val audioTensor = resources.openRawResourceFd(R.raw.audio_mono_16khz_f32le).use {
                    AudioTensorSource.fromRawPcmFile(it)
                }
                val result = audioTensor.use { speechRecognizer.run(audioTensor) }
                setSuccessfulResult(result)
            } catch (e: Exception) {
                setError(e)
            }
        }

        recordAudioButton.setOnClickListener {
            if (!hasRecordAudioPermission()) {
                requestPermissions(
                    arrayOf(Manifest.permission.RECORD_AUDIO),
                    RECORD_AUDIO_PERMISSION_REQUEST_CODE)
                return@setOnClickListener
            }

            // Disable record button first.
            // The stop button will be enabled by the recording task.
            recordAudioButton.isEnabled = false

            workerThreadExecutor.submit {
                try {
                    stopRecordingFlag.set(false)
                    updateAudioButtons(isRecording = true)

                    val audioTensor = AudioTensorSource.fromRecording(stopRecordingFlag)
                    val result = audioTensor.use { speechRecognizer.run(audioTensor) }
                    setSuccessfulResult(result)
                } catch (e: Exception) {
                    setError(e)
                } finally {
                    updateAudioButtons(isRecording = false)
                }
            }
        }

        stopRecordingAudioButton.setOnClickListener {
            // Disable stop button first.
            // The record button will be enabled at the end of the recording task.
            stopRecordingAudioButton.isEnabled = false
            stopRecordingFlag.set(true)
        }

        updateAudioButtons(isRecording = false)
    }

    override fun onDestroy() {
        super.onDestroy()
        workerThreadExecutor.shutdown()
        speechRecognizer.close()
    }

    companion object {
        const val TAG = "ORTSpeechRecognizer"
        const val RECORD_AUDIO_PERMISSION_REQUEST_CODE = 1
    }
}