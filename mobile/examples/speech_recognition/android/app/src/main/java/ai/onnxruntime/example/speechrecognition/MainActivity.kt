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
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private val useCannedAudioButton: Button by lazy { findViewById(R.id.use_canned_audio_button) }
    private val recordAudioButton: Button by lazy { findViewById(R.id.record_audio_button) }
    private val resultText: TextView by lazy { findViewById(R.id.result_text) }
    private val statusText: TextView by lazy { findViewById(R.id.status_text) }

    private val speechRecognizer: SpeechRecognizer by lazy {
        resources.openRawResource(R.raw.whisper_tiny_beamsearch_int8).use {
            val modelBytes = it.readBytes()
            SpeechRecognizer(modelBytes)
        }
    }

    private fun setSuccessfulResult(result: SpeechRecognizer.Result) {
        runOnUiThread {
            statusText.text = "Successful inference (${result.inferenceTimeInMs} ms)"
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

    private fun runSpeechRecognitionAndUpdateUi(runFn: () -> SpeechRecognizer.Result) {
        try {
            val result = runFn()
            setSuccessfulResult(result)
        } catch (e: Exception) {
            setError(e)
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
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        useCannedAudioButton.setOnClickListener {
            runSpeechRecognitionAndUpdateUi {
                val audioTensor = resources.openRawResourceFd(R.raw.audio_mono_16khz_f32le).use {
                    AudioTensorSource.fromRawPcmFile(it)
                }

                audioTensor.use { speechRecognizer.run(audioTensor) }
            }
        }

        recordAudioButton.setOnClickListener {
            if (hasRecordAudioPermission()) {
                runSpeechRecognitionAndUpdateUi {
                    val stopRecordingFlag = AtomicBoolean(false)
                    val audioTensor = AudioTensorSource.fromRecording(stopRecordingFlag)

                    audioTensor.use { speechRecognizer.run(audioTensor) }
                }
            } else {
                requestPermissions(
                    arrayOf(Manifest.permission.RECORD_AUDIO),
                    RECORD_AUDIO_PERMISSION_REQUEST_CODE)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.close()
    }

    companion object {
        const val TAG = "ORTSpeechRecognizer"
        const val RECORD_AUDIO_PERMISSION_REQUEST_CODE = 1
    }
}