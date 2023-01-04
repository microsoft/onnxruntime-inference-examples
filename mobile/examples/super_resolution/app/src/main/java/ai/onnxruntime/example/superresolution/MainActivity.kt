package ai.onnxruntime.example.superresolution

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.io.InputStream
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private var ortEnv: OrtEnvironment? = null

    private var outputImage: ImageView? = null
    private var superResolutionButton: Button? = null

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        outputImage = findViewById(R.id.imageView2);
        superResolutionButton = findViewById(R.id.super_resolution_button)

        ortEnv = OrtEnvironment.getEnvironment()

        superResolutionButton?.setOnClickListener {
            Toast.makeText(baseContext, "Super resolution performed!", Toast.LENGTH_SHORT).show()
            setSuperResPerformer()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv?.close()
    }

    private fun updateUI(result: Result) {
        outputImage?.setImageBitmap(result.outputBitmap)
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.pt_super_resolution_op16
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readInputImage(): InputStream {
        return assets.open("test_superresolution.png")
    }

    private fun createOrtSession(): OrtSession? {
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        return ortEnv?.createSession(readModel(), sessionOptions)
    }

    private fun setSuperResPerformer() {
        var result = Result()
        var superResPerformer = SuperResPerformer(createOrtSession(), result)
        superResPerformer.analyze(readInputImage())
        updateUI(result);
    }

    companion object {
        const val TAG = "ORTSuperResolution"
    }
}
