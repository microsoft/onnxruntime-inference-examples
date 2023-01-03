package ai.onnxruntime.example.superresolution

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.io.InputStream
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private var ortEnv: OrtEnvironment? = null

    var mImage: ImageView? = null
    var super_resulution_button: Button? = null

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mImage = findViewById(R.id.imageView2);
        super_resulution_button = findViewById(R.id.super_resolution_button)
        ortEnv = OrtEnvironment.getEnvironment()

        super_resulution_button?.setOnClickListener(object : View.OnClickListener {
            override fun onClick(view: View?) {
                Toast.makeText(baseContext,"Super resolution performed!", Toast.LENGTH_SHORT).show()
                setSuperResPerformer()
            }
        })
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv?.close()
    }

    private fun updateUI(result: Result) {
        mImage?.setImageBitmap(result.outputBitmap)
    }

    // Read ort model into a ByteArray
    private fun readModel(): ByteArray  {
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

    // Create a new ORT session and then change the ImageAnalysis.Analyzer
    // This part is done in background to avoid blocking the UI
    private fun setSuperResPerformer()  {
        var result = Result()
        var superResPerformer = SuperResPerformer(createOrtSession(), result)
        superResPerformer.analyze(readInputImage())
        updateUI(result);
    }

    companion object {
        public const val TAG = "ORTSuperResolution"
    }
}
