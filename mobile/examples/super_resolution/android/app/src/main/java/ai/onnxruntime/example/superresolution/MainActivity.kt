package ai.onnxruntime.example.superresolution

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.io.InputStream
import java.util.*


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var inputImage: ImageView? = null
    private var outputImage: ImageView? = null
    private var superResolutionButton: Button? = null

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImage = findViewById(R.id.imageView1)
        outputImage = findViewById(R.id.imageView2)
        superResolutionButton = findViewById(R.id.super_resolution_button)
        inputImage?.setImageBitmap(
            BitmapFactory.decodeStream(readInputImage())
        );

        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        superResolutionButton?.setOnClickListener {
            try {
                performSuperResolution(ortSession)
                Toast.makeText(baseContext, "Super resolution performed!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when perform super resolution", e)
                Toast.makeText(baseContext, "Failed to perform super resolution", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    private fun updateUI(result: Result) {
        outputImage?.setImageBitmap(result.outputBitmap)
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.pytorch_superresolution_with_pre_post_processing_op18
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readInputImage(): InputStream {
        return assets.open("test_superresolution.png")
    }

    private fun performSuperResolution(ortSession: OrtSession) {
        var superResPerformer = SuperResPerformer()
        var result = superResPerformer.upscale(readInputImage(), ortEnv, ortSession)
        updateUI(result);
    }

    companion object {
        const val TAG = "ORTSuperResolution"
    }
}
