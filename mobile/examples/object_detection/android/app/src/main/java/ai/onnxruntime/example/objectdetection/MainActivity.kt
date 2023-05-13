package ai.onnxruntime.example.objectdetection

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
    private lateinit var inputImage: ImageView
    private var outputImage: ImageView? = null
    private var objectDetectionButton: Button? = null
    private var imageid = 0;

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImage = findViewById(R.id.imageView1)
        outputImage = findViewById(R.id.imageView2)
        objectDetectionButton = findViewById(R.id.object_detection_button)
        inputImage.setImageBitmap(
            BitmapFactory.decodeStream(readInputImage())
        );
        imageid = 0

        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        objectDetectionButton?.setOnClickListener {
            try {
                performObjectDetection(ortSession)
                Toast.makeText(baseContext, "ObjectDetection performed!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when perform ObjectDetection", e)
                Toast.makeText(baseContext, "Failed to perform ObjectDetection", Toast.LENGTH_SHORT)
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
        val modelID = R.raw.yolov8n_with_pre_post_processing
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readInputImage(): InputStream {
        imageid = imageid.xor(1)
        return assets.open("test_object_detection_${imageid}.jpg")
    }

    private fun performObjectDetection(ortSession: OrtSession) {
        var objDetector = ObjectDetector()
        var imagestream = readInputImage()
        inputImage.setImageBitmap(
            BitmapFactory.decodeStream(imagestream)
        );
        imagestream.reset()
        var result = objDetector.detect(imagestream, ortEnv, ortSession)
        updateUI(result);
    }

    companion object {
        const val TAG = "ORTObjectDetection"
    }
}
