package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
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
    private lateinit var outputImage: ImageView
    private lateinit var objectDetectionButton: Button
    private var imageid = 0;
    private lateinit var classes:List<String>

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
        classes = readClasses();
        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        objectDetectionButton.setOnClickListener {
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
        val mutableBitmap: Bitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.color = Color.WHITE // Text Color

        paint.textSize = 28f // Text Size

        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER) // Text Overlapping Pattern

        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, paint)
        var boxit = result.outputBox.iterator()
        while(boxit.hasNext()) {
            var box_info = boxit.next()
            canvas.drawText("%s:%.2f".format(classes[box_info[5].toInt()],box_info[4]),
                box_info[0]-box_info[2]/2, box_info[1]-box_info[3]/2, paint)
        }

        outputImage.setImageBitmap(mutableBitmap)
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.yolov8n_with_pre_post_processing
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
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