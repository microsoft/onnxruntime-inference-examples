// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.AdapterView
import android.widget.Spinner
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.lang.Runnable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val labelData: List<String> by lazy { readLabels() }
    private val scope = CoroutineScope(Job() + Dispatchers.Main)

    private var ortEnv: OrtEnvironment? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var currentEp: String = "CPU"
    private var enableFp16: Boolean = false
    private var enableQuantizedModel: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // Request Camera permission
        if (allPermissionsGranted()) {
            ortEnv = OrtEnvironment.getEnvironment()
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        val ep_selector: Spinner = findViewById(R.id.ep_selector)
        var ep_adapter: ArrayAdapter<*> = ArrayAdapter.createFromResource(
                this,
                R.array.execution_providers,
                android.R.layout.simple_spinner_item
        )
        ep_adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        ep_selector.adapter = ep_adapter
        ep_selector.onItemSelectedListener = object: AdapterView.OnItemSelectedListener {
            override fun onItemSelected(adapterView: AdapterView<*>,  view: View, pos: Int, id: Long) {
                val ep: String = adapterView.getItemAtPosition(pos) as String
                selectEP(ep)
            }

            override fun onNothingSelected(adapterView: AdapterView<*>) {
                Toast.makeText(applicationContext, "No EP", Toast.LENGTH_SHORT).show()
            }
        }

        enable_quantizedmodel_toggle.setOnCheckedChangeListener { _, isChecked ->
            enableQuantizedModel = isChecked
            setORTAnalyzer()
        }

        enable_fp16_toggle.setOnCheckedChangeListener { _, isChecked ->
            enableFp16 = isChecked
            setORTAnalyzer()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                    .build()
                    .also {
                        it.setSurfaceProvider(viewFinder.surfaceProvider)
                    }

            imageCapture = ImageCapture.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                    .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageCapture, imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

            setORTAnalyzer()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv?.close()
    }

    override fun onRequestPermissionsResult(
            requestCode: Int,
            permissions: Array<out String>,
            grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                        this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT
                ).show()
                finish()
            }

        }
    }

    private fun updateUI(result: Result) {
        if (result.detectedScore.isEmpty())
            return

        runOnUiThread {
            percentMeter.progress = (result.detectedScore[0] * 100).toInt()
            detected_item_1.text = labelData[result.detectedIndices[0]]
            detected_item_value_1.text = "%.2f%%".format(result.detectedScore[0] * 100)

            if (result.detectedIndices.size > 1) {
                detected_item_2.text = labelData[result.detectedIndices[1]]
                detected_item_value_2.text = "%.2f%%".format(result.detectedScore[1] * 100)
            }

            if (result.detectedIndices.size > 2) {
                detected_item_3.text = labelData[result.detectedIndices[2]]
                detected_item_value_3.text = "%.2f%%".format(result.detectedScore[2] * 100)
            }

            inference_time_value.text = result.processTimeMs.toString() + "ms"
        }
    }

    // Read MobileNet V2 classification labels
    private fun readLabels(): List<String> {
        return resources.openRawResource(R.raw.imagenet_classes).bufferedReader().readLines()
    }

    // Read ort model into a ByteArray, run in background
    private suspend fun readModel(): ByteArray = withContext(Dispatchers.IO) {
        val modelID =
            if (enableQuantizedModel) R.raw.mobilenet_v2_uint8 else R.raw.mobilenet_v2_float
        resources.openRawResource(modelID).readBytes()
    }

    // Create a new ORT session in background
    private suspend fun createOrtSession(): OrtSession? = withContext(Dispatchers.Default) {
        var sess_options = OrtSession.SessionOptions()
        if (currentEp == "OpenCL") {
            sess_options.addOpenCL(enableFp16)
        } else if (currentEp == "NNAPI") {
            sess_options.addNnapi()
        }
        ortEnv?.createSession(readModel(), sess_options)
    }

    private fun selectEP(ep: String) {
        currentEp = ep
        Toast.makeText(applicationContext, "Switching to EP " + ep, Toast.LENGTH_SHORT).show()
        showEpDependentOptions()
        setORTAnalyzer()
    }

    private fun showEpDependentOptions() {
        (findViewById(R.id.cpu_options) as View).visibility = if (currentEp == "CPU") View.VISIBLE else View.GONE
        (findViewById(R.id.opencl_options) as View).visibility = if (currentEp == "OpenCL") View.VISIBLE else View.GONE
    }

    // Create a new ORT session and then change the ImageAnalysis.Analyzer
    // This part is done in background to avoid blocking the UI
    private fun setORTAnalyzer(){
        scope.launch {
            imageAnalysis?.clearAnalyzer()
            imageAnalysis?.setAnalyzer(
                    backgroundExecutor,
                    ORTAnalyzer(createOrtSession(), ::updateUI)
            )
        }
    }

    companion object {
        public const val TAG = "ORTImageClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
