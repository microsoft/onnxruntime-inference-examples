package com.onnxruntime.example.modeltester

import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import com.onnxruntime.example.modeltester.databinding.ActivityMainBinding
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var currentModel: Any = "" // Can be String (path) or ByteArray (default model)

    // ActivityResultLauncher for picking a model file
    private val pickFileLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { handleSelectedModelFile(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        loadDefaultModel()
        setupUI()
        runInitialModel()
    }

    private fun loadDefaultModel() {
        try {
            currentModel = resources.openRawResource(R.raw.yolo11n).readBytes()
        } catch (e: IOException) {
            // Handle error loading default model, e.g., show a toast or log
            displayError("Failed to load default model: ${e.message}")
        }
    }

    private fun setupUI() {
        binding.browseModelButton.setOnClickListener { pickFileLauncher.launch("*/*") }
        binding.runButton.setOnClickListener { runModelFromUI() }

        // Setup Spinners with ArrayAdapter if not already done via XML entries
        // ArrayAdapter.createFromResource(
        //     this,
        //     R.array.ep_array,
        //     android.R.layout.simple_spinner_item
        // ).also { adapter ->
        //     adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        //     binding.epSpinner.adapter = adapter
        // }

        // ArrayAdapter.createFromResource(
        //     this,
        //     R.array.log_level_array,
        //     android.R.layout.simple_spinner_item
        // ).also { adapter ->
        //     adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        //     binding.logLevelSpinner.adapter = adapter
        // }
    }

    private fun handleSelectedModelFile(uri: Uri) {
        binding.modelPathEditText.setText(uri.toString())
        // Attempt to get a real path or copy to a cache file if it's a content URI
        val filePath = getPathFromUri(uri)
        if (filePath != null) {
            currentModel = filePath
        } else {
            // Fallback or error handling if path resolution fails
            binding.modelPathEditText.error = "Could not resolve file path"
            // Optionally, revert to default model or prevent run
        }
    }

    // Helper to attempt to get a real file path from a URI
    // This can be complex. For content URIs, copying to a cache file is often most reliable.
    private fun getPathFromUri(uri: Uri): String? {
        if ("content".equals(uri.scheme, ignoreCase = true)) {
            return try {
                val inputStream = contentResolver.openInputStream(uri) ?: return null
                val tempFile = File(cacheDir, "temp_model_file")
                FileOutputStream(tempFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
                inputStream.close()
                tempFile.absolutePath
            } catch (e: IOException) {
                e.printStackTrace()
                null
            }
        }
        return uri.path // For file URIs or if direct path access is possible
    }


    private fun runModelFromUI() {
        val numIterations = binding.iterationsEditText.text.toString().toIntOrNull() ?: 10
        val runWarmup = binding.warmupSwitchMaterial.isChecked
        
        val selectedEpString = binding.epSpinner.selectedItem.toString()
        val epName = if (selectedEpString.equals("CPU", ignoreCase = true)) {
            null
        } else {
            selectedEpString
        }
        
        val logLevel = mapSpinnerPositionToLogLevel(binding.logLevelSpinner.selectedItemPosition)

        val modelPathString = binding.modelPathEditText.text.toString()
        if (modelPathString.isNotEmpty() && modelPathString != currentModel.toString()) {
            // User typed a path directly or it wasn't a content URI initially handled by pickFileLauncher
            if (modelPathString.startsWith("content://")) {
                 val newUri = Uri.parse(modelPathString)
                 val resolvedPath = getPathFromUri(newUri)
                 if (resolvedPath != null) {
                     currentModel = resolvedPath
                 } else {
                     binding.modelPathEditText.error = "Invalid model path/URI"
                     return
                 }
            } else {
                 currentModel = modelPathString // Assume direct path
            }
        } else if (modelPathString.isEmpty()) {
            loadDefaultModel() // Revert to default if path is cleared
        }
        // If currentModel is already a ByteArray (default model), it's used directly.
        // If it's a String (path), it's used directly.

        executeNativeRun(currentModel, numIterations, runWarmup, epName, logLevel)
    }

    private fun runInitialModel() {
        // Use default values for the initial run
        val defaultNumIterations = 10
        val defaultRunWarmup = true
        // For initial run, always use CPU, which means passing null for epName
        val defaultEpName: String? = null 
        val defaultLogLevel = -1 // ORT default

        // Ensure default model (ByteArray) is used for initial run
        if (currentModel !is ByteArray) {
            loadDefaultModel()
        }
        executeNativeRun(currentModel, defaultNumIterations, defaultRunWarmup, defaultEpName, defaultLogLevel)
    }

    private fun mapSpinnerPositionToLogLevel(position: Int): Int {
        return when (position) {
            0 -> -1 // Default (ORT default)
            1 -> 0  // Verbose
            2 -> 1  // Info
            3 -> 2  // Warning
            4 -> 3  // Error
            5 -> 4  // Fatal
            else -> -1 // Default to ORT default
        }
    }

    private fun executeNativeRun(model: Any, numIterations: Int, runWarmup: Boolean, epName: String?, logLevel: Int) {
        try {
            val summary = run(
                model, // This is currentModel (String path or ByteArray)
                numIterations,
                runWarmup,
                epName,
                null, // executionProviderOptionNames - not used in this example
                null, // executionProviderOptionValues - not used in this example
                logLevel
            )
            parseAndDisplaySummary(summary)
        } catch (e: Exception) {
            displayError("Native run failed: ${e.message}", e.stackTraceToString())
        }
    }

    private fun parseAndDisplaySummary(summary: String) {
        val na = getString(R.string.na)
        val loadTimeRegex = "Load time: (\\S+)".toRegex()
        val numRunsRegex = "N \\(number of runs\\): (\\d+)".toRegex()
        val avgLatencyRegex = "avg: (\\S+)".toRegex()
        val p50LatencyRegex = "p50: (\\S+)".toRegex()
        val p90LatencyRegex = "p90: (\\S+)".toRegex()
        val p99LatencyRegex = "p99: (\\S+)".toRegex()
        val minLatencyRegex = "min: (\\S+)".toRegex()
        val maxLatencyRegex = "max: (\\S+)".toRegex()

        binding.loadTimeTextView.text = getString(R.string.load_time_label, loadTimeRegex.find(summary)?.groupValues?.get(1) ?: na)
        binding.numRunsTextView.text = getString(R.string.num_runs_label, numRunsRegex.find(summary)?.groupValues?.get(1) ?: na)
        binding.latencyTitleTextView.text = getString(R.string.latency_title_label)
        binding.avgLatencyTextView.text = getString(R.string.avg_latency_label, avgLatencyRegex.find(summary)?.groupValues?.get(1) ?: na)
        binding.p50LatencyTextView.text = getString(R.string.p50_latency_label, p50LatencyRegex.find(summary)?.groupValues?.get(1) ?: na)
        binding.p90LatencyTextView.text = getString(R.string.p90_latency_label, p90LatencyRegex.find(summary)?.groupValues?.get(1) ?: na)
        binding.p99LatencyTextView.text = getString(R.string.p99_latency_label, p99LatencyRegex.find(summary)?.groupValues?.get(1) ?: na)
        binding.minLatencyTextView.text = getString(R.string.min_latency_label, minLatencyRegex.find(summary)?.groupValues?.get(1) ?: na)
        binding.maxLatencyTextView.text = getString(R.string.max_latency_label, maxLatencyRegex.find(summary)?.groupValues?.get(1) ?: na)

        binding.rawSummaryText.text = summary
        binding.rawSummaryText.visibility = View.GONE // Hide by default, show on error or if explicitly toggled
    }

    private fun displayError(errorMessage: String, stackTrace: String? = null) {
        binding.loadTimeTextView.text = getString(R.string.error_prefix, errorMessage)
        binding.numRunsTextView.text = ""
        binding.avgLatencyTextView.text = ""
        binding.p50LatencyTextView.text = ""
        binding.p90LatencyTextView.text = ""
        binding.p99LatencyTextView.text = ""
        binding.minLatencyTextView.text = ""
        binding.maxLatencyTextView.text = ""
        binding.latencyTitleTextView.text = getString(R.string.latency_title_label) // Keep title

        if (stackTrace != null) {
            binding.rawSummaryText.text = getString(R.string.exception_prefix, stackTrace)
            binding.rawSummaryText.visibility = View.VISIBLE
        } else {
            binding.rawSummaryText.text = ""
            binding.rawSummaryText.visibility = View.GONE
        }
    }

    /**
     * A native method that is implemented by the 'modeltester' native library,
     * which is packaged with this application.
     */
    external fun run(modelPathOrBytes: Any, // Can be String (path) or ByteArray (bytes)
                     numIterations: Int,
                     runWarmupIteration: Boolean,
                     executionProviderType: String?,
                     executionProviderOptionNames: Array<String>?,
                     executionProviderOptionValues: Array<String>?,
                     logLevel: Int
    ): String

    companion object {
        // Used to load the 'modeltester' library on application startup.
        init {
            System.loadLibrary("modeltester")
        }
    }
}