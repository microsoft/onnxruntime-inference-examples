package com.onnxruntime.example.modeltester

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.onnxruntime.example.modeltester.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val modelResourceId = R.raw.model
        val modelBytes = resources.openRawResource(modelResourceId).readBytes()

        val summary = run(modelBytes, 10, null, null, null)

        binding.sampleText.text = summary
    }

    /**
     * A native method that is implemented by the 'modeltester' native library,
     * which is packaged with this application.
     */
    external fun run(modelBytes: ByteArray,
                     numIterations: Int,
                     executionProviderType: String?,
                     executionProviderOptionNames: Array<String>?,
                     executionProviderOptionValues: Array<String>?,
                     ): String

    companion object {
        // Used to load the 'modeltester' library on application startup.
        init {
            System.loadLibrary("modeltester")
        }
    }
}