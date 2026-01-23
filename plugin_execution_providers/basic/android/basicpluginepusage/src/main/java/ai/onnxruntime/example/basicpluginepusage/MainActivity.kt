package ai.onnxruntime.example.basicpluginepusage

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.example.basicpluginep.getLibraryPath as getBasicPluginEpLibraryPath
import ai.onnxruntime.example.basicpluginep.getEpName as getBasicPluginEpName
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import ai.onnxruntime.example.basicpluginepusage.ui.theme.BasicPluginEpTheme
import java.nio.FloatBuffer

class MainActivity : ComponentActivity() {
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private val pluginEpRegistrationName: String = "basic_ep_registration"

    private fun readResourceBytes(resourceId: Int): ByteArray {
        return resources.openRawResource(resourceId).readBytes()
    }

    private fun setUpOnnxRuntime() {
        ortEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)

        val pluginEpLibraryPath = getBasicPluginEpLibraryPath()
        ortEnv.registerExecutionProviderLibrary(pluginEpRegistrationName, pluginEpLibraryPath)

        val modelBytes = readResourceBytes(R.raw.mul)
        val sessionOptions = OrtSession.SessionOptions()
        val allEpDevices = ortEnv.epDevices
        val basicEpName = getBasicPluginEpName()
        val epDevicesToUse = allEpDevices.filter { it.epName == basicEpName }
        sessionOptions.addExecutionProvider(epDevicesToUse, emptyMap())
        ortSession = ortEnv.createSession(modelBytes, sessionOptions)
    }

    private fun tearDownOnnxRuntime() {
        ortEnv.unregisterExecutionProviderLibrary(pluginEpRegistrationName)

        ortSession.close()
        ortEnv.close()
    }

    private fun multiplyWithModel(x: Float, y: Float) : Float {
        val inputShape = longArrayOf(2, 3)
        val numElements = inputShape.reduce { product, dim -> product * dim }.toInt()

        val xValues = FloatBuffer.allocate(numElements)
        xValues.put(FloatArray(numElements) { x })
        xValues.rewind()

        val yValues = FloatBuffer.allocate(numElements)
        yValues.put(FloatArray(numElements) { y })
        yValues.rewind()

        val xTensor = OnnxTensor.createTensor(ortEnv, xValues, inputShape)
        return xTensor.use {
            val yTensor = OnnxTensor.createTensor(ortEnv, yValues, inputShape)
            yTensor.use {
                val outputs = ortSession.run(mapOf("x" to xTensor, "y" to yTensor))
                outputs.use {
                    val outputValues = outputs.get(0).value as Array<FloatArray>
                    outputValues[0][0]
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setUpOnnxRuntime()

        val x = 3.0f
        val y = 5.0f

        enableEdgeToEdge()
        setContent {
            BasicPluginEpTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Text(
                        text = "ONNX Runtime with a plugin EP! ${x} * ${y} is ${multiplyWithModel(x, y)}",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        tearDownOnnxRuntime()
    }
}
