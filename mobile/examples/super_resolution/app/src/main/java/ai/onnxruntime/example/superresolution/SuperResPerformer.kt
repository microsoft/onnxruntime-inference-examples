package ai.onnxruntime.example.superresolution

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.InputStream
import java.nio.ByteBuffer
import java.util.*


// TODO: modify to super res use case
internal data class Result(
    // TODO: output image bytes -> display to png
    var outputBitmap: Bitmap? = null
) {}

internal class SuperResPerformer(
    private val ortSession: OrtSession?,
    private val result: Result?
) {

    public fun analyze(inputStream: InputStream) {
        // Step 1: convert image into byte array (raw image bytes)
        val rawImageBytes = inputStream.readBytes()

        // Step 2: get the shape of the byte array and make ort tensor
        val shape = longArrayOf(rawImageBytes.size.toLong())
        val env = OrtEnvironment.getEnvironment()
        env.use {
            val inputTensor = OnnxTensor.createTensor(
                env,
                ByteBuffer.wrap(rawImageBytes),
                shape,
                OnnxJavaType.UINT8
            )
            inputTensor.use {
                // Step 3: call ort inferenceSession run
                val output = ortSession?.run(Collections.singletonMap("image", inputTensor))

                // Step 4: output analysis
                output.use {
                    @Suppress("UNCHECKED_CAST")
                    val rawOutput = (output?.get(0)?.value) as ByteArray
                    val outputImageBitmap =
                        byteArrayToBitmap(rawOutput)

                // Step 5: set output result
                    this.result?.outputBitmap = outputImageBitmap
                }
            }
        }
    }

    fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}