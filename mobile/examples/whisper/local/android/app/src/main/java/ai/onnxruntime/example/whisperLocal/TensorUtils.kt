package ai.onnxruntime.example.whisperLocal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.FloatBuffer
import java.nio.IntBuffer

internal fun createIntTensor(env: OrtEnvironment, data: IntArray, shape: LongArray): OnnxTensor {
    return OnnxTensor.createTensor(env, IntBuffer.wrap(data), shape)
}

internal fun createFloatTensor(
    env: OrtEnvironment,
    data: FloatArray,
    shape: LongArray
): OnnxTensor {
    return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
}

internal fun tensorShape(vararg dims: Long) = longArrayOf(*dims)