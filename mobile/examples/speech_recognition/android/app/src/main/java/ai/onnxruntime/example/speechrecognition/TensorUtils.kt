package ai.onnxruntime.example.speechrecognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.FloatBuffer
import java.nio.IntBuffer

internal fun createIntTensor(env_: OrtEnvironment, data: IntArray, shape: LongArray): OnnxTensor {
    return OnnxTensor.createTensor(env_, IntBuffer.wrap(data), shape)
}

internal fun createFloatTensor(env_: OrtEnvironment, data: FloatArray, shape: LongArray): OnnxTensor {
    return OnnxTensor.createTensor(env_, FloatBuffer.wrap(data), shape)
}

internal fun tensorShape(vararg dims: Long) = longArrayOf(*dims)