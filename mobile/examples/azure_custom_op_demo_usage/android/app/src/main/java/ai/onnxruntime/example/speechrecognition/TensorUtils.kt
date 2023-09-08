package ai.onnxruntime.example.speechrecognition

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

internal fun createIntTensor(env: OrtEnvironment, data: IntArray, shape: LongArray): OnnxTensor {
    return OnnxTensor.createTensor(env, IntBuffer.wrap(data), shape)
}

internal fun createInt8Tensor(env: OrtEnvironment, data: ByteArray, shape: LongArray, type: OnnxJavaType): OnnxTensor {
    return OnnxTensor.createTensor(env, ByteBuffer.wrap(data), shape, type)
}

internal fun createStringTensor(env: OrtEnvironment, data: Array<String>, shape: LongArray): OnnxTensor {
    return OnnxTensor.createTensor(env, data, shape)
}

internal fun createFloatTensor(

    env: OrtEnvironment,
    data: FloatArray,
    shape: LongArray
): OnnxTensor {
    return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
}

internal fun tensorShape(vararg dims: Long) = longArrayOf(*dims)