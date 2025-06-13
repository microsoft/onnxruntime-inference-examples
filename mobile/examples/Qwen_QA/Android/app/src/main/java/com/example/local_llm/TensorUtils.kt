package com.example.local_llm

import ai.onnxruntime.*

import java.nio.ShortBuffer

fun Float.toFloat16Bits(): Short {
    val bits = java.lang.Float.floatToIntBits(this)
    val sign = (bits ushr 16) and 0x8000
    val exponent = ((bits ushr 23) and 0xFF) - 127 + 15
    val mantissa = (bits shr 13) and 0x3FF

    return when {
        exponent <= 0 -> sign.toShort()
        exponent >= 0x1F -> (sign or 0x7C00).toShort()
        else -> (sign or (exponent shl 10) or mantissa).toShort()
    }
}

fun createFloat16Tensor(env: OrtEnvironment, floatArray: FloatArray, shape: LongArray): OnnxTensor {
    val float16Shorts = ShortArray(floatArray.size) { i -> floatArray[i].toFloat16Bits() }
    val buffer = ShortBuffer.wrap(float16Shorts)
    return OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT16)
}
