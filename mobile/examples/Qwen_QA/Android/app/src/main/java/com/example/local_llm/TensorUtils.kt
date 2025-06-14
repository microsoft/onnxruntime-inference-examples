package com.example.local_llm

import ai.onnxruntime.*
import java.nio.ShortBuffer

// Converts a Float32 to IEEE 754 float16 bit pattern stored as a Short
fun Float.toFloat16Bits(): Short {
    val bits = java.lang.Float.floatToIntBits(this)
    val sign = (bits ushr 16) and 0x8000                   // extract sign (1 bit)
    val exponent = ((bits ushr 23) and 0xFF) - 127 + 15    // adjust exponent bias (8 → 5 bits)
    val mantissa = (bits shr 13) and 0x3FF                 // truncate mantissa (23 → 10 bits)

    return when {
        exponent <= 0 -> sign.toShort()                   // subnormal or zero
        exponent >= 0x1F -> (sign or 0x7C00).toShort()    // NaN or infinity
        else -> (sign or (exponent shl 10) or mantissa).toShort()
    }
}

// Creates a float16 ONNX tensor from a FloatArray and shape
fun createFloat16Tensor(env: OrtEnvironment, floatArray: FloatArray, shape: LongArray): OnnxTensor {
    val float16Shorts = ShortArray(floatArray.size) { i -> floatArray[i].toFloat16Bits() }
    val buffer = ShortBuffer.wrap(float16Shorts)
    return OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT16)
}