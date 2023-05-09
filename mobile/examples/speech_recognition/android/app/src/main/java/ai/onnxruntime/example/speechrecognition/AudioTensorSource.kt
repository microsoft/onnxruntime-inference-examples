package ai.onnxruntime.example.speechrecognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.annotation.SuppressLint
import android.content.res.AssetFileDescriptor
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean

class AudioTensorSource {
    companion object {
        fun fromRawPcmFile(fd: AssetFileDescriptor): OnnxTensor {
            val inputStream = fd.createInputStream()
            inputStream.use {
                val rawBytes = inputStream.readBytes()
                val rawByteBuffer = ByteBuffer.wrap(rawBytes)
                assert(ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN)
                // TODO handle big-endian native order...
                rawByteBuffer.order(ByteOrder.nativeOrder())
                val floatBuffer = rawByteBuffer.asFloatBuffer()
                val env = OrtEnvironment.getEnvironment()
                return OnnxTensor.createTensor(env, floatBuffer, tensorShape(1, floatBuffer.capacity().toLong()))
            }
        }

        @SuppressLint("MissingPermission")
        fun fromRecording(stopRecordingFlag: AtomicBoolean): OnnxTensor {
            val recordingChunkLengthInSeconds = 1
            val sampleRate = 16000
            val bytesPerFloat = 4

            val minBufferSize = maxOf(
                AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT),
                2 * recordingChunkLengthInSeconds * sampleRate * bytesPerFloat)

            val audioRecord = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.MIC)
                .setAudioFormat(AudioFormat.Builder()
                    .setSampleRate(16000)
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build())
                .setBufferSizeInBytes(minBufferSize)
                .build()

            try {
                val maxRecordingLengthInSeconds = 30

                val floatAudioData = FloatArray(maxRecordingLengthInSeconds * sampleRate) { 0.0f }
                var floatAudioDataOffset = 0

                audioRecord.startRecording()

                while (!stopRecordingFlag.get() && floatAudioDataOffset < floatAudioData.size) {
                    val numFloatsToRead = minOf(
                        recordingChunkLengthInSeconds * sampleRate,
                        floatAudioData.size - floatAudioDataOffset
                    )

                    val readResult = audioRecord.read(
                        floatAudioData, floatAudioDataOffset, numFloatsToRead,
                        AudioRecord.READ_BLOCKING
                    )

                    Log.d(MainActivity.TAG, "AudioRecord.read(float[], ...) returned $readResult")

                    if (readResult >= 0) {
                        floatAudioDataOffset += readResult
                    } else {
                        throw RuntimeException("AudioRecord.read() returned error code $readResult")
                    }
                }

                audioRecord.stop()

                val env = OrtEnvironment.getEnvironment()
                val floatAudioDataBuffer = FloatBuffer.wrap(floatAudioData)

                return OnnxTensor.createTensor(
                    env, floatAudioDataBuffer,
                    tensorShape(1, floatAudioData.size.toLong()))

            } finally {
                if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    audioRecord.stop()
                }
                audioRecord.release()
            }
        }
    }

}