package ai.onnxruntime.example.whisperAzure

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

class AudioTensorSource {
    companion object {
        private const val sampleRate = 16000
        private const val maxAudioLengthInSeconds = 30

        fun fromRawWavBytes(rawBytes: ByteArray): OnnxTensor {
            val rawByteBuffer = ByteBuffer.wrap(rawBytes)
            // TODO handle big-endian native order...
            if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
                throw NotImplementedError("Reading Wav data is only supported when native byte order is little-endian.")
            }
            rawByteBuffer.order(ByteOrder.nativeOrder())
            val numBytes = minOf(rawByteBuffer.capacity(), maxAudioLengthInSeconds * sampleRate)
            val env = OrtEnvironment.getEnvironment()
            return OnnxTensor.createTensor(
                env, rawByteBuffer, tensorShape(numBytes.toLong()), OnnxJavaType.UINT8)
        }

        @SuppressLint("MissingPermission")
        fun fromRecording(stopRecordingFlag: AtomicBoolean): OnnxTensor {
            val recordingChunkLengthInSeconds = 1

            val minBufferSize = maxOf(
                AudioRecord.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_8BIT,
                ),
                2 * recordingChunkLengthInSeconds * sampleRate
            )

            val audioRecord = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.MIC)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(sampleRate)
                        .setEncoding(AudioFormat.ENCODING_PCM_8BIT)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .build()
                )
                .setBufferSizeInBytes(minBufferSize)
                .build()

            try {
                val audioData = ByteArray(maxAudioLengthInSeconds * sampleRate) { 0 }
                var audioDataOffset = 0

                audioRecord.startRecording()

                while (!stopRecordingFlag.get() && audioDataOffset < audioData.size) {
                    val numBytesToRead = minOf(
                        recordingChunkLengthInSeconds * sampleRate,
                        audioData.size - audioDataOffset
                    )

                    val readResult = audioRecord.read(
                        audioData, audioDataOffset, numBytesToRead,
                        AudioRecord.READ_BLOCKING
                    )

                    Log.d(MainActivity.TAG, "AudioRecord.read(byte[], ...) returned $readResult")

                    if (readResult >= 0) {
                        audioDataOffset += readResult
                    } else {
                        throw RuntimeException("AudioRecord.read() returned error code $readResult")
                    }
                }

                audioRecord.stop()

                val env = OrtEnvironment.getEnvironment()
                val audioDataBuffer = ByteBuffer.wrap(audioData)

                return OnnxTensor.createTensor(
                    env, audioDataBuffer,
                    tensorShape(audioData.size.toLong()), OnnxJavaType.UINT8)

            } finally {
                if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    audioRecord.stop()
                }
                audioRecord.release()
            }
        }
    }

}