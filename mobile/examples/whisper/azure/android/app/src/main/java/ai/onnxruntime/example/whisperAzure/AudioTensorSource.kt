package ai.onnxruntime.example.whisperAzure

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.IOException
import java.io.FileOutputStream

class AudioTensorSource {
    companion object {
        private const val sampleRate = 16000
        private const val maxAudioLengthInSeconds = 30

        private fun readWavFile(filePath: String): ByteArray {
            val file = File(filePath)
            var inputStream = FileInputStream(file)
            inputStream.use {
                val fileLength = file.length().toInt()
                val fileData = ByteArray(fileLength)
                inputStream.read(fileData)
                return fileData
            }
        }

        private fun convertPcmToWav(
            pcmData: ByteArray,
            sampleRate: Int,
            bitsPerSample: Int,
            channels: Int,
            outputFilePath: String
        ) {
            val header = createWavHeader(pcmData.size, sampleRate, bitsPerSample, channels)
            val outputStream = FileOutputStream(outputFilePath)
            outputStream.use {
                // Write the WAV header
                outputStream.write(header)

                // Write the PCM audio data
                outputStream.write(pcmData)
            }
        }

        private fun createWavHeader(dataSize: Int, sampleRate: Int, bitsPerSample: Int, channels: Int): ByteArray {
            val headerSize = 44
            val totalSize = dataSize + headerSize - 8 // Subtract 8 for the RIFF chunk size

            val header = ByteArray(headerSize)

            // RIFF chunk descriptor
            header[0] = 'R'.code.toByte()
            header[1] = 'I'.code.toByte()
            header[2] = 'F'.code.toByte()
            header[3] = 'F'.code.toByte()
            ByteBuffer.wrap(header, 4, 4).order(ByteOrder.LITTLE_ENDIAN).putInt(totalSize)

            // Format chunk
            header[8] = 'W'.code.toByte()
            header[9] = 'A'.code.toByte()
            header[10] = 'V'.code.toByte()
            header[11] = 'E'.code.toByte()
            header[12] = 'f'.code.toByte()
            header[13] = 'm'.code.toByte()
            header[14] = 't'.code.toByte()
            header[15] = ' '.code.toByte()
            ByteBuffer.wrap(header, 16, 4).order(ByteOrder.LITTLE_ENDIAN).putInt(16)
            ByteBuffer.wrap(header, 20, 2).order(ByteOrder.LITTLE_ENDIAN).putShort(1.toShort()) // PCM format
            ByteBuffer.wrap(header, 22, 2).order(ByteOrder.LITTLE_ENDIAN).putShort(channels.toShort())
            ByteBuffer.wrap(header, 24, 4).order(ByteOrder.LITTLE_ENDIAN).putInt(sampleRate)
            ByteBuffer.wrap(header, 28, 4).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(sampleRate * channels * bitsPerSample / 8) // Byte rate
            ByteBuffer.wrap(header, 32, 2).order(ByteOrder.LITTLE_ENDIAN)
                .putShort((channels * bitsPerSample / 8).toShort()) // Block align
            ByteBuffer.wrap(header, 34, 2).order(ByteOrder.LITTLE_ENDIAN).putShort(bitsPerSample.toShort())

            // Data chunk
            header[36] = 'd'.code.toByte()
            header[37] = 'a'.code.toByte()
            header[38] = 't'.code.toByte()
            header[39] = 'a'.code.toByte()
            ByteBuffer.wrap(header, 40, 4).order(ByteOrder.LITTLE_ENDIAN).putInt(dataSize)

            return header
        }

        fun fromRawWavBytes(rawBytes: ByteArray, audioDataSize: Int = 0): OnnxTensor {
            val rawByteBuffer = ByteBuffer.wrap(rawBytes)
            var numBytes = 0
            if (audioDataSize == 0) {
                numBytes = minOf(rawByteBuffer.capacity(), maxAudioLengthInSeconds * sampleRate)
            } else {
                numBytes = audioDataSize + 44
            }
            val env = OrtEnvironment.getEnvironment()
            return OnnxTensor.createTensor(
                env, rawByteBuffer, tensorShape((numBytes).toLong()), OnnxJavaType.UINT8)
        }

        @SuppressLint("MissingPermission")
        fun fromRecording(stopRecordingFlag: AtomicBoolean, context: Context): OnnxTensor {
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

                // Convert raw pcm audio data and write to a wav format file
                val outputFile = File(context.getExternalFilesDir(null), "output.wav")
                val outputFilePath = outputFile.absolutePath
                convertPcmToWav(audioData, sampleRate, 8, 1, outputFilePath)
                val rawWavBytes = readWavFile(outputFilePath)
                return fromRawWavBytes(rawWavBytes, audioData.size)
            } finally {
                if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    audioRecord.stop()
                }
                audioRecord.release()
            }
        }

    }

}