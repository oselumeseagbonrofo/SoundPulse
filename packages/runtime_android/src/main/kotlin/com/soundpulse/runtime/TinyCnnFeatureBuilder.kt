package com.soundpulse.runtime

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

class TinyCnnFeatureBuilder(
    private val config: PreprocessingConfig,
    private val imageHeight: Int = 96,
    private val imageWidth: Int = 101,
    private val imageChannels: Int = 1,
) {
    private val stft = config.stft
    private val mel = config.mel
    private val normalization = config.normalization
    private val frameCount = 1 + (config.windowSamples - stft.nFft) / stft.hopLength
    private val fftBinCount = stft.nFft / 2 + 1
    private val hannWindow = buildCenteredHannWindow()
    private val melFilterBank = buildMelFilterBank()
    private val cosTable = Array(fftBinCount) { DoubleArray(stft.nFft) }
    private val sinTable = Array(fftBinCount) { DoubleArray(stft.nFft) }

    init {
        require(config.channels == 1) { "TinyCNN preprocessing expects mono audio" }
        require(!stft.center) { "TinyCNN preprocessing requires center=false" }
        require(stft.window.lowercase() == "hann") { "TinyCNN preprocessing supports only Hann windows" }
        require(stft.winLength <= stft.nFft) { "STFT win_length must be <= n_fft" }
        require(config.windowSamples >= stft.nFft) { "window_samples must be >= n_fft" }
        require(mel.nMels > 0 && mel.fMaxHz > mel.fMinHz) { "Invalid mel configuration" }
        require(mel.power > 0f) { "mel power must be positive" }

        for (bin in 0 until fftBinCount) {
            for (sample in 0 until stft.nFft) {
                val phase = 2.0 * PI * bin.toDouble() * sample.toDouble() / stft.nFft.toDouble()
                cosTable[bin][sample] = cos(phase)
                sinTable[bin][sample] = sin(phase)
            }
        }
    }

    fun buildImageTensor(windowPcmFloat32: FloatArray): Array<Array<Array<FloatArray>>> {
        val logMel = waveformToLogMel(fitToWindow(windowPcmFloat32))
        val resized = resizeBilinear(logMel, imageHeight, imageWidth)
        val standardized = standardize(resized)
        return Array(1) {
            Array(imageHeight) { row ->
                Array(imageWidth) { col ->
                    FloatArray(imageChannels) { standardized[row][col] }
                }
            }
        }
    }

    fun buildFlatImage(windowPcmFloat32: FloatArray): FloatArray {
        val tensor = buildImageTensor(windowPcmFloat32)
        val flat = FloatArray(imageHeight * imageWidth * imageChannels)
        var index = 0
        for (row in 0 until imageHeight) {
            for (col in 0 until imageWidth) {
                for (channel in 0 until imageChannels) {
                    flat[index++] = tensor[0][row][col][channel]
                }
            }
        }
        return flat
    }

    private fun fitToWindow(input: FloatArray): FloatArray {
        if (input.size == config.windowSamples) {
            return input.copyOf()
        }
        val output = FloatArray(config.windowSamples)
        val copySize = input.size.coerceAtMost(config.windowSamples)
        for (index in 0 until copySize) {
            output[index] = input[index]
        }
        return output
    }

    private fun waveformToLogMel(window: FloatArray): Array<FloatArray> {
        val spectrogram = Array(fftBinCount) { FloatArray(frameCount) }
        for (frameIndex in 0 until frameCount) {
            val frameStart = frameIndex * stft.hopLength
            for (bin in 0 until fftBinCount) {
                var real = 0.0
                var imag = 0.0
                for (sample in 0 until stft.nFft) {
                    val value = window[frameStart + sample].toDouble() * hannWindow[sample]
                    real += value * cosTable[bin][sample]
                    imag -= value * sinTable[bin][sample]
                }
                val magnitude = sqrt(real * real + imag * imag)
                spectrogram[bin][frameIndex] = magnitude.pow(mel.power.toDouble()).toFloat()
            }
        }

        val logMel = Array(mel.nMels) { FloatArray(frameCount) }
        for (melIndex in 0 until mel.nMels) {
            for (frameIndex in 0 until frameCount) {
                var energy = 0.0
                for (bin in 0 until fftBinCount) {
                    energy += melFilterBank[melIndex][bin] * spectrogram[bin][frameIndex]
                }
                val logged = ln(energy + mel.logOffset.toDouble()).toFloat()
                logMel[melIndex][frameIndex] =
                    logged.coerceIn(normalization.clipMinDb, normalization.clipMaxDb)
            }
        }
        return logMel
    }

    private fun buildCenteredHannWindow(): DoubleArray {
        val padded = DoubleArray(stft.nFft)
        val leftPad = (stft.nFft - stft.winLength) / 2
        for (index in 0 until stft.winLength) {
            padded[leftPad + index] = 0.5 - 0.5 * cos(2.0 * PI * index.toDouble() / stft.winLength.toDouble())
        }
        return padded
    }

    private fun buildMelFilterBank(): Array<FloatArray> {
        val fftFrequencies = DoubleArray(fftBinCount) { index ->
            config.sampleRateHz.toDouble() * index.toDouble() / stft.nFft.toDouble()
        }
        val minMel = hzToMel(mel.fMinHz.toDouble())
        val maxMel = hzToMel(mel.fMaxHz.toDouble())
        val melFrequencies = DoubleArray(mel.nMels + 2) { index ->
            melToHz(minMel + (maxMel - minMel) * index.toDouble() / (mel.nMels + 1).toDouble())
        }

        val weights = Array(mel.nMels) { FloatArray(fftBinCount) }
        for (melIndex in 0 until mel.nMels) {
            val lower = melFrequencies[melIndex]
            val center = melFrequencies[melIndex + 1]
            val upper = melFrequencies[melIndex + 2]
            val enorm = 2.0 / (upper - lower)

            for (bin in 0 until fftBinCount) {
                val frequency = fftFrequencies[bin]
                val lowerSlope = (frequency - lower) / (center - lower)
                val upperSlope = (upper - frequency) / (upper - center)
                weights[melIndex][bin] = (max(0.0, min(lowerSlope, upperSlope)) * enorm).toFloat()
            }
        }
        return weights
    }

    private fun resizeBilinear(input: Array<FloatArray>, targetHeight: Int, targetWidth: Int): Array<FloatArray> {
        val sourceHeight = input.size
        val sourceWidth = input.firstOrNull()?.size ?: 0
        val output = Array(targetHeight) { FloatArray(targetWidth) }
        val heightScale = sourceHeight.toDouble() / targetHeight.toDouble()
        val widthScale = sourceWidth.toDouble() / targetWidth.toDouble()

        for (row in 0 until targetHeight) {
            val sourceRow = row * heightScale
            val row0 = floor(sourceRow).toInt().coerceIn(0, sourceHeight - 1)
            val row1 = (row0 + 1).coerceAtMost(sourceHeight - 1)
            val rowWeight = sourceRow - row0
            for (col in 0 until targetWidth) {
                val sourceCol = col * widthScale
                val col0 = floor(sourceCol).toInt().coerceIn(0, sourceWidth - 1)
                val col1 = (col0 + 1).coerceAtMost(sourceWidth - 1)
                val colWeight = sourceCol - col0

                val top = input[row0][col0] * (1.0 - colWeight) + input[row0][col1] * colWeight
                val bottom = input[row1][col0] * (1.0 - colWeight) + input[row1][col1] * colWeight
                output[row][col] = (top * (1.0 - rowWeight) + bottom * rowWeight).toFloat()
            }
        }
        return output
    }

    private fun standardize(input: Array<FloatArray>): Array<FloatArray> {
        val pixelCount = input.size * input[0].size * imageChannels
        var sum = 0.0
        for (row in input.indices) {
            for (col in input[row].indices) {
                sum += input[row][col] * imageChannels
            }
        }
        val mean = sum / pixelCount.toDouble()

        var squaredDeltaSum = 0.0
        for (row in input.indices) {
            for (col in input[row].indices) {
                val delta = input[row][col] - mean
                squaredDeltaSum += delta * delta * imageChannels
            }
        }
        val stddev = sqrt(squaredDeltaSum / pixelCount.toDouble())
        val adjustedStddev = max(stddev, 1.0 / sqrt(pixelCount.toDouble()))

        return Array(input.size) { row ->
            FloatArray(input[row].size) { col ->
                ((input[row][col] - mean) / adjustedStddev).toFloat()
            }
        }
    }

    private fun hzToMel(hz: Double): Double {
        val fSp = 200.0 / 3.0
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp
        val logStep = ln(6.4) / 27.0
        return if (hz >= minLogHz) {
            minLogMel + ln(hz / minLogHz) / logStep
        } else {
            hz / fSp
        }
    }

    private fun melToHz(melValue: Double): Double {
        val fSp = 200.0 / 3.0
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp
        val logStep = ln(6.4) / 27.0
        return if (melValue >= minLogMel) {
            minLogHz * exp(logStep * (melValue - minLogMel))
        } else {
            melValue * fSp
        }
    }
}
