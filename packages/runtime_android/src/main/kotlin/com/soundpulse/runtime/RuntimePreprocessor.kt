package com.soundpulse.runtime

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

class RuntimePreprocessor(
    private val config: PreprocessingConfig,
) {
    fun prepareWindow(windowPcmFloat32: FloatArray): FloatArray {
        return fitToWindow(windowPcmFloat32, config.windowSamples)
    }

    fun loudnessDbfs(windowPcmFloat32: FloatArray): Float {
        if (windowPcmFloat32.isEmpty()) {
            return -120f
        }

        var sumSquares = 0.0
        for (sample in windowPcmFloat32) {
            sumSquares += sample * sample
        }
        val rms = sqrt(sumSquares / windowPcmFloat32.size.toDouble())
        return (20.0 * ln(max(rms, 1e-8)) / ln(10.0)).toFloat()
    }

    private fun fitToWindow(input: FloatArray, targetSize: Int): FloatArray {
        if (input.size == targetSize) {
            return input.copyOf()
        }

        val output = FloatArray(targetSize)
        val copySize = input.size.coerceAtMost(targetSize)
        for (index in 0 until copySize) {
            output[index] = input[index]
        }
        return output
    }

}
