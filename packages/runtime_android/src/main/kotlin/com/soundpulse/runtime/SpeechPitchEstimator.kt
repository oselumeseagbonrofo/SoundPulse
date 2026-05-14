package com.soundpulse.runtime

class SpeechPitchEstimator {
    fun estimate(windowPcmFloat32: FloatArray, sampleRateHz: Int): SpeechPitchCue {
        if (windowPcmFloat32.size < 2 || sampleRateHz <= 0) {
            return SpeechPitchCue(
                f0Hz = 0f,
                bucket = "unknown",
            )
        }

        var zeroCrossings = 0
        for (index in 1 until windowPcmFloat32.size) {
            val previous = windowPcmFloat32[index - 1]
            val current = windowPcmFloat32[index]
            val crossed =
                (previous >= 0f && current < 0f) ||
                    (previous < 0f && current >= 0f)
            if (crossed) {
                zeroCrossings += 1
            }
        }

        val estimatedF0 = (zeroCrossings.toFloat() * sampleRateHz) / (2f * windowPcmFloat32.size)
        val bucket =
            when {
                estimatedF0 in 85f..165f -> "male_like"
                estimatedF0 in 165f..255f -> "female_like"
                else -> "unknown"
            }

        return SpeechPitchCue(
            f0Hz = estimatedF0,
            bucket = bucket,
        )
    }
}
