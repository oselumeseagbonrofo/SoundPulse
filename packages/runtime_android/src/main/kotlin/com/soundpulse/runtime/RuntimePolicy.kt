package com.soundpulse.runtime

class RuntimePolicy {
    fun applyInvariants(raw: InferenceResult): InferenceResult {
        if (raw.flags.isBackground) {
            return raw.copy(
                label = "background",
                confidence = raw.confidence.coerceAtMost(0.2f),
                confidenceTier = "low",
                flags = raw.flags.copy(
                    isUnknown = false,
                    isBackground = true,
                    speechPitchRequired = false,
                ),
                speechPitch = null,
            )
        }

        if (raw.flags.isUnknown) {
            return raw.copy(
                label = "unknown",
                confidence = raw.confidence.coerceAtMost(0.3f),
                confidenceTier = "low",
                flags = raw.flags.copy(
                    isUnknown = true,
                    isBackground = false,
                    speechPitchRequired = false,
                ),
                speechPitch = null,
            )
        }

        if (raw.label != "speech") {
            return raw.copy(
                flags = raw.flags.copy(speechPitchRequired = false),
                speechPitch = null,
            )
        }

        return raw.copy(
            flags = raw.flags.copy(speechPitchRequired = true),
        )
    }
}
