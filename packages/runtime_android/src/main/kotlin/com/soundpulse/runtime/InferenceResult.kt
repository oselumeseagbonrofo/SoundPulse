package com.soundpulse.runtime

data class Candidate(
    val label: String,
    val distance: Float,
    val confidence: Float,
)

data class InferenceFlags(
    val isUnknown: Boolean,
    val isBackground: Boolean,
    val speechPitchRequired: Boolean,
)

data class SpeechPitchCue(
    val f0Hz: Float,
    val bucket: String,
)

data class InferenceResult(
    val label: String,
    val distance: Float,
    val confidence: Float,
    val confidenceTier: String,
    val topCandidates: List<Candidate>,
    val flags: InferenceFlags,
    val speechPitch: SpeechPitchCue? = null,
)
