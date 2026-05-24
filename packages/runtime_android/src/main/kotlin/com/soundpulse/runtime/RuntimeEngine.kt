package com.soundpulse.runtime

/**
 * Runtime seam for continuous audio inference.
 */
interface RuntimeEngine {
    fun start(mode: Mode)
    fun stop()
    fun updateMode(mode: Mode)
    fun pushAudioPcm(samplesPcmFloat32: FloatArray): List<InferenceResult>
    fun inferWindow(windowPcmFloat32: FloatArray): InferenceResult
}

class LiteRtRuntimeEngine(
    private val preprocessing: PreprocessingConfig,
    private val thresholds: ThresholdConfig,
    private val embeddingExtractor: EmbeddingExtractor,
    private val prototypeScorer: PrototypeScorer,
    private val policy: RuntimePolicy = RuntimePolicy(),
    private val pitchEstimator: SpeechPitchEstimator = SpeechPitchEstimator(),
    private val topN: Int = 3,
) : RuntimeEngine {
    private val segmenter = AudioWindowSegmenter(preprocessing.windowSamples, preprocessing.hopSamples)
    private val preprocessor = RuntimePreprocessor(preprocessing)
    private var currentMode: Mode = Mode.GENERAL
    private var monitoring = false

    override fun start(mode: Mode) {
        currentMode = mode
        monitoring = true
        segmenter.reset()
    }

    override fun stop() {
        monitoring = false
        segmenter.reset()
    }

    override fun updateMode(mode: Mode) {
        currentMode = mode
    }

    override fun pushAudioPcm(samplesPcmFloat32: FloatArray): List<InferenceResult> {
        if (!monitoring) {
            return emptyList()
        }
        return segmenter
            .append(samplesPcmFloat32)
            .map { inferWindow(it) }
    }

    override fun inferWindow(windowPcmFloat32: FloatArray): InferenceResult {
        val preprocessedWindow = preprocessor.prepareWindow(windowPcmFloat32)
        val loudnessDbfs = preprocessor.loudnessDbfs(preprocessedWindow)
        val ranked = prototypeScorer.score(embeddingExtractor.extractEmbedding(preprocessedWindow), topN)
        val best = ranked.firstOrNull()
        val second = ranked.getOrNull(1)
        val modeThresholds = thresholds.modes[currentMode.key]

        val effectiveDistanceThreshold =
            thresholds.distanceThreshold + (modeThresholds?.distanceThresholdDelta ?: 0f)
        val effectiveLoudnessFloor = thresholds.loudnessFloorDbfs + (modeThresholds?.loudnessFloorDeltaDb ?: 0f)
        val suppressedClasses = modeThresholds?.suppressedClasses ?: emptySet()

        val noBest = best == null
        val lowEnergy = loudnessDbfs < effectiveLoudnessFloor
        val suppressedByMode = best?.label in suppressedClasses
        val distanceUnknown = (best?.distance ?: Float.MAX_VALUE) > effectiveDistanceThreshold
        val ambiguityRatio = when {
            best == null || second == null || second.distance <= 1e-6f -> 0f
            else -> best.distance / second.distance
        }
        val ambiguousUnknown = ambiguityRatio >= thresholds.unknownRatioThreshold

        val isBackground = noBest || lowEnergy || suppressedByMode
        val isUnknown = !isBackground && (distanceUnknown || ambiguousUnknown)
        val outputLabel =
            when {
                isBackground -> "background"
                isUnknown -> "unknown"
                else -> best.label
            }

        val confidence = computeConfidence(best?.distance ?: 1f, effectiveDistanceThreshold)
        val tier = confidenceTier(confidence, thresholds.confidenceTiers)
        val speechPitchRequired = outputLabel == "speech" && !isUnknown && !isBackground
        val speechPitch =
            if (speechPitchRequired) {
                pitchEstimator.estimate(preprocessedWindow, preprocessing.sampleRateHz)
            } else {
                null
            }

        val raw =
            InferenceResult(
                label = outputLabel,
                distance = best?.distance ?: 1f,
                confidence = confidence,
                confidenceTier = tier,
                topCandidates = ranked,
                flags =
                    InferenceFlags(
                        isUnknown = isUnknown,
                        isBackground = isBackground,
                        speechPitchRequired = speechPitchRequired,
                    ),
                speechPitch = speechPitch,
            )

        return policy.applyInvariants(raw)
    }

    private fun computeConfidence(distance: Float, threshold: Float): Float {
        val boundedThreshold = threshold.coerceAtLeast(1e-6f)
        return (1f - (distance / boundedThreshold)).coerceIn(0f, 1f)
    }

    private fun confidenceTier(confidence: Float, tiers: ConfidenceTiers): String {
        return when {
            confidence >= tiers.high -> "high"
            confidence >= tiers.medium -> "medium"
            else -> "low"
        }
    }
}


private val Mode.key: String
    get() =
        when (this) {
            Mode.GENERAL -> "general"
            Mode.FOCUS -> "focus"
            Mode.SLEEP -> "sleep"
        }
