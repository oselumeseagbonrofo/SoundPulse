package com.soundpulse.runtime

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

private object Assert {
    fun that(condition: Boolean, message: String) {
        if (!condition) {
            throw IllegalStateException(message)
        }
    }

    fun near(actual: Float, expected: Float, tolerance: Float, label: String) {
        val delta = abs(actual - expected)
        if (delta > tolerance) {
            throw IllegalStateException("$label expected $expected (+/-$tolerance), got $actual")
        }
    }
}

private class FakeEmbeddingExtractor(
    private var nextEmbedding: FloatArray,
) : EmbeddingExtractor {
    override fun extractEmbedding(preprocessedWindow: FloatArray): FloatArray = nextEmbedding

    fun setEmbedding(embedding: FloatArray) {
        nextEmbedding = embedding
    }
}

private class FakeLiteRtInterpreter : LiteRtInterpreter {
    var lastInput: Any? = null

    override fun getInputShape(): IntArray = intArrayOf(1, 96, 101, 1)

    override fun run(input: Any, output: Any) {
        lastInput = input
        val outputArray = output as FloatArray
        for (index in outputArray.indices) {
            outputArray[index] = index.toFloat()
        }
    }
}

fun main() {
    testAudioWindowSegmenter()
    testPrototypeScorer()
    testRuntimePreprocessor()
    testTinyCnnFeatureBuilder()
    testTinyCnnEmbeddingExtractor()
    testSpeechPitchEstimator()
    testRuntimePolicyInvariants()
    testLiteRtRuntimeEngineIntegration()
    println("PASS: Runtime verification harness checks succeeded")
}

private fun testAudioWindowSegmenter() {
    val segmenter = AudioWindowSegmenter(windowSamples = 4, hopSamples = 2)
    Assert.that(segmenter.append(floatArrayOf()).isEmpty(), "segmenter should ignore empty append")
    Assert.that(segmenter.append(floatArrayOf(1f, 2f, 3f)).isEmpty(), "segmenter should wait for full window")

    val first = segmenter.append(floatArrayOf(4f, 5f, 6f, 7f))
    Assert.that(first.size == 2, "segmenter should emit two overlapping windows")
    Assert.that(first[0].contentEquals(floatArrayOf(1f, 2f, 3f, 4f)), "first window mismatch")
    Assert.that(first[1].contentEquals(floatArrayOf(3f, 4f, 5f, 6f)), "second window mismatch")

    segmenter.reset()
    Assert.that(segmenter.append(floatArrayOf(9f, 10f, 11f)).isEmpty(), "reset must clear queued samples")
}

private fun testPrototypeScorer() {
    val scorer = PrototypeScorer(prototypes = mapOf("speech" to floatArrayOf(0f, 0f), "dog_barking" to floatArrayOf(1f, 1f)))
    val ranked = scorer.score(embedding = floatArrayOf(0.1f, 0.1f), topN = 2)
    Assert.that(ranked.size == 2, "prototype scorer should return requested topN")
    Assert.that(ranked[0].label == "speech", "nearest prototype should rank first")
    Assert.that(ranked[0].distance <= ranked[1].distance, "ranked distances must be ascending")
    Assert.that(ranked[0].confidence >= ranked[1].confidence, "confidence should decrease with distance")
}

private fun testRuntimePreprocessor() {
    val config = PreprocessingConfig(4, 1, 1000, 500, 4, 4)
    val preprocessor = RuntimePreprocessor(config)

    val prepared = preprocessor.prepareWindow(floatArrayOf(1f, 2f, 3f, 4f))
    Assert.near(prepared[0], 1f, 1e-4f, "preprocessor first sample")
    Assert.near(prepared[1], 2f, 1e-4f, "preprocessor second sample")
    Assert.near(prepared[2], 3f, 1e-4f, "preprocessor third sample")
    Assert.near(prepared[3], 4f, 1e-4f, "preprocessor fourth sample")

    val loudWindow = preprocessor.loudnessDbfs(floatArrayOf(1f, -1f, 1f, -1f))
    Assert.near(loudWindow, 0f, 1e-4f, "loudness for full-scale waveform")

    val silentWindow = preprocessor.loudnessDbfs(floatArrayOf(0f, 0f, 0f, 0f))
    Assert.that(silentWindow < -100f, "silence loudness should be very low")
}

private fun testTinyCnnFeatureBuilder() {
    val config = tinyCnnTestConfig()
    val builder = TinyCnnFeatureBuilder(config)
    val samples = FloatArray(config.windowSamples)
    for (index in samples.indices) {
        val phase = 2.0 * PI * 440.0 * index.toDouble() / config.sampleRateHz.toDouble()
        samples[index] = sin(phase).toFloat()
    }

    val tensor = builder.buildImageTensor(samples)
    Assert.that(tensor.size == 1, "TinyCNN tensor batch dimension mismatch")
    Assert.that(tensor[0].size == 96, "TinyCNN tensor height mismatch")
    Assert.that(tensor[0][0].size == 101, "TinyCNN tensor width mismatch")
    Assert.that(tensor[0][0][0].size == 1, "TinyCNN tensor channel mismatch")

    val flat = builder.buildFlatImage(samples)
    var sum = 0.0
    var sumSquares = 0.0
    for (value in flat) {
        Assert.that(!value.isNaN() && !value.isInfinite(), "TinyCNN feature contains non-finite value")
        sum += value
        sumSquares += value * value
    }
    val mean = sum / flat.size
    val variance = (sumSquares / flat.size) - (mean * mean)
    Assert.that(variance > 0.01, "TinyCNN feature should retain variance for a sine wave")
    Assert.near(mean.toFloat(), 0f, 0.05f, "TinyCNN standardized sine mean")
    Assert.near(kotlin.math.sqrt(variance).toFloat(), 1f, 0.05f, "TinyCNN standardized sine stddev")

    val anchorIndices = intArrayOf(0, 500, 1000, 1500, 2000, 3000, 4000, 5000)
    val referenceAnchors =
        floatArrayOf(
            -0.3368011f,
            -0.26583502f,
            0.7662022f,
            4.1306567f,
            0.36138022f,
            -0.33906585f,
            -0.3390998f,
            -0.33915687f,
        )
    for (index in anchorIndices.indices) {
        Assert.near(
            flat[anchorIndices[index]],
            referenceAnchors[index],
            1e-4f,
            "TinyCNN sine anchor ${anchorIndices[index]}",
        )
    }

    val silent = builder.buildFlatImage(FloatArray(config.windowSamples))
    for (value in silent) {
        Assert.that(!value.isNaN() && !value.isInfinite(), "Silent TinyCNN feature contains non-finite value")
    }

    val shortFeature = builder.buildFlatImage(floatArrayOf(0.25f, -0.25f))
    Assert.that(shortFeature.size == 96 * 101 * 1, "Short TinyCNN feature should be padded to fixed image size")
}

private fun testTinyCnnEmbeddingExtractor() {
    val interpreter = FakeLiteRtInterpreter()
    val extractor =
        TinyCnnEmbeddingExtractor(
            interpreter = interpreter,
            preprocessing = tinyCnnTestConfig(),
            embeddingDim = 4,
        )
    val embedding = extractor.extractEmbedding(FloatArray(tinyCnnTestConfig().windowSamples) { 0.1f })
    Assert.that(embedding.contentEquals(floatArrayOf(0f, 1f, 2f, 3f)), "TinyCNN embedding output mismatch")

    val input = interpreter.lastInput
    Assert.that(input is Array<*>, "TinyCNN interpreter should receive a 4D tensor")
    val batch = input as Array<*>
    Assert.that(batch.size == 1, "TinyCNN interpreter batch size mismatch")
}

private fun testSpeechPitchEstimator() {
    val estimator = SpeechPitchEstimator()
    val sampleRate = 16000
    val freqHz = 120f
    val samples = FloatArray(sampleRate)
    for (index in samples.indices) {
        val phase = (2.0 * PI * freqHz.toDouble() * index.toDouble()) / sampleRate.toDouble()
        samples[index] = sin(phase).toFloat()
    }

    val cue = estimator.estimate(samples, sampleRate)
    Assert.that(cue.bucket == "male_like", "120Hz tone should map to male_like")
    Assert.that(cue.f0Hz in 100f..140f, "pitch estimate should be near expected tone")

    val invalid = estimator.estimate(floatArrayOf(0.1f), 0)
    Assert.that(invalid.bucket == "unknown", "invalid input must produce unknown pitch bucket")
}

private fun testRuntimePolicyInvariants() {
    val policy = RuntimePolicy()
    val candidates = listOf(Candidate("speech", 0.1f, 0.9f))

    val background =
        policy.applyInvariants(
            InferenceResult(
                label = "speech",
                distance = 0.1f,
                confidence = 0.95f,
                confidenceTier = "high",
                topCandidates = candidates,
                flags = InferenceFlags(isUnknown = true, isBackground = true, speechPitchRequired = true),
                speechPitch = SpeechPitchCue(100f, "male_like"),
            ),
        )
    Assert.that(background.label == "background", "background invariant must force label")
    Assert.that(background.confidence <= 0.2f, "background confidence must be capped")
    Assert.that(!background.flags.isUnknown && background.flags.isBackground, "background flags mismatch")
    Assert.that(background.speechPitch == null, "background result cannot carry pitch cue")

    val unknown =
        policy.applyInvariants(
            InferenceResult(
                label = "dog_barking",
                distance = 0.4f,
                confidence = 0.8f,
                confidenceTier = "high",
                topCandidates = candidates,
                flags = InferenceFlags(isUnknown = true, isBackground = false, speechPitchRequired = true),
                speechPitch = SpeechPitchCue(110f, "male_like"),
            ),
        )
    Assert.that(unknown.label == "unknown", "unknown invariant must force label")
    Assert.that(unknown.confidence <= 0.3f, "unknown confidence must be capped")
    Assert.that(unknown.flags.isUnknown && !unknown.flags.isBackground, "unknown flags mismatch")
    Assert.that(unknown.speechPitch == null, "unknown result cannot carry pitch cue")
}

private fun testLiteRtRuntimeEngineIntegration() {
    val config = PreprocessingConfig(4, 1, 1000, 500, 2, 2)
    val thresholds =
        ThresholdConfig(
            distanceThreshold = 0.5f,
            loudnessFloorDbfs = -40f,
            unknownRatioThreshold = 0.95f,
            confidenceTiers = ConfidenceTiers(high = 0.8f, medium = 0.5f, low = 0f),
            modes =
                mapOf(
                    "general" to ModeThresholds(0f, 0f, emptySet()),
                    "focus" to ModeThresholds(0f, 0f, emptySet()),
                    "sleep" to ModeThresholds(0f, 0f, setOf("speech")),
                ),
        )
    val extractor = FakeEmbeddingExtractor(floatArrayOf(0f, 0f))
    val engine =
        LiteRtRuntimeEngine(
            preprocessing = config,
            thresholds = thresholds,
            embeddingExtractor = extractor,
            prototypeScorer = PrototypeScorer(mapOf("speech" to floatArrayOf(0f, 0f), "dog_barking" to floatArrayOf(1f, 1f))),
        )

    val idle = engine.pushAudioPcm(floatArrayOf(1f, -1f, 1f, -1f))
    Assert.that(idle.isEmpty(), "engine should not emit results before start")

    engine.start(Mode.GENERAL)
    val active = engine.pushAudioPcm(floatArrayOf(1f, -1f, 1f, -1f))
    Assert.that(active.size == 1, "engine should emit one window result when enough samples are available")
    Assert.that(active[0].label == "speech", "best embedding match should classify as speech")
    Assert.that(active[0].flags.speechPitchRequired, "speech output should require pitch cue")
    Assert.that(active[0].speechPitch != null, "speech output should include pitch cue")

    extractor.setEmbedding(floatArrayOf(10f, 10f))
    val unknown = engine.inferWindow(floatArrayOf(1f, -1f, 1f, -1f))
    Assert.that(unknown.label == "unknown", "far embedding should be routed to unknown")

    engine.updateMode(Mode.SLEEP)
    extractor.setEmbedding(floatArrayOf(0f, 0f))
    val sleepResult = engine.inferWindow(floatArrayOf(1f, -1f, 1f, -1f))
    Assert.that(sleepResult.label == "background", "sleep mode suppression should route speech to background")

    engine.stop()
    val stopped = engine.pushAudioPcm(floatArrayOf(1f, -1f, 1f, -1f))
    Assert.that(stopped.isEmpty(), "engine should stop emitting after stop()")
}

private fun tinyCnnTestConfig(): PreprocessingConfig =
    PreprocessingConfig(
        sampleRateHz = 16000,
        channels = 1,
        windowDurationMs = 1000,
        hopDurationMs = 480,
        tinyCnnEmbeddingDim = 1024,
        yamnetEmbeddingDim = 1024,
        stft = StftConfig(nFft = 512, hopLength = 160, winLength = 480, window = "hann", center = false),
        mel = MelConfig(nMels = 96, fMinHz = 0f, fMaxHz = 8000f, power = 2f, logOffset = 0.001f),
        normalization = NormalizationConfig(type = "log_mel", clipMinDb = -80f, clipMaxDb = 0f),
    )
