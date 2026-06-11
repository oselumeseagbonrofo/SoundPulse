package com.soundpulse.runtime

import java.io.File
import kotlin.math.roundToInt
import org.json.JSONArray
import org.json.JSONObject

data class ClassDescriptor(
    val id: Int,
    val slug: String,
    val urgent: Boolean,
    val suppressed: Boolean,
)

data class PreprocessingConfig(
    val sampleRateHz: Int,
    val channels: Int,
    val windowDurationMs: Int,
    val hopDurationMs: Int,
    val tinyCnnEmbeddingDim: Int,
    val yamnetEmbeddingDim: Int,
    val stft: StftConfig = StftConfig(),
    val mel: MelConfig = MelConfig(),
    val normalization: NormalizationConfig = NormalizationConfig(),
) {
    val windowSamples: Int
        get() = (sampleRateHz * (windowDurationMs / 1000f)).roundToInt()

    val hopSamples: Int
        get() = (sampleRateHz * (hopDurationMs / 1000f)).roundToInt()
}

data class StftConfig(
    val nFft: Int = 512,
    val hopLength: Int = 160,
    val winLength: Int = 480,
    val window: String = "hann",
    val center: Boolean = false,
)

data class MelConfig(
    val nMels: Int = 96,
    val fMinHz: Float = 0f,
    val fMaxHz: Float = 8000f,
    val power: Float = 2f,
    val logOffset: Float = 0.001f,
)

data class NormalizationConfig(
    val type: String = "log_mel",
    val clipMinDb: Float = -80f,
    val clipMaxDb: Float = 0f,
)

data class ConfidenceTiers(
    val high: Float,
    val medium: Float,
    val low: Float,
)

data class ModeThresholds(
    val distanceThresholdDelta: Float,
    val loudnessFloorDeltaDb: Float,
    val suppressedClasses: Set<String>,
)

data class ThresholdConfig(
    val distanceThreshold: Float,
    val loudnessFloorDbfs: Float,
    val unknownRatioThreshold: Float,
    val confidenceTiers: ConfidenceTiers,
    val modes: Map<String, ModeThresholds>,
)

data class RuntimeArtifacts(
    val classes: List<ClassDescriptor>,
    val preprocessing: PreprocessingConfig,
    val thresholds: ThresholdConfig,
    val prototypes: Map<String, FloatArray>,
)

object RuntimeArtifactLoader {
    fun fromFiles(
        classLabelsFile: File,
        preprocessingConfigFile: File,
        thresholdDefaultsFile: File,
        prototypeStoreFile: File,
    ): RuntimeArtifacts {
        return fromJson(
            classLabelsJson = classLabelsFile.readText(Charsets.UTF_8),
            preprocessingConfigJson = preprocessingConfigFile.readText(Charsets.UTF_8),
            thresholdDefaultsJson = thresholdDefaultsFile.readText(Charsets.UTF_8),
            prototypeStoreJson = prototypeStoreFile.readText(Charsets.UTF_8),
        )
    }

    fun fromJson(
        classLabelsJson: String,
        preprocessingConfigJson: String,
        thresholdDefaultsJson: String,
        prototypeStoreJson: String,
    ): RuntimeArtifacts {
        val classRoot = JSONObject(classLabelsJson)
        val preprocessingRoot = JSONObject(preprocessingConfigJson)
        val thresholdRoot = JSONObject(thresholdDefaultsJson)
        val prototypeRoot = JSONObject(prototypeStoreJson)

        return RuntimeArtifacts(
            classes = parseClasses(classRoot),
            preprocessing = parsePreprocessing(preprocessingRoot),
            thresholds = parseThresholds(thresholdRoot),
            prototypes = parsePrototypeStore(prototypeRoot),
        )
    }

    private fun parseClasses(root: JSONObject): List<ClassDescriptor> {
        val classes = root.getJSONArray("classes")
        return buildList(classes.length()) {
            for (index in 0 until classes.length()) {
                val entry = classes.getJSONObject(index)
                add(
                    ClassDescriptor(
                        id = entry.getInt("id"),
                        slug = entry.getString("slug"),
                        urgent = entry.optBoolean("urgent", false),
                        suppressed = entry.optBoolean("suppressed", false),
                    ),
                )
            }
        }
    }

    private fun parsePreprocessing(root: JSONObject): PreprocessingConfig {
        val audio = root.getJSONObject("audio")
        val stft = root.getJSONObject("stft")
        val mel = root.getJSONObject("mel")
        val normalization = root.getJSONObject("normalization")
        val embedding = root.getJSONObject("embedding_contract")
        return PreprocessingConfig(
            sampleRateHz = audio.getInt("sample_rate_hz"),
            channels = audio.getInt("channels"),
            windowDurationMs = audio.getInt("window_duration_ms"),
            hopDurationMs = audio.getInt("hop_duration_ms"),
            tinyCnnEmbeddingDim = embedding.getInt("tiny_cnn_dim"),
            yamnetEmbeddingDim = embedding.getInt("yamnet_dim"),
            stft =
                StftConfig(
                    nFft = stft.getInt("n_fft"),
                    hopLength = stft.getInt("hop_length"),
                    winLength = stft.getInt("win_length"),
                    window = stft.getString("window"),
                    center = stft.getBoolean("center"),
                ),
            mel =
                MelConfig(
                    nMels = mel.getInt("n_mels"),
                    fMinHz = mel.getDouble("f_min_hz").toFloat(),
                    fMaxHz = mel.getDouble("f_max_hz").toFloat(),
                    power = mel.getDouble("power").toFloat(),
                    logOffset = mel.getDouble("log_offset").toFloat(),
                ),
            normalization =
                NormalizationConfig(
                    type = normalization.getString("type"),
                    clipMinDb = normalization.getDouble("clip_min_db").toFloat(),
                    clipMaxDb = normalization.getDouble("clip_max_db").toFloat(),
                ),
        )
    }

    private fun parseThresholds(root: JSONObject): ThresholdConfig {
        val confidence = root.getJSONObject("confidence_tiers")
        val modesRoot = root.getJSONObject("modes")
        val modeEntries = mutableMapOf<String, ModeThresholds>()
        val keys = modesRoot.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val mode = modesRoot.getJSONObject(key)
            modeEntries[key] =
                ModeThresholds(
                    distanceThresholdDelta = mode.getDouble("distance_threshold_delta").toFloat(),
                    loudnessFloorDeltaDb = mode.getDouble("loudness_floor_delta_db").toFloat(),
                    suppressedClasses = jsonArrayToStringSet(mode.optJSONArray("suppressed_classes")),
                )
        }

        return ThresholdConfig(
            distanceThreshold = root.getDouble("distance_threshold").toFloat(),
            loudnessFloorDbfs = root.getDouble("loudness_floor_dbfs").toFloat(),
            unknownRatioThreshold = root.getDouble("unknown_ratio_threshold").toFloat(),
            confidenceTiers =
                ConfidenceTiers(
                    high = confidence.getDouble("high").toFloat(),
                    medium = confidence.getDouble("medium").toFloat(),
                    low = confidence.getDouble("low").toFloat(),
                ),
            modes = modeEntries,
        )
    }

    private fun parsePrototypeStore(root: JSONObject): Map<String, FloatArray> {
        val prototypes = mutableMapOf<String, FloatArray>()
        val keys = root.keys()
        while (keys.hasNext()) {
            val label = keys.next()
            val values = root.getJSONArray(label)
            prototypes[label] = jsonArrayToFloatArray(values)
        }
        return prototypes
    }

    private fun jsonArrayToFloatArray(values: JSONArray): FloatArray {
        val output = FloatArray(values.length())
        for (index in 0 until values.length()) {
            output[index] = values.getDouble(index).toFloat()
        }
        return output
    }

    private fun jsonArrayToStringSet(values: JSONArray?): Set<String> {
        if (values == null) {
            return emptySet()
        }
        val output = mutableSetOf<String>()
        for (index in 0 until values.length()) {
            output.add(values.getString(index))
        }
        return output
    }
}
