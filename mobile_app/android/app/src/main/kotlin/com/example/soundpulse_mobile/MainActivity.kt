package com.example.soundpulse_mobile

import com.soundpulse.runtime.InferenceResult
import com.soundpulse.runtime.Mode
import com.soundpulse.runtime.RuntimeEngine
import com.soundpulse.runtime.StubRuntimeEngine
import android.os.Handler
import android.os.Looper
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.embedding.android.FlutterActivity
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import java.util.Timer
import java.util.TimerTask
import kotlin.math.PI
import kotlin.math.sin

class MainActivity : FlutterActivity() {
    private val methodChannelName = "soundpulse.runtime/methods"
    private val eventChannelName = "soundpulse.runtime/events"
    private val mainHandler = Handler(Looper.getMainLooper())

    private val runtimeEngine: RuntimeEngine = StubRuntimeEngine()

    private var eventSink: EventChannel.EventSink? = null
    private var mode: Mode = Mode.GENERAL
    private var contextAware: Boolean = false
    private var monitoringTimer: Timer? = null
    private var syntheticFrameIndex: Int = 0

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, methodChannelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "startMonitoring" -> {
                        mode = parseMode(call.argument<String>("mode"))
                        startMonitoring()
                        result.success(null)
                    }

                    "stopMonitoring" -> {
                        stopMonitoring()
                        result.success(null)
                    }

                    "updateMode" -> {
                        mode = parseMode(call.argument<String>("mode"))
                        runtimeEngine.updateMode(mode)
                        result.success(null)
                    }

                    "setContextAware" -> {
                        contextAware = call.argument<Boolean>("enabled") ?: false
                        result.success(null)
                    }

                    "pushAudioPcm" -> {
                        val rawSamples = call.argument<List<Number>>("samples").orEmpty()
                        val samples = FloatArray(rawSamples.size) { index -> rawSamples[index].toFloat() }
                        val results = runtimeEngine.pushAudioPcm(samples)
                        dispatchResults(results, samples)
                        result.success(results.map { item -> toPayload(item, samples) })
                    }

                    else -> result.notImplemented()
                }
            }

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, eventChannelName)
            .setStreamHandler(
                object : EventChannel.StreamHandler {
                    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                        eventSink = events
                    }

                    override fun onCancel(arguments: Any?) {
                        eventSink = null
                    }
                },
            )
    }

    override fun onDestroy() {
        stopMonitoring()
        super.onDestroy()
    }

    private fun parseMode(modeArg: String?): Mode {
        return when (modeArg?.trim()?.lowercase()) {
            "sleep" -> Mode.SLEEP
            "focus" -> Mode.FOCUS
            else -> Mode.GENERAL
        }
    }

    private fun startMonitoring() {
        stopMonitoring()
        syntheticFrameIndex = 0
        runtimeEngine.start(mode)

        monitoringTimer =
            Timer("soundpulse-runtime-feed", false).apply {
                scheduleAtFixedRate(
                    object : TimerTask() {
                        override fun run() {
                            val chunk = buildSyntheticFrame()
                            val results = runtimeEngine.pushAudioPcm(chunk)
                            dispatchResults(results, chunk)
                        }
                    },
                    0L,
                    500L,
                )
            }
    }

    private fun stopMonitoring() {
        monitoringTimer?.cancel()
        monitoringTimer = null
        runtimeEngine.stop()
    }

    private fun buildSyntheticFrame(): FloatArray {
        val frameLength = 1024
        val frequencyHz = if (contextAware && mode == Mode.SLEEP) 180.0 else 420.0
        val amplitude =
            when (mode) {
                Mode.SLEEP -> 0.15f
                Mode.FOCUS -> 0.28f
                Mode.GENERAL -> 0.45f
            }
        val sampleRateHz = 16_000.0
        val phaseStart = syntheticFrameIndex * frameLength
        val chunk =
            FloatArray(frameLength) { index ->
                val position = phaseStart + index
                val phase = (2.0 * PI * frequencyHz * position) / sampleRateHz
                (sin(phase) * amplitude).toFloat()
            }
        syntheticFrameIndex += 1
        return chunk
    }

    private fun dispatchResults(results: List<InferenceResult>, frameSamples: FloatArray?) {
        if (results.isEmpty()) {
            return
        }
        mainHandler.post {
            val sink = eventSink ?: return@post
            results.forEach { result ->
                sink.success(toPayload(result, frameSamples))
            }
        }
    }

    private fun toPayload(result: InferenceResult, frameSamples: FloatArray? = null): Map<String, Any?> {
        return mapOf(
            "label" to result.label,
            "distance" to result.distance.toDouble(),
            "confidence" to result.confidence.toDouble(),
            "confidence_tier" to result.confidenceTier,
            "waveform_samples" to (frameSamples?.map { sample -> sample.toDouble() }.orEmpty()),
            "top_candidates" to
                result.topCandidates.map { candidate ->
                    mapOf(
                        "label" to candidate.label,
                        "distance" to candidate.distance.toDouble(),
                        "confidence" to candidate.confidence.toDouble(),
                    )
                },
            "flags" to
                mapOf(
                    "is_unknown" to result.flags.isUnknown,
                    "is_background" to result.flags.isBackground,
                    "speech_pitch_required" to result.flags.speechPitchRequired,
                ),
            "speech_pitch" to
                result.speechPitch?.let {
                    mapOf(
                        "f0_hz" to it.f0Hz.toDouble(),
                        "bucket" to it.bucket,
                    )
                },
        )
    }
}
