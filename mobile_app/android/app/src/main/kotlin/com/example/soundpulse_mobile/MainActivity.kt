package com.example.soundpulse_mobile

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.soundpulse.runtime.InferenceResult
import com.soundpulse.runtime.Mode
import com.soundpulse.runtime.RuntimeEngine
import com.soundpulse.runtime.LiteRtRuntimeFactory
import com.soundpulse.runtime.RuntimeArtifactLoader
import com.soundpulse.runtime.LiteRtModelLoader
import com.soundpulse.runtime.RuntimeBackbone
import com.soundpulse.runtime.LiteRtEmbeddingExtractor
import com.soundpulse.runtime.TinyCnnEmbeddingExtractor
import android.os.Handler
import android.os.Looper
import androidx.core.content.ContextCompat
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.embedding.android.FlutterActivity
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import android.content.Context
import android.os.Vibrator
import android.os.VibrationEffect
import android.os.Build
import android.hardware.camera2.CameraManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import androidx.core.app.NotificationCompat

class MainActivity : FlutterActivity() {
    private val methodChannelName = "soundpulse.runtime/methods"
    private val eventChannelName = "soundpulse.runtime/events"
    private val mainHandler = Handler(Looper.getMainLooper())
    private val monitoringLock = Any()
    private val sampleRateHz = 16_000
    private val captureChunkSamples = 1024
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    private var runtimeEngine: RuntimeEngine? = null

    private var eventSink: EventChannel.EventSink? = null
    private var mode: Mode = Mode.GENERAL
    private var contextAware: Boolean = false
    private var audioRecord: AudioRecord? = null
    private var captureThread: Thread? = null
    @Volatile private var captureRunning: Boolean = false

    private val channelId = "soundpulse_alerts"
    private val channelName = "Sound Detection Alerts"
    private val channelDescription = "Notifications for detected sounds"
    private var notificationIdCounter = 0

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        createNotificationChannel()

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, methodChannelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "startMonitoring" -> {
                        mode = parseMode(call.argument<String>("mode"))
                        val backbone = call.argument<String>("backbone") ?: "tiny_cnn"
                        val modelPath = call.argument<String>("modelPath")
                        val classLabelsPath = call.argument<String>("classLabelsPath")
                        val preprocessingConfigPath = call.argument<String>("preprocessingConfigPath")
                        val thresholdDefaultsPath = call.argument<String>("thresholdDefaultsPath")
                        val prototypeStorePath = call.argument<String>("prototypeStorePath")

                        val startError = initializeRuntime(
                            backbone,
                            modelPath,
                            classLabelsPath,
                            preprocessingConfigPath,
                            thresholdDefaultsPath,
                            prototypeStorePath
                        ) ?: startMonitoring()

                        if (startError == null) {
                            result.success(null)
                        } else {
                            result.error(startError.code, startError.message, null)
                        }
                    }

                    "initializeRuntime" -> {
                        val initError = initializeRuntime(
                            call.argument<String>("backbone") ?: "tiny_cnn",
                            call.argument<String>("modelPath"),
                            call.argument<String>("classLabelsPath"),
                            call.argument<String>("preprocessingConfigPath"),
                            call.argument<String>("thresholdDefaultsPath"),
                            call.argument<String>("prototypeStorePath")
                        )
                        if (initError == null) {
                            result.success(null)
                        } else {
                            result.error(initError.code, initError.message, null)
                        }
                    }

                    "stopMonitoring" -> {
                        stopMonitoring()
                        result.success(null)
                    }

                    "updateMode" -> {
                        mode = parseMode(call.argument<String>("mode"))
                        runtimeEngine?.updateMode(mode)
                        result.success(null)
                    }

                    "setContextAware" -> {
                        contextAware = call.argument<Boolean>("enabled") ?: false
                        result.success(null)
                    }

                    "pushAudioPcm" -> {
                        val rawSamples = call.argument<List<Number>>("samples").orEmpty()
                        val samples = FloatArray(rawSamples.size) { index -> rawSamples[index].toFloat() }
                        val results = runtimeEngine?.pushAudioPcm(samples).orEmpty()
                        dispatchResults(results, samples)
                        result.success(results.map { item -> toPayload(item, samples) })
                    }

                    "extractEmbeddingFromPcm" -> {
                        val rawSamples = call.argument<List<Number>>("samples").orEmpty()
                        val samples = FloatArray(rawSamples.size) { index -> rawSamples[index].toFloat() }
                        val engine = runtimeEngine
                        if (engine == null) {
                            result.error("runtime_not_initialized", "Initialize runtime before extracting embeddings.", null)
                        } else {
                            try {
                                result.success(engine.extractEmbedding(samples).map { value -> value.toDouble() })
                            } catch (e: Exception) {
                                result.error("embedding_failed", e.message ?: "Embedding extraction failed.", null)
                            }
                        }
                    }

                    "updatePrototype" -> {
                        val label = call.argument<String>("label").orEmpty()
                        val rawEmbedding = call.argument<List<Number>>("embedding").orEmpty()
                        val engine = runtimeEngine
                        if (engine == null) {
                            result.error("runtime_not_initialized", "Initialize runtime before updating prototypes.", null)
                        } else if (label.isBlank() || rawEmbedding.isEmpty()) {
                            result.error("invalid_arguments", "Label and embedding are required.", null)
                        } else {
                            try {
                                engine.updatePrototype(label, FloatArray(rawEmbedding.size) { index -> rawEmbedding[index].toFloat() })
                                result.success(null)
                            } catch (e: Exception) {
                                result.error("prototype_update_failed", e.message ?: "Prototype update failed.", null)
                            }
                        }
                    }

                    "resetPrototype" -> {
                        val label = call.argument<String>("label").orEmpty()
                        val engine = runtimeEngine
                        if (engine == null) {
                            result.error("runtime_not_initialized", "Initialize runtime before resetting prototypes.", null)
                        } else if (label.isBlank()) {
                            result.error("invalid_arguments", "Label is required.", null)
                        } else {
                            try {
                                engine.resetPrototype(label)
                                result.success(null)
                            } catch (e: Exception) {
                                result.error("prototype_reset_failed", e.message ?: "Prototype reset failed.", null)
                            }
                        }
                    }

                    "showNotification" -> {
                        val title = call.argument<String>("title").orEmpty()
                        val body = call.argument<String>("body").orEmpty()
                        showNotification(title, body)
                        result.success(null)
                    }

                    "vibrate" -> {
                        vibrate()
                        result.success(null)
                    }

                    "triggerCameraFlash" -> {
                        triggerCameraFlash()
                        result.success(null)
                    }

                    "requestNotificationPermission" -> {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 101)
                        }
                        result.success(true)
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

    private fun initializeRuntime(
        backbone: String,
        modelPath: String?,
        classLabelsPath: String?,
        preprocessingConfigPath: String?,
        thresholdDefaultsPath: String?,
        prototypeStorePath: String?,
    ): StartMonitoringError? {
        if (modelPath == null || classLabelsPath == null || preprocessingConfigPath == null ||
            thresholdDefaultsPath == null || prototypeStorePath == null) {
            return StartMonitoringError("invalid_arguments", "Missing file paths for engine initialization.")
        }

        return try {
            val classFile = java.io.File(classLabelsPath)
            val preprocessingFile = java.io.File(preprocessingConfigPath)
            val thresholdFile = java.io.File(thresholdDefaultsPath)
            val prototypeFile = java.io.File(prototypeStorePath)
            val modelFile = java.io.File(modelPath)

            if (!classFile.exists() || !preprocessingFile.exists() || !thresholdFile.exists() || !prototypeFile.exists()) {
                StartMonitoringError("missing_files", "One or more configuration files do not exist.")
            } else if (!modelFile.exists()) {
                StartMonitoringError("missing_model", "TFLite model file does not exist.")
            } else {
                val artifacts = RuntimeArtifactLoader.fromFiles(
                    classFile,
                    preprocessingFile,
                    thresholdFile,
                    prototypeFile
                )
                val interpreter = LiteRtModelLoader.loadInterpreter(modelFile)
                val activeBackbone = if (backbone.lowercase() == "yamnet") {
                    RuntimeBackbone.YAMNET
                } else {
                    RuntimeBackbone.TINY_CNN
                }
                val extractor = if (activeBackbone == RuntimeBackbone.YAMNET) {
                    LiteRtEmbeddingExtractor(interpreter, artifacts.preprocessing.yamnetEmbeddingDim)
                } else {
                    TinyCnnEmbeddingExtractor(
                        interpreter,
                        artifacts.preprocessing,
                        artifacts.preprocessing.tinyCnnEmbeddingDim
                    )
                }
                runtimeEngine = LiteRtRuntimeFactory.createFromArtifacts(artifacts, extractor)
                null
            }
        } catch (e: Exception) {
            StartMonitoringError("engine_init_failed", "Failed to initialize ML engine: ${e.message}")
        }
    }

    private fun startMonitoring(): StartMonitoringError? {
        synchronized(monitoringLock) {
            stopMonitoringLocked()
            if (!hasRecordAudioPermission()) {
                return StartMonitoringError(
                    "record_audio_permission_denied",
                    "RECORD_AUDIO permission is required to start monitoring.",
                )
            }
            val minBufferSize = AudioRecord.getMinBufferSize(sampleRateHz, channelConfig, audioFormat)
            if (minBufferSize <= 0) {
                return StartMonitoringError(
                    "audio_record_init_failed",
                    "Unable to initialize microphone capture.",
                )
            }
            val bufferSizeInBytes = maxOf(minBufferSize, captureChunkSamples * 2)
            val recorder =
                AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRateHz,
                    channelConfig,
                    audioFormat,
                    bufferSizeInBytes,
                )
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                return StartMonitoringError(
                    "audio_record_init_failed",
                    "Microphone is not available for live monitoring.",
                )
            }
            try {
                recorder.startRecording()
            } catch (_: IllegalStateException) {
                recorder.release()
                return StartMonitoringError(
                    "audio_record_start_failed",
                    "Failed to start microphone capture.",
                )
            }

            runtimeEngine?.start(mode)
            captureRunning = true
            audioRecord = recorder
            val worker =
                Thread(
                    { captureLoop(recorder) },
                    "soundpulse-runtime-feed",
                ).apply { isDaemon = true }
            captureThread = worker
            worker.start()
            return null
        }
    }

    private fun stopMonitoring() {
        synchronized(monitoringLock) {
            stopMonitoringLocked()
        }
    }

    private fun stopMonitoringLocked() {
        captureRunning = false
        val recorder = audioRecord
        audioRecord = null
        val worker = captureThread
        captureThread = null

        if (recorder != null) {
            if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                try {
                    recorder.stop()
                } catch (_: IllegalStateException) {
                }
            }
            recorder.release()
        }
        worker?.interrupt()
        runtimeEngine?.stop()
    }

    private fun captureLoop(recorder: AudioRecord) {
        val readBuffer = ShortArray(captureChunkSamples)
        while (captureRunning) {
            val samplesRead = recorder.read(readBuffer, 0, readBuffer.size, AudioRecord.READ_BLOCKING)
            if (samplesRead <= 0) {
                if (!captureRunning) {
                    break
                }
                if (samplesRead == AudioRecord.ERROR_DEAD_OBJECT) {
                    dispatchMonitoringError("audio_record_error", "Microphone capture was interrupted.")
                    break
                }
                if (samplesRead == AudioRecord.ERROR_BAD_VALUE || samplesRead == AudioRecord.ERROR_INVALID_OPERATION) {
                    dispatchMonitoringError(
                        "audio_record_error",
                        "Microphone capture failed with code $samplesRead.",
                    )
                    break
                }
                continue
            }
            val frameSamples =
                FloatArray(samplesRead) { index ->
                    (readBuffer[index] / 32768.0f).coerceIn(-1.0f, 1.0f)
                }
            try {
                val results = runtimeEngine?.pushAudioPcm(frameSamples).orEmpty()
                dispatchResults(results, frameSamples)
            } catch (e: Exception) {
                dispatchMonitoringError("inference_error", e.message ?: "Inference execution failed.")
                break
            }
        }
        synchronized(monitoringLock) {
            if (audioRecord === recorder) {
                stopMonitoringLocked()
            }
        }
    }

    private fun hasRecordAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun dispatchMonitoringError(code: String, message: String) {
        mainHandler.post {
            eventSink?.error(code, message, null)
        }
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

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val importance = NotificationManager.IMPORTANCE_HIGH
            val channel = NotificationChannel(channelId, channelName, importance).apply {
                description = channelDescription
            }
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun showNotification(title: String, body: String) {
        try {
            val intent = Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val pendingIntent = PendingIntent.getActivity(
                this,
                0,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val builder = NotificationCompat.Builder(this, channelId)
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentTitle(title)
                .setContentText(body)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)

            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(notificationIdCounter++, builder.build())
        } catch (e: Exception) {
            // Ignore notification errors
        }
    }

    private fun vibrate() {
        try {
            val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            if (vibrator != null && vibrator.hasVibrator()) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    vibrator.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(500)
                }
            }
        } catch (e: Exception) {
            // Ignore vibration errors
        }
    }

    private fun triggerCameraFlash() {
        try {
            val cameraManager = getSystemService(Context.CAMERA_SERVICE) as? CameraManager
            if (cameraManager != null) {
                val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                    val characteristics = cameraManager.getCameraCharacteristics(id)
                    characteristics.get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
                }
                if (cameraId != null) {
                    Thread {
                        try {
                            for (i in 0 until 3) {
                                cameraManager.setTorchMode(cameraId, true)
                                Thread.sleep(100)
                                cameraManager.setTorchMode(cameraId, false)
                                Thread.sleep(100)
                            }
                        } catch (e: Exception) {
                            try {
                                cameraManager.setTorchMode(cameraId, false)
                            } catch (_: Exception) {}
                        }
                    }.start()
                }
            }
        } catch (e: Exception) {
            // Ignore camera flash errors
        }
    }

    private data class StartMonitoringError(
        val code: String,
        val message: String,
    )
}
