package com.example.soundpulse_mobile

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.VibrationEffect
import android.os.Vibrator
import android.provider.Settings
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.soundpulse.runtime.InferenceResult
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val methodChannelName = "soundpulse.runtime/methods"
    private val eventChannelName = "soundpulse.runtime/events"
    private val mainHandler = Handler(Looper.getMainLooper())
    private val channelId = "soundpulse_alerts"
    private val channelName = "Sound Detection Alerts"
    private val channelDescription = "Notifications for detected sounds"
    private var notificationIdCounter = 0

    private var soundService: SoundMonitoringService? = null
    private var isServiceBound = false
    private var eventSink: EventChannel.EventSink? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as? SoundMonitoringService.SoundMonitoringBinder
            val s = binder?.getService()
            soundService = s
            isServiceBound = true

            // Set the listener to receive results
            s?.registerListener(object : SoundMonitoringService.Listener {
                override fun onInferenceResult(result: InferenceResult, frameSamples: FloatArray) {
                    dispatchResults(listOf(result), frameSamples)
                }
                override fun onError(code: String, message: String) {
                    dispatchMonitoringError(code, message)
                }
            })
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            soundService?.unregisterListener()
            soundService = null
            isServiceBound = false
        }
    }

    override fun onStart() {
        super.onStart()
        val prefs = getSharedPreferences("soundpulse", Context.MODE_PRIVATE)
        val active = prefs.getBoolean("monitoring_active", false)
        if (active) {
            val intent = Intent(this, SoundMonitoringService::class.java)
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        }
    }

    override fun onStop() {
        if (isServiceBound) {
            soundService?.unregisterListener()
            unbindService(serviceConnection)
            isServiceBound = false
            soundService = null
        }
        super.onStop()
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        createNotificationChannel()

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, methodChannelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "startMonitoring" -> {
                        val activeModeStr = call.argument<String>("mode") ?: "general"
                        val backbone = call.argument<String>("backbone") ?: "tiny_cnn"
                        val modelPath = call.argument<String>("modelPath")
                        val classLabelsPath = call.argument<String>("classLabelsPath")
                        val preprocessingConfigPath = call.argument<String>("preprocessingConfigPath")
                        val thresholdDefaultsPath = call.argument<String>("thresholdDefaultsPath")
                        val prototypeStorePath = call.argument<String>("prototypeStorePath")

                        val startError = startMonitoring(
                            backbone,
                            modelPath,
                            classLabelsPath,
                            preprocessingConfigPath,
                            thresholdDefaultsPath,
                            prototypeStorePath,
                            activeModeStr
                        )

                        if (startError == null) {
                            result.success(null)
                        } else {
                            result.error(startError.code, startError.message, null)
                        }
                    }

                    "initializeRuntime" -> {
                        val backbone = call.argument<String>("backbone") ?: "tiny_cnn"
                        val modelPath = call.argument<String>("modelPath")
                        val classLabelsPath = call.argument<String>("classLabelsPath")
                        val preprocessingConfigPath = call.argument<String>("preprocessingConfigPath")
                        val thresholdDefaultsPath = call.argument<String>("thresholdDefaultsPath")
                        val prototypeStorePath = call.argument<String>("prototypeStorePath")

                        val initError = initializeRuntime(
                            backbone,
                            modelPath,
                            classLabelsPath,
                            preprocessingConfigPath,
                            thresholdDefaultsPath,
                            prototypeStorePath
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
                        val activeModeStr = call.argument<String>("mode") ?: "general"
                        soundService?.updateMode(activeModeStr)
                        result.success(null)
                    }

                    "setContextAware" -> {
                        result.success(null)
                    }

                    "pushAudioPcm" -> {
                        val rawSamples = call.argument<List<Number>>("samples").orEmpty()
                        val samples = FloatArray(rawSamples.size) { index -> rawSamples[index].toFloat() }
                        val engine = soundService?.getRuntimeEngine()
                        if (engine == null) {
                            result.error("runtime_not_initialized", "Initialize runtime before pushing audio.", null)
                        } else {
                            val results = engine.pushAudioPcm(samples)
                            dispatchResults(results, samples)
                            result.success(results.map { item -> toPayload(item, samples) })
                        }
                    }

                    "extractEmbeddingFromPcm" -> {
                        val rawSamples = call.argument<List<Number>>("samples").orEmpty()
                        val samples = FloatArray(rawSamples.size) { index -> rawSamples[index].toFloat() }
                        val engine = soundService?.getRuntimeEngine()
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
                        val engine = soundService?.getRuntimeEngine()
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
                        val engine = soundService?.getRuntimeEngine()
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

                    "isMonitoringActive" -> {
                        val prefs = getSharedPreferences("soundpulse", Context.MODE_PRIVATE)
                        result.success(prefs.getBoolean("monitoring_active", false))
                    }

                    "isBatteryOptimizationIgnored" -> {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
                            result.success(pm.isIgnoringBatteryOptimizations(packageName))
                        } else {
                            result.success(true)
                        }
                    }

                    "requestIgnoreBatteryOptimizations" -> {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                                data = android.net.Uri.parse("package:$packageName")
                            }
                            startActivity(intent)
                            result.success(true)
                        } else {
                            result.success(true)
                        }
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
        if (isServiceBound) {
            soundService?.unregisterListener()
            unbindService(serviceConnection)
            isServiceBound = false
            soundService = null
        }
        super.onDestroy()
    }

    private fun startMonitoring(
        backbone: String,
        modelPath: String?,
        classLabelsPath: String?,
        preprocessingConfigPath: String?,
        thresholdDefaultsPath: String?,
        prototypeStorePath: String?,
        mode: String
    ): StartMonitoringError? {
        if (modelPath == null || classLabelsPath == null || preprocessingConfigPath == null ||
            thresholdDefaultsPath == null || prototypeStorePath == null) {
            return StartMonitoringError("invalid_arguments", "Missing file paths for engine initialization.")
        }

        if (!hasRecordAudioPermission()) {
            return StartMonitoringError(
                "record_audio_permission_denied",
                "RECORD_AUDIO permission is required to start monitoring.",
            )
        }

        val intent = Intent(this, SoundMonitoringService::class.java).apply {
            action = "START_MONITORING"
            putExtra("backbone", backbone)
            putExtra("mode", mode)
            putExtra("modelPath", modelPath)
            putExtra("classLabelsPath", classLabelsPath)
            putExtra("preprocessingConfigPath", preprocessingConfigPath)
            putExtra("thresholdDefaultsPath", thresholdDefaultsPath)
            putExtra("prototypeStorePath", prototypeStorePath)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }

        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        return null
    }

    private fun initializeRuntime(
        backbone: String,
        modelPath: String?,
        classLabelsPath: String?,
        preprocessingConfigPath: String?,
        thresholdDefaultsPath: String?,
        prototypeStorePath: String?
    ): StartMonitoringError? {
        if (modelPath == null || classLabelsPath == null || preprocessingConfigPath == null ||
            thresholdDefaultsPath == null || prototypeStorePath == null) {
            return StartMonitoringError("invalid_arguments", "Missing file paths for engine initialization.")
        }

        val intent = Intent(this, SoundMonitoringService::class.java).apply {
            action = "INITIALIZE_RUNTIME"
            putExtra("backbone", backbone)
            putExtra("modelPath", modelPath)
            putExtra("classLabelsPath", classLabelsPath)
            putExtra("preprocessingConfigPath", preprocessingConfigPath)
            putExtra("thresholdDefaultsPath", thresholdDefaultsPath)
            putExtra("prototypeStorePath", prototypeStorePath)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }

        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        return null
    }

    private fun stopMonitoring() {
        if (isServiceBound) {
            soundService?.unregisterListener()
            unbindService(serviceConnection)
            isServiceBound = false
            soundService = null
        }
        val intent = Intent(this, SoundMonitoringService::class.java).apply {
            action = "STOP_MONITORING"
        }
        startService(intent)
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
