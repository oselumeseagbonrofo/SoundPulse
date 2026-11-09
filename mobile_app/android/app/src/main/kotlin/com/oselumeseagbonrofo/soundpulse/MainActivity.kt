package com.oselumeseagbonrofo.soundpulse

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
import com.soundpulse.runtime.*
import android.os.Bundle
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private var methodChannel: MethodChannel? = null
    private var activeCriticalAlertSound: String? = null

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
    private var pendingStartResult: MethodChannel.Result? = null
    private var pendingInitResult: MethodChannel.Result? = null

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

            // Complete any pending start or initialization results once engine is ready
            val engine = s?.getRuntimeEngine()
            if (engine != null) {
                pendingStartResult?.success(null)
                pendingInitResult?.success(null)
            } else {
                pendingStartResult?.error("runtime_not_initialized", "Engine failed to initialize.", null)
                pendingInitResult?.error("runtime_not_initialized", "Engine failed to initialize.", null)
            }
            pendingStartResult = null
            pendingInitResult = null
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        handleCriticalAlertIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleCriticalAlertIntent(intent)
    }

    private fun handleCriticalAlertIntent(intent: Intent?) {
        if (intent == null) return
        if (intent.getBooleanExtra("critical_alert_triggered", false)) {
            val soundLabel = intent.getStringExtra("alert_sound_label") ?: "Urgent Sound"
            activeCriticalAlertSound = soundLabel
            
            // Wake screen and show over lock screen
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                setShowWhenLocked(true)
                setTurnScreenOn(true)
            } else {
                @Suppress("DEPRECATION")
                window.addFlags(
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                    WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                    WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
                )
            }

            // Send to Flutter UI immediately if method channel is ready
            mainHandler.post {
                methodChannel?.invokeMethod("onCriticalAlertReceived", mapOf("sound" to soundLabel))
            }
        }
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        createNotificationChannel()

        val channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, methodChannelName)
        methodChannel = channel
        channel.setMethodCallHandler { call, result ->
                when (call.method) {
                    "getActiveCriticalAlert" -> {
                        result.success(activeCriticalAlertSound)
                    }

                    "clearActiveCriticalAlert" -> {
                        activeCriticalAlertSound = null
                        result.success(null)
                    }

                    "startMonitoring" -> {
                        val activeModeStr = call.argument<String>("mode") ?: "general"
                        val backbone = call.argument<String>("backbone") ?: "yamnet"
                        val modelPath = call.argument<String>("modelPath")
                        val classLabelsPath = call.argument<String>("classLabelsPath")
                        val preprocessingConfigPath = call.argument<String>("preprocessingConfigPath")
                        val thresholdDefaultsPath = call.argument<String>("thresholdDefaultsPath")
                        val prototypeStorePath = call.argument<String>("prototypeStorePath")

                        startMonitoring(
                            backbone,
                            modelPath,
                            classLabelsPath,
                            preprocessingConfigPath,
                            thresholdDefaultsPath,
                            prototypeStorePath,
                            activeModeStr,
                            result
                        )
                    }

                    "initializeRuntime" -> {
                        val backbone = call.argument<String>("backbone") ?: "yamnet"
                        val modelPath = call.argument<String>("modelPath")
                        val classLabelsPath = call.argument<String>("classLabelsPath")
                        val preprocessingConfigPath = call.argument<String>("preprocessingConfigPath")
                        val thresholdDefaultsPath = call.argument<String>("thresholdDefaultsPath")
                        val prototypeStorePath = call.argument<String>("prototypeStorePath")

                        initializeRuntime(
                            backbone,
                            modelPath,
                            classLabelsPath,
                            preprocessingConfigPath,
                            thresholdDefaultsPath,
                            prototypeStorePath,
                            result
                        )
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
                        val pattern = call.argument<String>("pattern") ?: "single"
                        vibrate(pattern)
                        result.success(null)
                    }

                    "triggerCameraFlash" -> {
                        val pattern = call.argument<String>("pattern") ?: "single"
                        triggerCameraFlash(pattern)
                        result.success(null)
                    }

                    "hasCameraPermission" -> {
                        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                        result.success(granted)
                    }

                    "requestCameraPermission" -> {
                        requestPermissions(arrayOf(Manifest.permission.CAMERA), 102)
                        result.success(true)
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

                    "getBatteryLevel" -> {
                        result.success(getBatteryLevel())
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
        mode: String,
        result: MethodChannel.Result
    ) {
        if (modelPath == null || classLabelsPath == null || preprocessingConfigPath == null ||
            thresholdDefaultsPath == null || prototypeStorePath == null) {
            result.error("invalid_arguments", "Missing file paths for engine initialization.", null)
            return
        }

        if (!hasRecordAudioPermission()) {
            result.error(
                "record_audio_permission_denied",
                "RECORD_AUDIO permission is required to start monitoring.",
                null
            )
            return
        }

        val s = soundService
        if (s != null && s.getRuntimeEngine() != null) {
            result.success(null)
            return
        }

        pendingStartResult = result

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

        val bound = bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        if (!bound) {
            result.error("service_bind_failed", "Failed to bind to SoundMonitoringService.", null)
            pendingStartResult = null
        }
    }

    private fun initializeRuntime(
        backbone: String,
        modelPath: String?,
        classLabelsPath: String?,
        preprocessingConfigPath: String?,
        thresholdDefaultsPath: String?,
        prototypeStorePath: String?,
        result: MethodChannel.Result
    ) {
        if (modelPath == null || classLabelsPath == null || preprocessingConfigPath == null ||
            thresholdDefaultsPath == null || prototypeStorePath == null) {
            result.error("invalid_arguments", "Missing file paths for engine initialization.", null)
            return
        }

        val s = soundService
        if (s != null && s.getRuntimeEngine() != null) {
            result.success(null)
            return
        }

        pendingInitResult = result

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

        val bound = bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        if (!bound) {
            result.error("service_bind_failed", "Failed to bind to SoundMonitoringService.", null)
            pendingInitResult = null
        }
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

    private fun vibrate(pattern: String) {
        val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        if (vibrator == null || !vibrator.hasVibrator()) return

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val effect = when (pattern) {
                    "double" -> VibrationEffect.createWaveform(longArrayOf(0, 300, 200, 300), intArrayOf(0, VibrationEffect.DEFAULT_AMPLITUDE, 0, VibrationEffect.DEFAULT_AMPLITUDE), -1)
                    "triple" -> VibrationEffect.createWaveform(longArrayOf(0, 200, 100, 200, 100, 200), intArrayOf(0, VibrationEffect.DEFAULT_AMPLITUDE, 0, VibrationEffect.DEFAULT_AMPLITUDE, 0, VibrationEffect.DEFAULT_AMPLITUDE), -1)
                    "long" -> VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE)
                    else -> VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE)
                }
                vibrator.vibrate(effect)
            } else {
                @Suppress("DEPRECATION")
                when (pattern) {
                    "double" -> vibrator.vibrate(longArrayOf(0, 300, 200, 300), -1)
                    "triple" -> vibrator.vibrate(longArrayOf(0, 200, 100, 200, 100, 200), -1)
                    "long" -> vibrator.vibrate(1000)
                    else -> vibrator.vibrate(500)
                }
            }
        } catch (e: Exception) {
            // Ignore vibration errors
        }
    }

    private fun triggerCameraFlash(pattern: String) {
        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(Manifest.permission.CAMERA), 102)
                return
            }
            val cameraManager = getSystemService(Context.CAMERA_SERVICE) as? CameraManager ?: return
            val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                val characteristics = cameraManager.getCameraCharacteristics(id)
                characteristics.get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
            } ?: return

            val pulseCount = when (pattern) {
                "double" -> 2
                "strobe" -> 5
                "none" -> 0
                else -> 1
            }

            if (pulseCount > 0) {
                Thread {
                    try {
                        for (i in 0 until pulseCount) {
                            cameraManager.setTorchMode(cameraId, true)
                            Thread.sleep(100)
                            cameraManager.setTorchMode(cameraId, false)
                            Thread.sleep(100)
                        }
                    } catch (e: Exception) {
                        try {
                            cameraManager.setTorchMode(cameraId, false)
                        } catch (ex: Exception) {}
                    }
                }.start()
            }
        } catch (e: Exception) {
            // Ignore camera flash errors
        }
    }

    private fun getBatteryLevel(): Int {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            val bm = getSystemService(Context.BATTERY_SERVICE) as? android.os.BatteryManager
            return bm?.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
        } else {
            val intent = registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val level = intent?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = intent?.getIntExtra(android.os.BatteryManager.EXTRA_SCALE, -1) ?: -1
            return if (level >= 0 && scale > 0) (level * 100) / scale else -1
        }
    }


}
