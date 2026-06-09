package com.oselumeseagbonrofo.soundpulse

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.camera2.CameraManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Binder
import android.os.Build
import androidx.core.content.ContextCompat
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import androidx.core.app.NotificationCompat
import com.soundpulse.runtime.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Date
import java.util.UUID

class SoundMonitoringService : Service() {

    private val binder = SoundMonitoringBinder()
    private val monitoringLock = Any()
    private val sampleRateHz = 16_000
    private val captureChunkSamples = 16_000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    private var runtimeEngine: RuntimeEngine? = null
    private var audioRecord: AudioRecord? = null
    private var captureThread: Thread? = null
    @Volatile private var captureRunning: Boolean = false

    private val serviceChannelId = "soundpulse_background_service"
    private val alertChannelId = "soundpulse_alerts"
    private val serviceNotificationId = 999
    private var notificationIdCounter = 1000

    private var activeMode: String = "general"
    private var backbone: String = "tiny_cnn"

    // Callback for MainActivity when bound
    interface Listener {
        fun onInferenceResult(result: InferenceResult, frameSamples: FloatArray)
        fun onError(code: String, message: String)
    }

    private var clientListener: Listener? = null

    // For background decision state
    private val recentEventsBuffer = mutableListOf<TemporalEventEntry>()
    private val lastAlertTimestamps = HashMap<String, Long>()
    private var activeNotifiedSound: String? = null
    private val predictionWindow = mutableListOf<String>()

    data class TemporalEventEntry(val slug: String, val timestamp: Long, val confidence: Float)

    inner class SoundMonitoringBinder : Binder() {
        fun getService(): SoundMonitoringService = this@SoundMonitoringService
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    fun registerListener(listener: Listener) {
        clientListener = listener
    }

    fun unregisterListener() {
        clientListener = null
    }

    // Methods called by MainActivity to control the engine directly
    fun updateMode(modeStr: String) {
        activeMode = modeStr
        val runtimeMode = parseMode(modeStr)
        runtimeEngine?.updateMode(runtimeMode)
        
        // Update persistent notification to show active mode
        updateForegroundNotification()
    }
    
    fun getRuntimeEngine(): RuntimeEngine? = runtimeEngine

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: "START_MONITORING"
        
        if (action == "STOP_MONITORING") {
            stopMonitoring()
            stopSelf()
            return START_NOT_STICKY
        }

        val startedFromBoot = intent?.getBooleanExtra("started_from_boot", false) ?: false
        val prefs = getSharedPreferences("soundpulse", Context.MODE_PRIVATE)

        if (startedFromBoot) {
            // Load arguments from SharedPreferences
            backbone = prefs.getString("backbone", "tiny_cnn") ?: "tiny_cnn"
            activeMode = prefs.getString("mode", "general") ?: "general"
            val modelPath = prefs.getString("modelPath", "") ?: ""
            val classLabelsPath = prefs.getString("classLabelsPath", "") ?: ""
            val preprocessingConfigPath = prefs.getString("preprocessingConfigPath", "") ?: ""
            val thresholdDefaultsPath = prefs.getString("thresholdDefaultsPath", "") ?: ""
            val prototypeStorePath = prefs.getString("prototypeStorePath", "") ?: ""

            if (modelPath.isNotEmpty()) {
                val initError = initializeEngine(
                    backbone,
                    modelPath,
                    classLabelsPath,
                    preprocessingConfigPath,
                    thresholdDefaultsPath,
                    prototypeStorePath
                )
                if (initError == null) {
                    startMonitoringInternal()
                }
            } else {
                stopSelf()
            }
        } else {
            // Extract arguments from Intent
            val b = intent?.getStringExtra("backbone") ?: "tiny_cnn"
            backbone = b
            val m = intent?.getStringExtra("mode") ?: "general"
            activeMode = m
            val modelPath = intent?.getStringExtra("modelPath") ?: ""
            val classLabelsPath = intent?.getStringExtra("classLabelsPath") ?: ""
            val preprocessingConfigPath = intent?.getStringExtra("preprocessingConfigPath") ?: ""
            val thresholdDefaultsPath = intent?.getStringExtra("thresholdDefaultsPath") ?: ""
            val prototypeStorePath = intent?.getStringExtra("prototypeStorePath") ?: ""

            // Save arguments to SharedPreferences for reboot recovery
            prefs.edit().apply {
                putString("backbone", backbone)
                putString("mode", activeMode)
                putString("modelPath", modelPath)
                putString("classLabelsPath", classLabelsPath)
                putString("preprocessingConfigPath", preprocessingConfigPath)
                putString("thresholdDefaultsPath", thresholdDefaultsPath)
                putString("prototypeStorePath", prototypeStorePath)
                putBoolean("monitoring_active", action == "START_MONITORING")
                apply()
            }

            val initError = initializeEngine(
                backbone,
                modelPath,
                classLabelsPath,
                preprocessingConfigPath,
                thresholdDefaultsPath,
                prototypeStorePath
            )
            if (initError == null) {
                startForeground(serviceNotificationId, buildForegroundNotification())
                if (action == "START_MONITORING") {
                    startAudioCapture()
                }
            }
        }

        return START_STICKY
    }

    private fun startMonitoringInternal() {
        startForeground(serviceNotificationId, buildForegroundNotification())
        startAudioCapture()
    }

    private fun parseMode(modeArg: String?): Mode {
        return when (modeArg?.trim()?.lowercase()) {
            "sleep" -> Mode.SLEEP
            "focus" -> Mode.FOCUS
            else -> Mode.GENERAL
        }
    }

    private fun initializeEngine(
        backbone: String,
        modelPath: String,
        classLabelsPath: String,
        preprocessingConfigPath: String,
        thresholdDefaultsPath: String,
        prototypeStorePath: String
    ): String? {
        return try {
            val classFile = File(classLabelsPath)
            val preprocessingFile = File(preprocessingConfigPath)
            val thresholdFile = File(thresholdDefaultsPath)
            val prototypeFile = File(prototypeStorePath)
            val modelFile = File(modelPath)

            if (!classFile.exists() || !preprocessingFile.exists() || !thresholdFile.exists() || !prototypeFile.exists()) {
                "missing_files"
            } else if (!modelFile.exists()) {
                "missing_model"
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
            "engine_init_failed: ${e.message}"
        }
    }

    private fun startAudioCapture() {
        synchronized(monitoringLock) {
            stopAudioCaptureLocked()
            val minBufferSize = AudioRecord.getMinBufferSize(sampleRateHz, channelConfig, audioFormat)
            if (minBufferSize <= 0) {
                clientListener?.onError("audio_record_init_failed", "Unable to initialize microphone capture.")
                return
            }
            val bufferSizeInBytes = maxOf(minBufferSize, captureChunkSamples * 2)
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRateHz,
                channelConfig,
                audioFormat,
                bufferSizeInBytes
            )
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                clientListener?.onError("audio_record_init_failed", "Microphone is not available.")
                return
            }
            try {
                recorder.startRecording()
            } catch (e: IllegalStateException) {
                recorder.release()
                clientListener?.onError("audio_record_start_failed", "Failed to start microphone capture.")
                return
            }

            runtimeEngine?.start(parseMode(activeMode))
            captureRunning = true
            audioRecord = recorder
            val worker = Thread({ captureLoop(recorder) }, "soundpulse-runtime-feed-service").apply {
                isDaemon = true
            }
            captureThread = worker
            worker.start()
        }
    }

    private fun stopMonitoring() {
        synchronized(monitoringLock) {
            stopAudioCaptureLocked()
        }
        val prefs = getSharedPreferences("soundpulse", Context.MODE_PRIVATE)
        prefs.edit().putBoolean("monitoring_active", false).apply()
        stopForeground(true)
    }

    private fun stopAudioCaptureLocked() {
        captureRunning = false
        val recorder = audioRecord
        audioRecord = null
        val worker = captureThread
        captureThread = null

        if (recorder != null) {
            if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                try {
                    recorder.stop()
                } catch (e: IllegalStateException) {}
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
                if (!captureRunning) break
                if (samplesRead == AudioRecord.ERROR_DEAD_OBJECT) {
                    clientListener?.onError("audio_record_error", "Microphone capture was interrupted.")
                    break
                }
                continue
            }
            val frameSamples = FloatArray(samplesRead) { index ->
                (readBuffer[index] / 32768.0f).coerceIn(-1.0f, 1.0f)
            }
            try {
                val results = runtimeEngine?.pushAudioPcm(frameSamples).orEmpty()
                
                // If we have a bound UI listener, forward results to it
                val listener = clientListener
                if (listener != null) {
                    Handler(Looper.getMainLooper()).post {
                        results.forEach { result ->
                            listener.onInferenceResult(result, frameSamples)
                        }
                    }
                } else {
                    // Running in background: evaluate alert decisions natively in Kotlin
                    results.forEach { result ->
                        evaluateBackgroundResult(result, frameSamples)
                    }
                }
            } catch (e: Exception) {
                clientListener?.onError("inference_error", e.message ?: "Inference failed.")
                break
            }
        }
    }

    private fun evaluateBackgroundResult(result: InferenceResult, frameSamples: FloatArray) {
        val label = if (result.flags.isBackground || result.flags.isUnknown) "background" else result.label
        predictionWindow.add(label)
        if (predictionWindow.size > 1) {
            predictionWindow.removeAt(0)
        }

        val frequencies = HashMap<String, Int>()
        for (l in predictionWindow) {
            frequencies[l] = (frequencies[l] ?: 0) + 1
        }

        val active = activeNotifiedSound
        if (active == null) {
            var candidateLabel: String? = null
            var maxCount = 0
            for ((key, count) in frequencies) {
                if (key != "background" && count > maxCount) {
                    maxCount = count
                    candidateLabel = key
                }
            }

            if (candidateLabel != null) {
                // Check settings overrides from JSON
                val overrides = readClassOverride(candidateLabel)
                val enabled = overrides?.optBoolean("enabled", true) ?: true
                if (!enabled) return

                activeNotifiedSound = candidateLabel

                // Decide alerts
                val decision = decideAlert(candidateLabel, result, overrides)
                if (!decision.suppressed) {
                    triggerBackgroundAlert(candidateLabel, decision)
                }

                val eventId = "${System.currentTimeMillis()}_${UUID.randomUUID().toString().take(4)}"
                val wavPath = persistAudioClip(eventId, frameSamples)
                logBackgroundEvent(eventId, candidateLabel, decision, wavPath)
            }
        } else {
            val count = frequencies[active] ?: 0
            if (count == 0) {
                activeNotifiedSound = null
            }
        }
    }

    private fun triggerBackgroundAlert(soundLabel: String, decision: AlertDecision) {
        val cleanLabel = formatSoundLabel(soundLabel)
        val title = "Sound Detected!"
        val body = "$cleanLabel (${(decision.confidence * 100).toInt()}% confidence)"

        // 1. Post notification
        when (decision.modality) {
            "text_only", "standard_vibration_text", "vibration_flash_text" -> {
                showNotification(title, body, false)
            }
            "critical_full_screen" -> {
                showNotification("CRITICAL ALERT: $cleanLabel", "SoundPulse detected $cleanLabel.", true)
            }
        }

        // 2. Trigger haptic/visual cues
        when (decision.modality) {
            "standard_vibration_text" -> {
                vibrate(decision.customVibration)
            }
            "vibration_flash_text" -> {
                vibrate(decision.customVibration)
                triggerCameraFlash(decision.customFlash)
            }
            "critical_full_screen" -> {
                vibrate("long")
                triggerCameraFlash("strobe")
            }
        }
    }

    // Decides alerts based on activeMode and class config overrides
    private fun decideAlert(slug: String, result: InferenceResult, overrides: JSONObject?): AlertDecision {
        val now = System.currentTimeMillis()

        // Defaults
        var priority = getDefaultPriority(slug)
        var modality = getDefaultModality(slug)
        var customVibration = overrides?.optString("custom_vibration", "single") ?: "single"
        var customFlash = overrides?.optString("custom_flash", "single") ?: "single"

        // Overrides
        if (overrides != null) {
            val priorityLevel = overrides.optString("priority_level", "")
            if (priorityLevel.isNotEmpty()) {
                priority = priorityLevel
            }
            val modesObj = overrides.optJSONObject("modes")
            if (modesObj != null && modesObj.has(activeMode)) {
                modality = modesObj.getString(activeMode)
            } else {
                val userAlertModality = overrides.optString("user_alert_modality", "")
                if (userAlertModality.isNotEmpty()) {
                    modality = userAlertModality
                }
            }
        }

        // Temporal Sequence Boost: knock_then_speech
        var finalConfidence = result.confidence
        var chainBoostApplied = false
        recentEventsBuffer.removeAll { now - it.timestamp > 15000 }

        if (slug == "speech") {
            val hasRecentKnock = recentEventsBuffer.any { it.slug == "door_knock" || it.slug == "doorbell" }
            if (hasRecentKnock) {
                finalConfidence = (result.confidence + 0.05f).coerceAtMost(1f)
                priority = "important"
                modality = "vibration_flash_text"
                chainBoostApplied = true
            }
        }

        recentEventsBuffer.add(TemporalEventEntry(slug, now, finalConfidence))

        // Gating Rules
        var suppressed = false
        var suppressionReason = ""

        val hasModeOverride = overrides?.optJSONObject("modes")?.has(activeMode) ?: false
        if (!hasModeOverride) {
            if (activeMode == "sleep") {
                if (priority == "useful" || priority == "low" || priority == "background") {
                    suppressed = true
                    modality = "silent_log"
                    suppressionReason = "Suppressed low-value $priority sound in Sleep Mode."
                } else if (priority == "important") {
                    modality = "vibration_flash_text"
                } else if (priority == "critical") {
                    modality = "critical_full_screen"
                }
            } else if (activeMode == "focus") {
                if (priority != "critical") {
                    suppressed = true
                    modality = "silent_log"
                    suppressionReason = "Suppressed non-critical sound in Focus Mode."
                } else {
                    modality = "critical_full_screen"
                }
            } else if (activeMode == "cooking") {
                val isCookingRelevant = slug == "fire_alarm" || slug == "clock_alarm" ||
                        slug == "door_knock" || slug == "doorbell" || slug == "water_running" || slug == "speech"
                if (!isCookingRelevant) {
                    suppressed = true
                    modality = "silent_log"
                    suppressionReason = "Suppressed non-cooking sound in Cooking Mode."
                }
            } else if (activeMode == "bathroom") {
                if (slug == "water_running") {
                    suppressed = true
                    modality = "silent_log"
                    suppressionReason = "Suppressed running water in Bathroom Mode."
                } else if (priority == "important") {
                    modality = "vibration_flash_text"
                }
            }
        }

        // Deduplication & Rate Limiting
        if (!suppressed && modality != "none" && modality != "silent_log") {
            val lastAlert = lastAlertTimestamps[slug]
            if (lastAlert != null) {
                val elapsed = now - lastAlert
                val minWindow = when (priority) {
                    "critical" -> 10000L
                    "important" -> 45000L
                    "useful" -> 120000L
                    else -> 300000L
                }
                if (elapsed < minWindow) {
                    suppressed = true
                    modality = "silent_log"
                    suppressionReason = "Rate-limited ($priority): Triggered within ${minWindow / 1000}s."
                }
            }
        }

        if (!suppressed && modality != "none" && modality != "silent_log") {
            lastAlertTimestamps[slug] = now
        }

        return AlertDecision(slug, priority, modality, finalConfidence, suppressed, suppressionReason, chainBoostApplied, customVibration, customFlash)
    }

    private class AlertDecision(
        val classSlug: String,
        val priority: String,
        val modality: String,
        val confidence: Float,
        val suppressed: Boolean,
        val suppressionReason: String,
        val chainBoostApplied: Boolean,
        val customVibration: String,
        val customFlash: String
    )

    private fun readClassOverride(slug: String): JSONObject? {
        return try {
            val file = File(flutterDocumentsDir(), "app_settings_sync.json")
            if (file.exists()) {
                val json = JSONObject(file.readText())
                val configs = json.optJSONObject("configs")
                configs?.optJSONObject(slug)
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun flutterDocumentsDir(): File {
        return applicationContext.getDir("flutter", Context.MODE_PRIVATE)
    }

    private fun getDefaultPriority(slug: String): String {
        return when (slug) {
            "fire_alarm", "clock_alarm" -> "critical"
            "door_knock", "doorbell", "door_wood_creaking", "crying_baby", "dog_barking", "speech" -> "important"
            "water_running" -> "useful"
            "background" -> "background"
            else -> "low"
        }
    }

    private fun getDefaultModality(slug: String): String {
        return when (slug) {
            "fire_alarm", "clock_alarm" -> "critical_full_screen"
            "door_knock", "doorbell", "door_wood_creaking", "crying_baby" -> "vibration_flash_text"
            "dog_barking", "water_running", "speech" -> "standard_vibration_text"
            "background" -> "none"
            else -> "silent_log"
        }
    }

    private fun formatSoundLabel(slug: String): String {
        return slug.split("_").joinToString(" ") { it.replaceFirstChar { char -> char.uppercase() } }
    }

    // Vibrations implementation
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
        } catch (e: Exception) {}
    }

    // Camera Flash Strobe implementation
    private fun triggerCameraFlash(pattern: String) {
        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
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
        } catch (e: Exception) {}
    }

    private fun persistAudioClip(id: String, samples: FloatArray): String {
        val detectionsDir = File(flutterDocumentsDir(), "detections")
        detectionsDir.mkdirs()
        val file = File(detectionsDir, "detection_$id.wav")
        
        try {
            FileOutputStream(file).use { out ->
                val wavHeader = getWavHeader(samples.size * 2)
                out.write(wavHeader)
                val buffer = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                for (sample in samples) {
                    val intVal = (sample * 32767.0f).toInt().coerceIn(-32768, 32767)
                    buffer.putShort(intVal.toShort())
                }
                out.write(buffer.array())
            }
        } catch (e: Exception) {}
        return file.absolutePath
    }

    private fun getWavHeader(pcmDataSize: Int): ByteArray {
        val header = ByteArray(44)
        val totalSize = 36 + pcmDataSize
        
        // RIFF header
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        
        // Chunk size
        header[4] = (totalSize and 0xff).toByte()
        header[5] = ((totalSize shr 8) and 0xff).toByte()
        header[6] = ((totalSize shr 16) and 0xff).toByte()
        header[7] = ((totalSize shr 24) and 0xff).toByte()
        
        // WAVEfmt 
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        
        // Subchunk1Size (16)
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        
        // AudioFormat (1 = PCM)
        header[20] = 1
        header[21] = 0
        
        // NumChannels (1)
        header[22] = 1
        header[23] = 0
        
        // SampleRate (16000)
        header[24] = (16000 and 0xff).toByte()
        header[25] = ((16000 shr 8) and 0xff).toByte()
        header[26] = ((16000 shr 16) and 0xff).toByte()
        header[27] = ((16000 shr 24) and 0xff).toByte()
        
        // ByteRate (16000 * 1 * 2) = 32000
        header[28] = (32000 and 0xff).toByte()
        header[29] = ((32000 shr 8) and 0xff).toByte()
        header[30] = ((32000 shr 16) and 0xff).toByte()
        header[31] = ((32000 shr 24) and 0xff).toByte()
        
        // BlockAlign (1 * 2 = 2)
        header[32] = 2
        header[33] = 0
        
        // BitsPerSample (16)
        header[34] = 16
        header[35] = 0
        
        // data chunk
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        
        // data size
        header[40] = (pcmDataSize and 0xff).toByte()
        header[41] = ((pcmDataSize shr 8) and 0xff).toByte()
        header[42] = ((pcmDataSize shr 16) and 0xff).toByte()
        header[43] = ((pcmDataSize shr 24) and 0xff).toByte()
        
        return header
    }

    private fun logBackgroundEvent(id: String, slug: String, decision: AlertDecision, wavPath: String) {
        try {
            val syncFile = File(flutterDocumentsDir(), "background_events_sync.json")
            val list = if (syncFile.exists()) {
                JSONArray(syncFile.readText())
            } else {
                JSONArray()
            }

            val format = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", java.util.Locale.US).apply {
                timeZone = java.util.TimeZone.getTimeZone("UTC")
            }

            val item = JSONObject().apply {
                put("event_id", id)
                put("timestamp", format.format(Date()))
                put("predicted_label", slug)
                put("confidence_tier", when {
                    decision.confidence >= 0.8f -> "high"
                    decision.confidence >= 0.5f -> "medium"
                    else -> "low"
                })
                put("confidence", decision.confidence.toDouble())
                put("is_unknown", false)
                put("is_background", false)
                put("audio_file_path", wavPath)
            }
            list.put(item)
            syncFile.writeText(list.toString())
        } catch (e: Exception) {}
    }

    // Notifications channel configuration
    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

            // Background Service channel (silent, low importance)
            val serviceChannel = NotificationChannel(
                serviceChannelId,
                "Background Listening Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Keeps SoundPulse alive in the background"
            }
            manager.createNotificationChannel(serviceChannel)

            // Alert notifications channel (high importance, noisy)
            val alertChannel = NotificationChannel(
                alertChannelId,
                "Sound Detection Alerts",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Heads-up visual and haptic alerts for detected sounds"
            }
            manager.createNotificationChannel(alertChannel)
        }
    }

    private fun buildForegroundNotification(): Notification {
        val intent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val cleanMode = activeMode.replaceFirstChar { it.uppercase() }

        return NotificationCompat.Builder(this, serviceChannelId)
            .setSmallIcon(applicationContext.applicationInfo.icon)
            .setContentTitle("SoundPulse is listening")
            .setContentText("Active Mode: $cleanMode. Continuous offline monitoring.")
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }

    private fun updateForegroundNotification() {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(serviceNotificationId, buildForegroundNotification())
    }

    private fun showNotification(title: String, body: String, isCritical: Boolean) {
        try {
            val intent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val pendingIntent = PendingIntent.getActivity(
                this,
                0,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val builder = NotificationCompat.Builder(this, alertChannelId)
                .setSmallIcon(applicationContext.applicationInfo.icon)
                .setContentTitle(title)
                .setContentText(body)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)

            if (isCritical) {
                builder.setPriority(NotificationCompat.PRIORITY_MAX)
                    .setCategory(NotificationCompat.CATEGORY_ALARM)
                    .setFullScreenIntent(pendingIntent, true)
            } else {
                builder.setPriority(NotificationCompat.PRIORITY_HIGH)
            }

            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.notify(notificationIdCounter++, builder.build())
        } catch (e: Exception) {}
    }

    override fun onDestroy() {
        stopMonitoring()
        super.onDestroy()
    }
}
