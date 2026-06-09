import "dart:async";
import "dart:io";
import "dart:math" as math;
import "dart:typed_data";

import "package:flutter/material.dart";
import "package:path_provider/path_provider.dart";
import "package:record/record.dart";

import "../models/custom_sound_sample.dart";
import "../models/inference_event.dart";
import "../models/monitor_history_entry.dart";
import "../runtime/runtime_bridge.dart";
import "../storage/local_storage_service.dart";
import "monitor_history_tab.dart";
import "monitor_live_feed_tab.dart";
import "monitor_record_tab.dart";
import "monitor_settings_tab.dart";
import "monitor_style.dart";
import "../services/context_manager.dart";

class MonitorPage extends StatefulWidget {
  const MonitorPage({super.key});

  @override
  State<MonitorPage> createState() => _MonitorPageState();
}

class _MonitorPageState extends State<MonitorPage> with WidgetsBindingObserver {
  final RuntimeBridge _runtimeBridge = RuntimeBridge();
  final LocalStorageService _storageService = LocalStorageService();
  final ContextManager _contextManager = ContextManager();
  final List<MonitorHistoryEntry> _history = <MonitorHistoryEntry>[];
  final List<CustomSoundSample> _recordedSounds = <CustomSoundSample>[];
  final AudioRecorder _audioRecorder = AudioRecorder();

  StreamSubscription<InferenceEvent>? _eventsSub;
  Timer? _historyCleanupTimer;
  int _detectionCounter = 0;

  bool _monitoring = false;
  bool _contextAware = false;
  String _mode = "general";
  String _backbone = "yamnet";
  String _personalisationLevel = "default";
  int _tabIndex = 0;
  int _previousTabIndex = 0;
  bool _vibrationEnabled = false;
  bool _cameraFlashEnabled = false;
  String? _runtimeReadyBackbone;

  static const List<PersonalizableSoundOption> _builtInOptions = [
    PersonalizableSoundOption("crying_baby", "Crying baby"),
    PersonalizableSoundOption("water_running", "Water running"),
    PersonalizableSoundOption("dog_barking", "Dog barking"),
    PersonalizableSoundOption("door_wood_creaking", "Door opening"),
    PersonalizableSoundOption("background", "Ambient background"),
  ];

  final List<PersonalizableSoundOption> _soundOptions = <PersonalizableSoundOption>[
    ..._builtInOptions
  ];

  final List<String> _predictionWindow = <String>[];
  String? _activeNotifiedSound;

  static const int _windowSize = 5;
  static const double _onsetThreshold = 0.60;
  static const double _offsetThreshold = 0.20;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadPreferences();
    _historyCleanupTimer = Timer.periodic(const Duration(minutes: 5), (_) {
      _cleanupAndSortHistory();
    });
  }

  Future<void> _loadPreferences() async {
    String storedMode = "general";
    String storedBackbone = "yamnet";
    String storedPersonalisationLevel = "default";
    bool storedContextAware = false;
    List<CustomSoundSample> storedSounds = const <CustomSoundSample>[];
    bool storedVibration = false;
    bool storedCameraFlash = false;
    List<Map<String, dynamic>> storedHistory = const <Map<String, dynamic>>[];
    List<Map<String, dynamic>> storedCustomClasses = const <Map<String, dynamic>>[];
    bool storedMonitoring = false;

    try {
      storedMode = await _storageService.readMode();
      storedBackbone = await _storageService.readBackbone();
      storedPersonalisationLevel = await _storageService.readPersonalisationLevel();
      storedContextAware = await _storageService.readContextAware();
      storedSounds = await _storageService.readRecordedSounds();
      storedVibration = await _storageService.readVibrationEnabled();
      storedCameraFlash = await _storageService.readCameraFlashEnabled();
      storedHistory = await _storageService.readEventHistory();
      storedCustomClasses = await _storageService.readAllClassConfigs();
      storedMonitoring = await _runtimeBridge.isMonitoringActive();
    } catch (_) {
      storedMode = "general";
      storedBackbone = "yamnet";
      storedPersonalisationLevel = "default";
      storedContextAware = false;
      storedSounds = const <CustomSoundSample>[];
      storedVibration = false;
      storedCameraFlash = false;
      storedHistory = const <Map<String, dynamic>>[];
      storedCustomClasses = const <Map<String, dynamic>>[];
      storedMonitoring = false;
    }

    if (!mounted) {
      return;
    }

    unawaited(_runtimeBridge.requestNotificationPermission());
    
    try {
      final bool batteryIgnored = await _runtimeBridge.isBatteryOptimizationIgnored();
      if (!batteryIgnored) {
        unawaited(_runtimeBridge.requestIgnoreBatteryOptimizations());
      }
    } catch (_) {
      // Ignored in test/non-Android environments
    }

    setState(() {
      _monitoring = storedMonitoring;
      _mode = storedMode;
      _backbone = storedBackbone;
      _personalisationLevel = storedPersonalisationLevel;
      _contextAware = storedContextAware;
      _vibrationEnabled = storedVibration;
      _cameraFlashEnabled = storedCameraFlash;
      _recordedSounds
        ..clear()
        ..addAll(storedSounds);

      _soundOptions
        ..clear()
        ..addAll(_builtInOptions);
      for (final map in storedCustomClasses) {
        final String? slug = map["slug"] as String?;
        final String? label = map["label"] as String?;
        if (slug != null && label != null &&
            !_soundOptions.any((opt) => opt.slug == slug)) {
          _soundOptions.add(PersonalizableSoundOption(slug, label));
        }
      }

      int legacyCounter = 0;
      final List<MonitorHistoryEntry> loadedHistory = [];

      for (final map in storedHistory) {
        final eventMap = map["event"] as Map<dynamic, dynamic>? ?? {};
        final event = InferenceEvent(
          label: (map["predicted_label"] as String?) ?? (eventMap["label"] as String?) ?? "unknown",
          confidenceTier: (map["confidence_tier"] as String?) ?? (eventMap["confidence_tier"] as String?) ?? "low",
          confidence: ((map["confidence"] as num?) ?? (eventMap["confidence"] as num?) ?? 0.0).toDouble(),
          isUnknown: (map["is_unknown"] as bool?) ?? (eventMap["flags"]?["is_unknown"] as bool?) ?? false,
          isBackground: (map["is_background"] as bool?) ?? (eventMap["flags"]?["is_background"] as bool?) ?? false,
          waveformSamples: (map["waveform_samples"] as List<dynamic>? ?? eventMap["waveform_samples"] as List<dynamic>? ?? const <dynamic>[])
              .map((e) => (e as num).toDouble())
              .toList(growable: false),
        );
        final String fallbackId = "${DateTime.now().microsecondsSinceEpoch}_${legacyCounter++}";
        loadedHistory.add(MonitorHistoryEntry(
          id: (map["event_id"] as String?) ?? fallbackId,
          event: event,
          capturedAt: DateTime.tryParse(map["timestamp"] as String? ?? "") ?? DateTime.now(),
          waveformSamples: event.waveformSamples,
          audioFilePath: (map["audio_file_path"] as String?) ?? "",
        ));
      }

      _history
        ..clear()
        ..addAll(loadedHistory);
    });

    if (storedMonitoring) {
      _eventsSub = _runtimeBridge.streamEvents().listen((InferenceEvent event) {
        unawaited(_onEvent(event));
      });
    }

    await _cleanupAndSortHistory();
    await _syncBackgroundEvents();
  }

  Future<void> _toggleMonitoring() async {
    if (_monitoring) {
      await _runtimeBridge.stopMonitoring();
      await _eventsSub?.cancel();
      if (!mounted) {
        return;
      }
      setState(() {
        _monitoring = false;
      });
      return;
    }

    final bool permissionGranted = await _audioRecorder.hasPermission();
    if (!permissionGranted) {
      _showErrorSnackBar(
          "Microphone permission is required to monitor sounds.");
      return;
    }

    // Prompt for camera permission if any active configs use flash alerts
    try {
      final List<Map<String, dynamic>> configs = await _storageService.readAllClassConfigs();
      final bool hasCameraFlashConfigured = configs.any((Map<String, dynamic> c) => c["custom_flash"] != "none");
      if (hasCameraFlashConfigured) {
        final bool hasCamPermission = await _runtimeBridge.hasCameraPermission();
        if (!hasCamPermission) {
          await _runtimeBridge.requestCameraPermission();
        }
      }
    } catch (_) {}

    try {
      await _runtimeBridge.startMonitoring(mode: _mode, backbone: _backbone);
      _runtimeReadyBackbone = _backbone;
      await _applyPersonalizedPrototypes();
    } catch (e) {
      _showErrorSnackBar("Failed to initialize ML engine: $e");
      return;
    }
    _eventsSub = _runtimeBridge.streamEvents().listen((InferenceEvent event) {
      unawaited(_onEvent(event));
    });

    if (!mounted) {
      return;
    }

    setState(() {
      _monitoring = true;
    });
  }

  void _showErrorSnackBar(String message) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 3),
        backgroundColor: const Color(0xFFB00020),
      ),
    );
  }

  Future<void> _onEvent(InferenceEvent event) async {
    if (!mounted) {
      return;
    }
    // Print Label & confidence for debugging. To be removed in final version.
    debugPrint("Received event: label=${event.label}, confidence=${event.confidence}");

    // 1. Standardize label (background/unknown mapped to "background")
    final String label =
        (event.isBackground || event.isUnknown) ? "background" : event.label;
    // 2. Add to prediction window
    _predictionWindow.add(label);
    if (_predictionWindow.length > _windowSize) {
      _predictionWindow.removeAt(0);
    }

    // 3. Count frequencies in the sliding window
    final Map<String, int> frequencies = <String, int>{};
    for (final String l in _predictionWindow) {
      frequencies[l] = (frequencies[l] ?? 0) + 1;
    }

    final String? active = _activeNotifiedSound;

    if (active == null) {
      // 4. Onset check: Look for a non-background label that meets the onset threshold
      String? candidateLabel;
      int maxCount = 0;
      frequencies.forEach((String key, int count) {
        if (key != "background" && count > maxCount) {
          maxCount = count;
          candidateLabel = key;
        }
      });

      if (candidateLabel != null) {
        final double ratio = maxCount / _predictionWindow.length;
        if (ratio >= _onsetThreshold) {
          // Load user overrides and check if detection is active
          final Map<String, dynamic>? userOverrides = await _storageService.readClassConfig(candidateLabel!);
          final bool enabled = userOverrides == null || (userOverrides["enabled"] as bool? ?? true);
          if (!enabled) {
            _activeNotifiedSound = null;
            return;
          }

          // Onset triggered!
          _activeNotifiedSound = candidateLabel;

          final InferenceEvent evaluatedEvent = InferenceEvent(
            label: candidateLabel!,
            confidenceTier: event.confidenceTier,
            confidence: event.confidence,
            isUnknown: event.isUnknown,
            isBackground: event.isBackground,
            waveformSamples: event.waveformSamples,
          );

          // Evaluate event with the ContextManager and active mode settings
          final AlertDecision decision = _contextManager.decideAlert(
            event: evaluatedEvent,
            activeMode: _mode,
            userOverrides: userOverrides,
          );

          // Dispatch visual and haptic modalities if NOT suppressed
          if (!decision.suppressed) {
            final String cleanSoundName = formatSoundLabel(candidateLabel!);
            final String title = "Sound Detected!";
            final String body =
                "$cleanSoundName (${(decision.confidence * 100).toStringAsFixed(0)}% confidence)";

            switch (decision.modality) {
              case AlertModality.none:
              case AlertModality.silentLog:
                break;
              case AlertModality.textOnly:
                unawaited(_runtimeBridge.showNotification(title: title, body: body));
                break;
              case AlertModality.standardVibrationText:
              case AlertModality.vibrationFlashText:
                unawaited(_runtimeBridge.showNotification(title: title, body: body));
                final String vibPattern = (userOverrides?["custom_vibration"] as String?) ?? "single";
                final String flsPattern = (userOverrides?["custom_flash"] as String?) ?? "single";
                unawaited(_triggerCustomAlertEffects(vibPattern, flsPattern));
                break;
              case AlertModality.criticalFullScreen:
                unawaited(_runtimeBridge.showNotification(
                  title: "CRITICAL ALERT: $cleanSoundName",
                  body: "Sound Pulse detected $cleanSoundName.",
                ));
                // Critical alert continuous bypass haptic cues: uses long and strobe flash for absolute safety
                unawaited(_triggerCustomAlertEffects("long", "strobe"));
                break;
            }
          }

          final String detectionId =
              "${DateTime.now().microsecondsSinceEpoch}_${_detectionCounter++}";
          final DateTime capturedAt = DateTime.now();
          final List<double> waveformSamples =
              List<double>.from(event.waveformSamples, growable: false);
          final String audioFilePath = waveformSamples.isEmpty
              ? ""
              : await _persistDetectionAudioClip(detectionId, waveformSamples);

          if (!mounted) {
            await _deleteDetectionFile(audioFilePath);
            return;
          }

          final InferenceEvent finalEvent = InferenceEvent(
            label: candidateLabel!,
            confidenceTier: event.confidenceTier,
            confidence: decision.confidence, // Use boosted confidence if any!
            isUnknown: event.isUnknown,
            isBackground: event.isBackground,
            waveformSamples: event.waveformSamples,
          );

          setState(() {
            _history.insert(
              0,
              MonitorHistoryEntry(
                id: detectionId,
                event: finalEvent,
                capturedAt: capturedAt,
                waveformSamples: waveformSamples,
                audioFilePath: audioFilePath,
              ),
            );
          });

          await _cleanupAndSortHistory();
        }
      }
    } else {
      // 5. Offset check: Check if the currently active sound falls below offset threshold
      final int count = frequencies[active] ?? 0;
      final double ratio = count / _predictionWindow.length;
      if (ratio < _offsetThreshold) {
        // Offset triggered! Clear the active state
        _activeNotifiedSound = null;
      }
    }
  }

  Future<String> _persistDetectionAudioClip(
      String id, List<double> waveformSamples) async {
    final Directory appDirectory = await getApplicationDocumentsDirectory();
    final Directory detectionsDirectory = Directory(
      "${appDirectory.path}${Platform.pathSeparator}detections",
    );
    await detectionsDirectory.create(recursive: true);
    final String clipPath =
        "${detectionsDirectory.path}${Platform.pathSeparator}detection_$id.wav";
    final Uint8List wavBytes = _encodePcm16Wav(waveformSamples);
    await File(clipPath).writeAsBytes(wavBytes, flush: true);
    return clipPath;
  }

  Uint8List _encodePcm16Wav(List<double> samples, {int sampleRateHz = 16000}) {
    final int numChannels = 1;
    final int bitsPerSample = 16;
    final int bytesPerSample = bitsPerSample ~/ 8;
    final int dataBytes = samples.length * bytesPerSample;
    final ByteData header = ByteData(44);
    final Uint8List headerBytes = header.buffer.asUint8List();

    headerBytes.setAll(0, "RIFF".codeUnits);
    header.setUint32(4, 36 + dataBytes, Endian.little);
    headerBytes.setAll(8, "WAVE".codeUnits);
    headerBytes.setAll(12, "fmt ".codeUnits);
    header.setUint32(16, 16, Endian.little);
    header.setUint16(20, 1, Endian.little);
    header.setUint16(22, numChannels, Endian.little);
    header.setUint32(24, sampleRateHz, Endian.little);
    header.setUint32(
      28,
      sampleRateHz * numChannels * bytesPerSample,
      Endian.little,
    );
    header.setUint16(32, numChannels * bytesPerSample, Endian.little);
    header.setUint16(34, bitsPerSample, Endian.little);
    headerBytes.setAll(36, "data".codeUnits);
    header.setUint32(40, dataBytes, Endian.little);

    final ByteData pcmData = ByteData(dataBytes);
    for (int i = 0; i < samples.length; i += 1) {
      final double clamped = samples[i].clamp(-1.0, 1.0);
      final int value = (clamped * 32767.0).round();
      pcmData.setInt16(i * bytesPerSample, value, Endian.little);
    }

    return Uint8List.fromList(
      <int>[
        ...headerBytes,
        ...pcmData.buffer.asUint8List(),
      ],
    );
  }

  Future<void> _deleteDetectionFile(String filePath) async {
    if (filePath.isEmpty) {
      return;
    }
    final File file = File(filePath);
    if (await file.exists()) {
      await file.delete();
    }
  }

  Future<void> _changeMode(String mode) async {
    await _runtimeBridge.updateMode(mode);
    await _storageService.saveMode(mode);

    if (!mounted) {
      return;
    }

    setState(() {
      _mode = mode;
    });
  }

  Future<void> _changeBackbone(String backbone) async {
    await _storageService.saveBackbone(backbone);
    if (!mounted) {
      return;
    }
    setState(() {
      _backbone = backbone;
    });

    if (_monitoring) {
      try {
        await _runtimeBridge.stopMonitoring();
        await _runtimeBridge.startMonitoring(mode: _mode, backbone: backbone);
        _runtimeReadyBackbone = backbone;
        await _applyPersonalizedPrototypes();
      } catch (e) {
        _showErrorSnackBar("Failed to dynamically switch model: $e");
        setState(() {
          _monitoring = false;
        });
      }
    }
  }

  Future<void> _setContextAware(bool enabled) async {
    await _runtimeBridge.setContextAware(enabled);
    await _storageService.saveContextAware(enabled);

    if (!mounted) {
      return;
    }

    setState(() {
      _contextAware = enabled;
    });
  }

  Future<void> _setPersonalisationLevel(String level) async {
    await _storageService.savePersonalisationLevel(level);
    if (!mounted) {
      return;
    }
    setState(() {
      _personalisationLevel = level;
    });
    await _applyPersonalizedPrototypes();
  }

  Future<void> _addCustomSoundClass(String label) async {
    final String slug = "custom_${label.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), '_')}";
    if (_soundOptions.any((opt) => opt.slug == slug)) {
      return;
    }
    await _storageService.saveClassConfig(slug, {
      "slug": slug,
      "label": label,
      "is_custom": true,
    });
    if (!mounted) {
      return;
    }
    setState(() {
      _soundOptions.add(PersonalizableSoundOption(slug, label));
    });
    unawaited(_applyPersonalizedPrototypes());
  }

  Future<void> _saveRecordedSound(
    String name,
    String classSlug,
    String classLabel,
    String filePath,
    DateTime recordedAt,
    int durationSeconds,
  ) async {
    final List<double> samples = await _readPcm16WavSamples(filePath);
    final String? qualityWarning = _qualityWarning(samples, durationSeconds);
    await _ensureRuntimeReady();
    final List<double> embedding =
        await _runtimeBridge.extractEmbeddingFromPcm(samples);

    final String soundId =
        "${DateTime.now().microsecondsSinceEpoch}_${_recordedSounds.length}";
    final String encryptedPath = filePath.replaceFirst(".wav", ".enc");
    try {
      await _storageService.encryptFile(filePath, encryptedPath);
      final File plainFile = File(filePath);
      if (await plainFile.exists()) {
        await plainFile.delete();
      }
    } catch (e) {
      _showErrorSnackBar("Encryption failed: $e");
      return;
    }

    final CustomSoundSample sample = CustomSoundSample(
      id: soundId,
      name: name,
      classSlug: classSlug,
      classLabel: classLabel,
      recordedAt: recordedAt,
      durationSeconds: durationSeconds,
      filePath: encryptedPath,
      embedding: embedding,
      audit: <String, dynamic>{
        "created_at": DateTime.now().toIso8601String(),
        "backbone": _backbone,
        "sample_rate_hz": 16000,
        "quality_warning": qualityWarning,
      },
    );
    final List<CustomSoundSample> nextSounds = <CustomSoundSample>[
      sample,
      ..._recordedSounds
    ];
    await _storageService.saveRecordedSounds(nextSounds);
    await _refreshPersonalizedPrototype(classSlug, nextSounds);
    if (!mounted) {
      return;
    }
    setState(() {
      _recordedSounds
        ..clear()
        ..addAll(nextSounds);
    });
    if (qualityWarning != null) {
      _showErrorSnackBar(qualityWarning);
    }
  }

  Future<void> _renameRecordedSound(
      CustomSoundSample sample, String name) async {
    final List<CustomSoundSample> nextSounds = _recordedSounds
        .map((CustomSoundSample item) =>
            item.id == sample.id ? item.copyWith(name: name) : item)
        .toList(growable: false);
    await _storageService.saveRecordedSounds(nextSounds);
    if (!mounted) {
      return;
    }
    setState(() {
      _recordedSounds
        ..clear()
        ..addAll(nextSounds);
    });
  }

  Future<void> _deleteRecordedSound(CustomSoundSample sample) async {
    if (sample.filePath.isNotEmpty) {
      final File file = File(sample.filePath);
      if (await file.exists()) {
        await file.delete();
      }
    }

    final List<CustomSoundSample> nextSounds = _recordedSounds
        .where((CustomSoundSample item) => item.id != sample.id)
        .toList(growable: false);
    await _storageService.saveRecordedSounds(nextSounds);
    await _refreshPersonalizedPrototype(sample.classSlug, nextSounds);
    if (!mounted) {
      return;
    }
    setState(() {
      _recordedSounds
        ..clear()
        ..addAll(nextSounds);
    });
  }

  Future<void> _ensureRuntimeReady() async {
    if (_runtimeReadyBackbone == _backbone) {
      return;
    }
    await _runtimeBridge.initializeRuntime(backbone: _backbone);
    _runtimeReadyBackbone = _backbone;
    await _applyPersonalizedPrototypes();
  }

  Future<void> _applyPersonalizedPrototypes() async {
    final List<String> classSlugs = _soundOptions.map((opt) => opt.slug).toList();

    if (_personalisationLevel == "default") {
      for (final String slug in classSlugs) {
        await _runtimeBridge.resetPrototype(slug);
      }
      return;
    }

    final int minSamples = _personalisationLevel == "quick" ? 1 : 3;
    final String sourceType = _personalisationLevel == "quick"
        ? "quick_personalisation"
        : "advanced_personalisation";

    final Map<String, List<double>> centroids = {};

    for (final String slug in classSlugs) {
      final List<List<double>> embeddings = _recordedSounds
          .where((CustomSoundSample sample) =>
              sample.classSlug == slug && sample.embedding.isNotEmpty)
          .map((CustomSoundSample sample) => sample.embedding)
          .toList(growable: false);

      if (embeddings.length >= minSamples) {
        final int length = embeddings.first.length;
        final List<double> centroid = List<double>.filled(length, 0.0);
        for (final List<double> embedding in embeddings) {
          for (int index = 0;
              index < length && index < embedding.length;
              index += 1) {
            centroid[index] += embedding[index];
          }
        }
        for (int index = 0; index < centroid.length; index += 1) {
          centroid[index] /= embeddings.length;
        }

        centroids[slug] = centroid;
        await _runtimeBridge.updatePrototype(
          label: slug,
          embedding: centroid,
        );
      } else {
        await _runtimeBridge.resetPrototype(slug);
      }
    }

    await _storageService.savePrototypeCentroids(centroids, sourceType: sourceType);
  }

  Future<void> _refreshPersonalizedPrototype(
    String classSlug,
    List<CustomSoundSample> sounds,
  ) async {
    final List<List<double>> embeddings = sounds
        .where((CustomSoundSample sample) =>
            sample.classSlug == classSlug && sample.embedding.isNotEmpty)
        .map((CustomSoundSample sample) => sample.embedding)
        .toList(growable: false);
    final Map<String, List<double>> centroids = Map<String, List<double>>.from(
        await _storageService.readPrototypeCentroids());

    if (_personalisationLevel == "default") {
      centroids.remove(classSlug);
      await _storageService.savePrototypeCentroids(centroids);
      await _runtimeBridge.resetPrototype(classSlug);
      return;
    }

    final int minSamples = _personalisationLevel == "quick" ? 1 : 3;
    final String sourceType = _personalisationLevel == "quick"
        ? "quick_personalisation"
        : "advanced_personalisation";

    if (embeddings.length < minSamples) {
      centroids.remove(classSlug);
      await _storageService.savePrototypeCentroids(centroids, sourceType: sourceType);
      if (_runtimeReadyBackbone == _backbone) {
        await _runtimeBridge.resetPrototype(classSlug);
      }
      return;
    }

    final int length = embeddings.first.length;
    final List<double> centroid = List<double>.filled(length, 0.0);
    for (final List<double> embedding in embeddings) {
      for (int index = 0;
          index < length && index < embedding.length;
          index += 1) {
        centroid[index] += embedding[index];
      }
    }
    for (int index = 0; index < centroid.length; index += 1) {
      centroid[index] /= embeddings.length;
    }

    centroids[classSlug] = centroid;
    await _storageService.savePrototypeCentroids(centroids, sourceType: sourceType);
    await _ensureRuntimeReady();
    await _runtimeBridge.updatePrototype(label: classSlug, embedding: centroid);
  }
  Future<void> _cleanupAndSortHistory() async {
    final DateTime cutoff = DateTime.now().subtract(const Duration(hours: 24));
    
    // Sort by recency of occurrence: most recent first, oldest last
    _history.sort((a, b) => b.capturedAt.compareTo(a.capturedAt));
    
    final List<MonitorHistoryEntry> kept = [];
    final List<MonitorHistoryEntry> removed = [];
    
    for (final entry in _history) {
      if (entry.capturedAt.isAfter(cutoff)) {
        kept.add(entry);
      } else {
        removed.add(entry);
      }
    }
    
    if (removed.isNotEmpty) {
      for (final entry in removed) {
        if (entry.audioFilePath.isNotEmpty) {
          await _deleteDetectionFile(entry.audioFilePath);
        }
      }
    }
    
    if (mounted) {
      setState(() {
        _history.clear();
        _history.addAll(kept);
      });
    } else {
      _history.clear();
      _history.addAll(kept);
    }
    
    final List<Map<String, dynamic>> historyPayload = _history.map((entry) {
      return <String, dynamic>{
        "event_id": entry.id,
        "timestamp": entry.capturedAt.toIso8601String(),
        "predicted_label": entry.event.label,
        "confidence_tier": entry.event.confidenceTier,
        "confidence": entry.event.confidence,
        "is_unknown": entry.event.isUnknown,
        "is_background": entry.event.isBackground,
        "active_mode": _mode,
        "was_user_alerted": true,
        "waveform_samples": entry.waveformSamples,
        "audio_file_path": entry.audioFilePath,
      };
    }).toList();
    
    await _storageService.clearEventHistory();
    await _storageService.saveEventHistory(historyPayload);
  }

  Future<void> _syncBackgroundEvents() async {
    final List<Map<String, dynamic>> bgEvents =
        await _storageService.readAndClearBackgroundEvents();
    if (bgEvents.isEmpty) {
      return;
    }

    final List<MonitorHistoryEntry> newEntries = <MonitorHistoryEntry>[];
    int legacyCounter = 0;
    for (final Map<String, dynamic> map in bgEvents) {
      final String fallbackId =
          "${DateTime.now().microsecondsSinceEpoch}_${legacyCounter++}";
      final String label = map["predicted_label"] as String? ?? "unknown";
      final String tier = map["confidence_tier"] as String? ?? "low";
      final double confidence = ((map["confidence"] as num?) ?? 0.0).toDouble();
      final bool isUnknown = map["is_unknown"] as bool? ?? false;
      final bool isBackground = map["is_background"] as bool? ?? false;
      List<double> samples = (map["waveform_samples"] as List<dynamic>? ?? const <dynamic>[])
          .map((dynamic e) => (e as num).toDouble())
          .toList(growable: false);

      // Recover waveform samples from the persisted WAV file if not in JSON
      final String audioPath = map["audio_file_path"] as String? ?? "";
      if (samples.isEmpty && audioPath.isNotEmpty) {
        try {
          final File wavFile = File(audioPath);
          if (await wavFile.exists()) {
            samples = await _readPcm16WavSamples(audioPath);
          }
        } catch (_) {
          // Use empty samples if WAV recovery fails
        }
      }

      final InferenceEvent event = InferenceEvent(
        label: label,
        confidenceTier: tier,
        confidence: confidence,
        isUnknown: isUnknown,
        isBackground: isBackground,
        waveformSamples: samples,
      );

      newEntries.add(MonitorHistoryEntry(
        id: map["event_id"] as String? ?? fallbackId,
        event: event,
        capturedAt: DateTime.tryParse(map["timestamp"] as String? ?? "") ?? DateTime.now(),
        waveformSamples: samples,
        audioFilePath: audioPath,
      ));
    }

    if (newEntries.isEmpty) return;

    if (mounted) {
      setState(() {
        _history.addAll(newEntries);
      });
    } else {
      _history.addAll(newEntries);
    }

    await _cleanupAndSortHistory();
  }

  Future<List<double>> _readPcm16WavSamples(String filePath) async {
    final Uint8List bytes = filePath.endsWith(".enc")
        ? await _storageService.decryptFile(filePath)
        : await File(filePath).readAsBytes();
    if (bytes.length < 44) {
      throw const FormatException("Recorded WAV file is too short.");
    }
    final ByteData data = ByteData.sublistView(bytes);
    int offset = 12;
    int? dataOffset;
    int? dataSize;
    while (offset + 8 <= bytes.length) {
      final String chunkId =
          String.fromCharCodes(bytes.sublist(offset, offset + 4));
      final int chunkSize = data.getUint32(offset + 4, Endian.little);
      if (chunkId == "data") {
        dataOffset = offset + 8;
        dataSize = chunkSize;
        break;
      }
      offset += 8 + chunkSize + (chunkSize.isOdd ? 1 : 0);
    }
    if (dataOffset == null || dataSize == null) {
      throw const FormatException("Recorded WAV file has no PCM data chunk.");
    }
    final int sampleCount = dataSize ~/ 2;
    return List<double>.generate(sampleCount, (int index) {
      final int byteOffset = dataOffset! + (index * 2);
      if (byteOffset + 2 > bytes.length) {
        return 0.0;
      }
      return (data.getInt16(byteOffset, Endian.little) / 32768.0)
          .clamp(-1.0, 1.0);
    }, growable: false);
  }

  String? _qualityWarning(List<double> samples, int durationSeconds) {
    if (durationSeconds < 3 || samples.length < 16000 * 3) {
      return "Sample is shorter than the recommended minimum duration.";
    }
    double sumSquares = 0.0;
    double peak = 0.0;
    for (final double sample in samples) {
      final double absSample = sample.abs();
      peak = peak > absSample ? peak : absSample;
      sumSquares += sample * sample;
    }
    final double rms = samples.isEmpty
        ? 0.0
        : math.sqrt(sumSquares / samples.length.toDouble());
    if (peak < 0.05 || rms < 0.01) {
      return "Sample is very quiet. Re-record closer to the sound if accuracy is poor.";
    }
    if (peak > 0.98) {
      return "Sample may be clipped. Re-record with a little more distance if accuracy is poor.";
    }
    return null;
  }

  Future<void> _triggerCustomAlertEffects(String vibrationPattern, String flashPattern) async {
    if (_vibrationEnabled && vibrationPattern != "none") {
      await _runtimeBridge.vibrate(vibrationPattern);
    }

    if (_cameraFlashEnabled && flashPattern != "none") {
      final bool hasCamPermission = await _runtimeBridge.hasCameraPermission();
      if (!hasCamPermission) {
        await _runtimeBridge.requestCameraPermission();
      }
      await _runtimeBridge.triggerCameraFlash(flashPattern);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _historyCleanupTimer?.cancel();
    _contextManager.reset();
    for (final MonitorHistoryEntry entry in _history) {
      if (entry.audioFilePath.isNotEmpty) {
        unawaited(_deleteDetectionFile(entry.audioFilePath));
      }
    }
    unawaited(_audioRecorder.dispose());
    if (_monitoring) {
      _runtimeBridge.stopMonitoring();
    }
    _eventsSub?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      // Brief delay to let the native service finish any in-progress file writes
      // before reading the background events sync file.
      Future<void>.delayed(const Duration(milliseconds: 500), () {
        if (mounted) {
          _syncBackgroundEvents();
        }
      });
    }
  }

  Widget _buildTopBar() {
    return SafeArea(
      bottom: false,
      child: Container(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
        decoration: const BoxDecoration(
          color: monitorSurface,
          boxShadow: <BoxShadow>[
            BoxShadow(
              color: Color(0x14000000),
              offset: Offset(0, 4),
              blurRadius: 10,
            ),
          ],
        ),
        child: Row(
          children: <Widget>[
            GestureDetector(
              onTap: _toggleMonitoring,
              child: Row(
                children: <Widget>[
                  Container(
                    width: 44,
                    height: 24,
                    padding: const EdgeInsets.all(3),
                    decoration: monitorPressed(radius: 24),
                    child: Align(
                      alignment: _monitoring
                          ? Alignment.centerRight
                          : Alignment.centerLeft,
                      child: Container(
                        width: 18,
                        height: 18,
                        decoration: monitorRaised(radius: 18).copyWith(
                          color: _monitoring
                              ? monitorPrimary
                              : monitorOnSurfaceVariant,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _monitoring ? "ON" : "OFF",
                    style: const TextStyle(
                      fontSize: 10,
                      letterSpacing: 1.0,
                      fontWeight: FontWeight.w700,
                      color: monitorOnSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
            const Spacer(),
            const Text(
              "Sound Pulse",
              style: TextStyle(
                color: monitorOnSurface,
                fontSize: 22,
                fontWeight: FontWeight.w600,
              ),
            ),
            const Spacer(),
            GestureDetector(
              onTap: () {
                if (_tabIndex == 3) {
                  setState(() {
                    _tabIndex = _previousTabIndex;
                  });
                } else {
                  setState(() {
                    _previousTabIndex = _tabIndex;
                    _tabIndex = 3;
                  });
                }
              },
              child: Container(
                width: 40,
                height: 40,
                decoration: monitorRaised(radius: 12),
                child: Icon(
                  _tabIndex == 3 ? Icons.arrow_back : Icons.settings,
                  size: 20,
                  color: monitorOnSurfaceVariant,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: monitorSurface,
      body: Column(
        children: <Widget>[
          _buildTopBar(),
          Expanded(
            child: IndexedStack(
              index: _tabIndex,
              children: <Widget>[
                MonitorLiveFeedTab(
                  mode: _mode,
                  backbone: _backbone,
                  contextAware: _contextAware,
                  monitoring: _monitoring,
                  history: _history,
                  onModeChanged: _changeMode,
                  onBackboneChanged: _changeBackbone,
                  onContextAwareChanged: _setContextAware,
                  onViewAll: () => setState(() => _tabIndex = 1),
                ),
                MonitorHistoryTab(
                  history: _history,
                ),
                MonitorRecordTab(
                  key: const ValueKey("MonitorRecordTab"),
                  recordedSounds: _recordedSounds,
                  soundOptions: _soundOptions,
                  personalisationLevel: _personalisationLevel,
                  onPersonalisationLevelChanged: _setPersonalisationLevel,
                  onSaveSound: _saveRecordedSound,
                  onRenameSound: _renameRecordedSound,
                  onDeleteSound: _deleteRecordedSound,
                  onDecryptFile: _storageService.decryptToTempFile,
                  onAddCustomClass: _addCustomSoundClass,
                ),
                MonitorSettingsTab(
                  activeMode: _mode,
                  contextAware: _contextAware,
                  onModeChanged: _changeMode,
                  onContextAwareChanged: _setContextAware,
                  onBack: () => setState(() => _tabIndex = _previousTabIndex),
                ),
              ],
            ),
          ),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tabIndex == 3 ? _previousTabIndex : _tabIndex,
        onDestinationSelected: (int index) {
          setState(() => _tabIndex = index);
          if (index == 0 || index == 1) {
            unawaited(_syncBackgroundEvents());
          }
        },
        destinations: const <NavigationDestination>[
          NavigationDestination(
            icon: Icon(Icons.waves),
            label: "Live Feed",
          ),
          NavigationDestination(
            icon: Icon(Icons.history),
            label: "History",
          ),
          NavigationDestination(
            icon: Icon(Icons.mic),
            label: "Record",
          ),
        ],
      ),
    );
  }
}
