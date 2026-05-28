import "dart:async";
import "dart:io";
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
import "monitor_style.dart";

class MonitorPage extends StatefulWidget {
  const MonitorPage({super.key});

  @override
  State<MonitorPage> createState() => _MonitorPageState();
}

class _MonitorPageState extends State<MonitorPage> {
  final RuntimeBridge _runtimeBridge = RuntimeBridge();
  final LocalStorageService _storageService = LocalStorageService();
  final List<MonitorHistoryEntry> _history = <MonitorHistoryEntry>[];
  final List<CustomSoundSample> _recordedSounds = <CustomSoundSample>[];
  final AudioRecorder _audioRecorder = AudioRecorder();

  StreamSubscription<InferenceEvent>? _eventsSub;

  bool _monitoring = false;
  bool _contextAware = false;
  String _mode = "general";
  String _backbone = "yamnet";
  int _tabIndex = 0;
  bool _vibrationEnabled = false;
  bool _cameraFlashEnabled = false;

  final List<String> _predictionWindow = <String>[];
  String? _activeNotifiedSound;

  static const int _windowSize = 15;
  static const double _onsetThreshold = 0.0;
  static const double _offsetThreshold = 0.0;

  @override
  void initState() {
    super.initState();
    _loadPreferences();
  }

  Future<void> _loadPreferences() async {
    String storedMode = "general";
    String storedBackbone = "yamnet";
    bool storedContextAware = false;
    List<CustomSoundSample> storedSounds = const <CustomSoundSample>[];
    bool storedVibration = false;
    bool storedCameraFlash = false;

    try {
      storedMode = await _storageService.readMode();
      storedBackbone = await _storageService.readBackbone();
      storedContextAware = await _storageService.readContextAware();
      storedSounds = await _storageService.readRecordedSounds();
      storedVibration = await _storageService.readVibrationEnabled();
      storedCameraFlash = await _storageService.readCameraFlashEnabled();
    } catch (_) {
      storedMode = "general";
      storedBackbone = "yamnet";
      storedContextAware = false;
      storedSounds = const <CustomSoundSample>[];
      storedVibration = false;
      storedCameraFlash = false;
    }

    if (!mounted) {
      return;
    }

    // Request notification permission on Android 13+ to ensure system alerts work
    unawaited(_runtimeBridge.requestNotificationPermission());

    setState(() {
      _mode = storedMode;
      _backbone = storedBackbone;
      _contextAware = storedContextAware;
      _vibrationEnabled = storedVibration;
      _cameraFlashEnabled = storedCameraFlash;
      _recordedSounds
        ..clear()
        ..addAll(storedSounds);
    });
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

    try {
      await _runtimeBridge.startMonitoring(mode: _mode, backbone: _backbone);
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
    // debugPrint("Received event: label=${event.label}, confidence=${event.confidence}");

    // 1. Standardize label (background/unknown mapped to "background")
    final String label = (event.isBackground || event.isUnknown) ? "background" : event.label;
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
          // Onset triggered!
          _activeNotifiedSound = candidateLabel;

          // Trigger alerts & persist to logs
          final String title = "Sound Detected!";
          final String body = "$candidateLabel (${(event.confidence * 100).toStringAsFixed(0)}% confidence)";
          unawaited(_runtimeBridge.showNotification(title: title, body: body));

          if (_vibrationEnabled) {
            unawaited(_runtimeBridge.vibrate());
          }

          if (_cameraFlashEnabled) {
            unawaited(_runtimeBridge.triggerCameraFlash());
          }

          final String detectionId = DateTime.now().microsecondsSinceEpoch.toString();
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

          MonitorHistoryEntry? removedEntry;
          setState(() {
            _history.insert(
              0,
              MonitorHistoryEntry(
                id: detectionId,
                event: event,
                capturedAt: capturedAt,
                waveformSamples: waveformSamples,
                audioFilePath: audioFilePath,
              ),
            );
            if (_history.length > 20) {
              removedEntry = _history.removeLast();
            }
          });

          final String pathToDelete = removedEntry?.audioFilePath ?? "";
          if (pathToDelete.isNotEmpty) {
            unawaited(_deleteDetectionFile(pathToDelete));
          }
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

  Future<void> _setVibrationEnabled(bool enabled) async {
    await _storageService.saveVibrationEnabled(enabled);
    if (!mounted) {
      return;
    }
    setState(() {
      _vibrationEnabled = enabled;
    });
  }

  Future<void> _setCameraFlashEnabled(bool enabled) async {
    await _storageService.saveCameraFlashEnabled(enabled);
    if (!mounted) {
      return;
    }
    setState(() {
      _cameraFlashEnabled = enabled;
    });
  }

  Future<void> _saveRecordedSound(
    String name,
    String filePath,
    DateTime recordedAt,
    int durationSeconds,
  ) async {
    final CustomSoundSample sample = CustomSoundSample(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      name: name,
      recordedAt: recordedAt,
      durationSeconds: durationSeconds,
      filePath: filePath,
    );
    final List<CustomSoundSample> nextSounds = <CustomSoundSample>[
      sample,
      ..._recordedSounds
    ];
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
    if (!mounted) {
      return;
    }
    setState(() {
      _recordedSounds
        ..clear()
        ..addAll(nextSounds);
    });
  }

  @override
  void dispose() {
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
              onTap: () => setState(() => _tabIndex = 2),
              child: Container(
                width: 40,
                height: 40,
                decoration: monitorRaised(radius: 12),
                child: const Icon(Icons.settings,
                    size: 20, color: monitorOnSurfaceVariant),
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
                MonitorHistoryTab(history: _history),
                MonitorRecordTab(
                  recordedSounds: _recordedSounds,
                  vibrationEnabled: _vibrationEnabled,
                  cameraFlashEnabled: _cameraFlashEnabled,
                  onVibrationToggle: _setVibrationEnabled,
                  onCameraFlashToggle: _setCameraFlashEnabled,
                  onSaveSound: _saveRecordedSound,
                  onRenameSound: _renameRecordedSound,
                  onDeleteSound: _deleteRecordedSound,
                ),
              ],
            ),
          ),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tabIndex,
        onDestinationSelected: (int index) {
          setState(() => _tabIndex = index);
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
