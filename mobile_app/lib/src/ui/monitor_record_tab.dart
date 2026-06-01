import "dart:async";
import "dart:io";

import "package:audioplayers/audioplayers.dart";
import "package:flutter/material.dart";
import "package:path_provider/path_provider.dart";
import "package:record/record.dart";

import "../models/custom_sound_sample.dart";
import "monitor_style.dart";

class PersonalizableSoundOption {
  const PersonalizableSoundOption(this.slug, this.label);

  final String slug;
  final String label;
}

const List<PersonalizableSoundOption> personalizableSoundOptions =
    <PersonalizableSoundOption>[
  PersonalizableSoundOption("crying_baby", "Crying baby"),
  PersonalizableSoundOption("water_running", "Water running"),
  PersonalizableSoundOption("dog_barking", "Dog barking"),
  PersonalizableSoundOption("door_wood_creaking", "Door creaking"),
  PersonalizableSoundOption("background", "Ambient background"),
];

class MonitorRecordTab extends StatefulWidget {
  const MonitorRecordTab({
    super.key,
    required this.recordedSounds,
    required this.soundOptions,
    required this.personalisationLevel,
    required this.vibrationEnabled,
    required this.cameraFlashEnabled,
    required this.onPersonalisationLevelChanged,
    required this.onVibrationToggle,
    required this.onCameraFlashToggle,
    required this.onSaveSound,
    required this.onRenameSound,
    required this.onDeleteSound,
    required this.onAddCustomClass,
    this.onDecryptFile,
  });

  final List<CustomSoundSample> recordedSounds;
  final List<PersonalizableSoundOption> soundOptions;
  final String personalisationLevel;
  final bool vibrationEnabled;
  final bool cameraFlashEnabled;
  final ValueChanged<String> onPersonalisationLevelChanged;
  final ValueChanged<bool> onVibrationToggle;
  final ValueChanged<bool> onCameraFlashToggle;
  final Future<void> Function(
    String name,
    String classSlug,
    String classLabel,
    String filePath,
    DateTime recordedAt,
    int durationSeconds,
  ) onSaveSound;
  final Future<void> Function(CustomSoundSample sound, String name)
      onRenameSound;
  final Future<void> Function(CustomSoundSample sound) onDeleteSound;
  final Future<void> Function(String label) onAddCustomClass;
  final Future<String> Function(String path)? onDecryptFile;

  @override
  State<MonitorRecordTab> createState() => _MonitorRecordTabState();
}

class _MonitorRecordTabState extends State<MonitorRecordTab> {
  static const Duration _recordDuration = Duration(seconds: 5);
  static const Duration _tickDuration = Duration(milliseconds: 100);

  Timer? _recordTimer;
  bool _recording = false;
  double _progress = 0.0;
  bool _awaitingName = false;
  String? _editingSoundId;
  String? _pendingRecordedFilePath;
  DateTime? _pendingRecordedAt;
  String? _playingSoundId;
  bool _playbackPaused = false;
  String? _statusMessage;
  late String _selectedClassSlug;
  late String _selectedClassLabel;
  List<double> _levelBars = const <double>[];
  bool _pendingPlaybackConfirmed = false;
  final AudioRecorder _audioRecorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _editController = TextEditingController();
  StreamSubscription<void>? _playerCompleteSub;
  File? _currentTempPlayFile;

  @override
  void initState() {
    super.initState();
    _selectedClassSlug = widget.soundOptions.first.slug;
    _selectedClassLabel = widget.soundOptions.first.label;
    _playerCompleteSub = _audioPlayer.onPlayerComplete.listen((_) async {
      final File? tempFile = _currentTempPlayFile;
      if (tempFile != null && await tempFile.exists()) {
        await tempFile.delete();
      }
      _currentTempPlayFile = null;
      if (!mounted) {
        return;
      }
      setState(() {
        _playingSoundId = null;
        _playbackPaused = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.soundOptions.any((opt) => opt.slug == _selectedClassSlug)) {
      _selectedClassSlug = widget.soundOptions.first.slug;
      _selectedClassLabel = widget.soundOptions.first.label;
    }
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          const Text(
            "Record Sounds",
            style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: monitorOnSurface),
          ),
          const SizedBox(height: 6),
          Text(
            widget.personalisationLevel == "default"
                ? "Select a personalisation mode to record and train custom sounds."
                : widget.personalisationLevel == "quick"
                    ? "Capture 1-3 clear samples for a sound class. Play each sample before saving."
                    : "Capture 3-5 clear samples for a sound class. Play each sample before saving.",
            style: const TextStyle(color: monitorOnSurfaceVariant, fontSize: 12),
          ),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: monitorRaised(radius: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const Text(
                  "Personalisation Mode",
                  style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: monitorOnSurface),
                ),
                const SizedBox(height: 12),
                _buildPersonalisationOption(
                  id: "default",
                  title: "Default Setup",
                  subtitle: "Built-in offline model, no custom sounds",
                  icon: Icons.settings_suggest,
                ),
                const SizedBox(height: 8),
                _buildPersonalisationOption(
                  id: "quick",
                  title: "Quick Personalisation",
                  subtitle: "1-3 samples per sound, fast home setup",
                  icon: Icons.bolt,
                ),
                const SizedBox(height: 8),
                _buildPersonalisationOption(
                  id: "advanced",
                  title: "Advanced Personalisation",
                  subtitle: "3-5 samples, strong environmental adaptation",
                  icon: Icons.psychology,
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: monitorRaised(radius: 20),
            child: widget.personalisationLevel == "default"
                ? const Column(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: <Widget>[
                      Icon(
                        Icons.mic_off,
                        size: 48,
                        color: monitorOnSurfaceVariant,
                      ),
                      SizedBox(height: 12),
                      Text(
                        "Personalisation is Disabled",
                        style: TextStyle(
                          color: monitorOnSurface,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      SizedBox(height: 6),
                      Text(
                        "Please select Quick Personalisation or Advanced Personalisation above to begin recording custom sounds.",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: monitorOnSurfaceVariant,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  )
                : Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Row(
                        children: <Widget>[
                          Expanded(
                            child: Text(
                              _recording
                                  ? "Recording in progress..."
                                  : _awaitingName
                                      ? "Play the sample, then save it"
                                      : "Ready to record a new sample",
                              style: const TextStyle(
                                  color: monitorOnSurface,
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600),
                            ),
                          ),
                          ElevatedButton.icon(
                            onPressed: _recording ||
                                    _awaitingName ||
                                    _selectedClassSamples.length >= _maxSamplesLimit
                                ? null
                                : _startRecording,
                            icon: const Icon(Icons.mic),
                            label: Text(_recording ? "Recording..." : "Record (5s)"),
                            style: ElevatedButton.styleFrom(
                              elevation: 0,
                              foregroundColor: Colors.white,
                              backgroundColor: monitorPrimary,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: <Widget>[
                          Expanded(
                            child: DropdownButtonFormField<String>(
                              // ignore: deprecated_member_use
                              value: _selectedClassSlug,
                              decoration: const InputDecoration(
                                labelText: "Sound class",
                                border: OutlineInputBorder(),
                              ),
                              items: widget.soundOptions
                                  .map(
                                    (PersonalizableSoundOption option) =>
                                        DropdownMenuItem<String>(
                                      value: option.slug,
                                      child: Text(option.label),
                                    ),
                                  )
                                  .toList(growable: false),
                              onChanged: _recording || _awaitingName
                                  ? null
                                  : (String? value) {
                                      final PersonalizableSoundOption option =
                                          widget.soundOptions.firstWhere(
                                        (PersonalizableSoundOption item) =>
                                            item.slug == value,
                                        orElse: () => widget.soundOptions.first,
                                      );
                                      setState(() {
                                        _selectedClassSlug = option.slug;
                                        _selectedClassLabel = option.label;
                                      });
                                    },
                            ),
                          ),
                          const SizedBox(width: 8),
                          Container(
                            height: 56,
                            decoration: monitorRaised(radius: 12),
                            child: IconButton(
                              icon: const Icon(Icons.add, color: monitorPrimary),
                              tooltip: "Add Custom Sound Class",
                              onPressed: _recording || _awaitingName
                                  ? null
                                  : _showAddCustomClassDialog,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Text(
                        "${_selectedClassSamples.length}/$_maxSamplesLimit samples saved for ${formatSoundLabel(_selectedClassLabel)} · minimum $_minSamplesRequired",
                        style: const TextStyle(
                            color: monitorOnSurfaceVariant, fontSize: 12),
                      ),
                      const SizedBox(height: 12),
                      LinearProgressIndicator(
                        value: _recording ? _progress : 0,
                        minHeight: 10,
                        borderRadius: BorderRadius.circular(8),
                        backgroundColor: monitorPrimarySoft.withValues(alpha: 0.2),
                        valueColor:
                            const AlwaysStoppedAnimation<Color>(monitorPrimary),
                      ),
                      const SizedBox(height: 8),
                      if (_levelBars.isNotEmpty) ...<Widget>[
                        SizedBox(
                          height: 42,
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.end,
                            children: _levelBars
                                .map(
                                  (double level) => Expanded(
                                    child: Container(
                                      margin:
                                          const EdgeInsets.symmetric(horizontal: 1),
                                      height: 8 + (level.clamp(0.0, 1.0) * 34),
                                      decoration: BoxDecoration(
                                        color: monitorPrimarySoft,
                                        borderRadius: BorderRadius.circular(3),
                                      ),
                                    ),
                                  ),
                                )
                                .toList(growable: false),
                          ),
                        ),
                        const SizedBox(height: 8),
                      ],
                      Text(
                        _recording
                            ? "Progress: ${(_progress * _recordDuration.inSeconds).clamp(0.0, _recordDuration.inSeconds.toDouble()).toStringAsFixed(1)}s / ${_recordDuration.inSeconds}s"
                            : _awaitingName
                                ? "Recording complete. Enter a name and save the sample."
                                : "Tap Record to capture a 5-second sound sample.",
                        style: const TextStyle(
                            color: monitorOnSurfaceVariant, fontSize: 12),
                      ),
                      if (_statusMessage != null) ...<Widget>[
                        const SizedBox(height: 8),
                        Text(
                          _statusMessage!,
                          style: const TextStyle(
                            color: monitorOnSurfaceVariant,
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                      if (_awaitingName) ...<Widget>[
                        const SizedBox(height: 12),
                        TextField(
                          controller: _nameController,
                          decoration: const InputDecoration(
                            labelText: "Sound name",
                            border: OutlineInputBorder(),
                          ),
                        ),
                        const SizedBox(height: 10),
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: <Widget>[
                            OutlinedButton(
                              onPressed: _cancelPendingSample,
                              child: const Text("Discard"),
                            ),
                             OutlinedButton.icon(
                               onPressed: _playPendingSample,
                               icon: Icon((_playingSoundId == "pending" && !_playbackPaused)
                                   ? Icons.pause_circle_outline
                                   : Icons.play_circle_outline),
                               label: Text((_playingSoundId == "pending" && !_playbackPaused)
                                   ? "Pause"
                                   : (_playingSoundId == "pending" && _playbackPaused)
                                       ? "Resume"
                                       : "Play"),
                             ),
                            FilledButton(
                              onPressed: _pendingPlaybackConfirmed
                                  ? () async {
                                      await _savePendingSample();
                                    }
                                  : null,
                              child: const Text("Save sample"),
                            ),
                          ],
                        ),
                      ],
                    ],
                  ),
          ),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: monitorRaised(radius: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const Text(
                  "Alert Preferences",
                  style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: monitorOnSurface),
                ),
                const SizedBox(height: 12),
                Row(
                  children: <Widget>[
                    const Expanded(
                      child: Text(
                        "Vibrate on detection",
                        style: TextStyle(
                            color: monitorOnSurface,
                            fontSize: 14,
                            fontWeight: FontWeight.w500),
                      ),
                    ),
                    Switch.adaptive(
                      value: widget.vibrationEnabled,
                      activeThumbColor: monitorPrimary,
                      activeTrackColor:
                          monitorPrimarySoft.withValues(alpha: 0.35),
                      onChanged: widget.onVibrationToggle,
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: <Widget>[
                    const Expanded(
                      child: Text(
                        "Flash camera on detection",
                        style: TextStyle(
                            color: monitorOnSurface,
                            fontSize: 14,
                            fontWeight: FontWeight.w500),
                      ),
                    ),
                    Switch.adaptive(
                      value: widget.cameraFlashEnabled,
                      activeThumbColor: monitorPrimary,
                      activeTrackColor:
                          monitorPrimarySoft.withValues(alpha: 0.35),
                      onChanged: widget.onCameraFlashToggle,
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: monitorRaised(radius: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const Text(
                  "Saved Sounds",
                  style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: monitorOnSurface),
                ),
                const SizedBox(height: 8),
                if (widget.recordedSounds.isEmpty)
                  const Text(
                    "No saved sounds yet. Record one to build your custom dataset.",
                    style: TextStyle(color: monitorOnSurfaceVariant),
                  )
                else
                  ...widget.recordedSounds.map(_buildSoundCard),
              ],
            ),
          ),
        ],
      ),
    );
  }

  List<CustomSoundSample> get _selectedClassSamples => widget.recordedSounds
      .where(
          (CustomSoundSample sample) => sample.classSlug == _selectedClassSlug)
      .toList(growable: false);

  int get _maxSamplesLimit {
    if (widget.personalisationLevel == "quick") {
      return 3;
    } else if (widget.personalisationLevel == "advanced") {
      return 5;
    } else {
      return 0;
    }
  }

  int get _minSamplesRequired {
    if (widget.personalisationLevel == "quick") {
      return 1;
    } else if (widget.personalisationLevel == "advanced") {
      return 3;
    } else {
      return 0;
    }
  }

  Widget _buildPersonalisationOption({
    required String id,
    required String title,
    required String subtitle,
    required IconData icon,
  }) {
    final bool isSelected = widget.personalisationLevel == id;
    return GestureDetector(
      onTap: () => widget.onPersonalisationLevelChanged(id),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: isSelected
            ? monitorPressed(radius: 16).copyWith(
                border: Border.all(color: monitorPrimary, width: 1.5),
              )
            : monitorRaised(radius: 16),
        child: Row(
          children: <Widget>[
            Icon(
              icon,
              color: isSelected ? monitorPrimary : monitorOnSurfaceVariant,
              size: 24,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    title,
                    style: TextStyle(
                      color: monitorOnSurface,
                      fontSize: 14,
                      fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      color: monitorOnSurfaceVariant,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
            if (isSelected)
              const Icon(
                Icons.check_circle,
                color: monitorPrimary,
                size: 20,
              ),
          ],
        ),
      ),
    );
  }

  Future<void> _showAddCustomClassDialog() async {
    final TextEditingController classController = TextEditingController();
    await showDialog<void>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: monitorSurface,
          title: const Text(
            "Add Custom Sound",
            style: TextStyle(color: monitorOnSurface),
          ),
          content: TextField(
            controller: classController,
            autofocus: true,
            decoration: const InputDecoration(
              hintText: "e.g., Microwave beep, Doorbell",
              hintStyle: TextStyle(color: monitorOnSurfaceVariant),
            ),
            style: const TextStyle(color: monitorOnSurface),
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text("Cancel"),
            ),
            FilledButton(
              onPressed: () async {
                final String label = classController.text.trim();
                if (label.isNotEmpty) {
                  await widget.onAddCustomClass(label);
                  final String slug = "custom_${label.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), '_')}";
                  if (mounted) {
                    setState(() {
                      _selectedClassSlug = slug;
                      _selectedClassLabel = label;
                    });
                  }
                }
                if (context.mounted) {
                  Navigator.of(context).pop();
                }
              },
              style: FilledButton.styleFrom(backgroundColor: monitorPrimary),
              child: const Text("Add"),
            ),
          ],
        );
      },
    );
  }

  Future<void> _startRecording() async {
    if (_recording ||
        _awaitingName ||
        widget.personalisationLevel == "default" ||
        _selectedClassSamples.length >= _maxSamplesLimit) {
      return;
    }

    final bool permissionGranted = await _audioRecorder.hasPermission();
    if (!permissionGranted) {
      _setStatus("Microphone permission is required to record sounds.");
      return;
    }

    if (_playingSoundId != null) {
      await _audioPlayer.stop();
      if (!mounted) {
        return;
      }
      setState(() {
        _playingSoundId = null;
        _playbackPaused = false;
      });
    }

    final Directory appDirectory = await getApplicationDocumentsDirectory();
    final Directory recordingsDirectory = Directory(
      "${appDirectory.path}${Platform.pathSeparator}recordings",
    );
    await recordingsDirectory.create(recursive: true);
    final String recordingPath =
        "${recordingsDirectory.path}${Platform.pathSeparator}soundpulse_${DateTime.now().millisecondsSinceEpoch}.wav";

    final DateTime startedAt = DateTime.now();
    await _audioRecorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: recordingPath,
    );

    _recordTimer?.cancel();
    if (!mounted) {
      return;
    }
    setState(() {
      _recording = true;
      _progress = 0;
      _pendingRecordedAt = startedAt;
      _pendingRecordedFilePath = recordingPath;
      _pendingPlaybackConfirmed = false;
      _levelBars = const <double>[];
      _statusMessage = null;
    });

    _recordTimer = Timer.periodic(_tickDuration, (Timer timer) async {
      final Duration elapsed = DateTime.now().difference(startedAt);
      final double nextProgress =
          elapsed.inMilliseconds / _recordDuration.inMilliseconds;
      if (nextProgress >= 1) {
        timer.cancel();
        await _finishRecording();
        return;
      }
      if (!mounted) {
        timer.cancel();
        return;
      }
      double normalizedLevel = 0.0;
      try {
        final Amplitude amplitude = await _audioRecorder.getAmplitude();
        normalizedLevel = ((amplitude.current + 60.0) / 60.0).clamp(0.0, 1.0);
      } catch (_) {
        normalizedLevel = nextProgress.clamp(0.0, 1.0);
      }
      setState(() {
        _progress = nextProgress;
        _levelBars = <double>[
          ..._levelBars.take(23),
          normalizedLevel,
        ];
      });
    });
  }

  Future<void> _finishRecording() async {
    final String? stoppedPath = await _audioRecorder.stop();
    final String? resolvedPath = stoppedPath ?? _pendingRecordedFilePath;
    if (resolvedPath == null || resolvedPath.isEmpty) {
      if (!mounted) {
        return;
      }
      setState(() {
        _recording = false;
        _progress = 0;
        _pendingRecordedAt = null;
        _pendingRecordedFilePath = null;
      });
      _setStatus("Recording failed. Please try again.");
      return;
    }

    if (!mounted) {
      return;
    }

    setState(() {
      _recording = false;
      _progress = 0;
      _awaitingName = true;
      _pendingRecordedFilePath = resolvedPath;
      _pendingRecordedAt ??= DateTime.now();
      _nameController.text =
          "${formatSoundLabel(_selectedClassLabel)} sample ${_selectedClassSamples.length + 1}";
      _statusMessage = null;
    });
  }

  Future<void> _cancelPendingSample() async {
    final String? pendingPath = _pendingRecordedFilePath;
    if (pendingPath != null && pendingPath.isNotEmpty) {
      final File pendingFile = File(pendingPath);
      if (await pendingFile.exists()) {
        await pendingFile.delete();
      }
    }

    if (_playingSoundId == "pending") {
      await _audioPlayer.stop();
    }

    if (!mounted) {
      return;
    }

    setState(() {
      _awaitingName = false;
      _nameController.clear();
      _pendingRecordedFilePath = null;
      _pendingRecordedAt = null;
      _pendingPlaybackConfirmed = false;
      _playingSoundId = null;
      _playbackPaused = false;
      _statusMessage = "Discarded recording.";
    });
  }

  Future<void> _playPendingSample() async {
    if (_playingSoundId == "pending") {
      if (_playbackPaused) {
        await _audioPlayer.resume();
        if (!mounted) return;
        setState(() {
          _playbackPaused = false;
          _statusMessage = "Playback resumed.";
        });
      } else {
        await _audioPlayer.pause();
        if (!mounted) return;
        setState(() {
          _playbackPaused = true;
          _statusMessage = "Playback paused.";
        });
      }
      return;
    }

    final String? filePath = _pendingRecordedFilePath;
    if (filePath == null || filePath.isEmpty) {
      _setStatus("No recorded sample is ready to play.");
      return;
    }
    final File file = File(filePath);
    if (!await file.exists()) {
      _setStatus("The recorded audio file is missing on this device.");
      return;
    }
    await _audioPlayer.stop();
    final File? oldTempFile = _currentTempPlayFile;
    if (oldTempFile != null && await oldTempFile.exists()) {
      await oldTempFile.delete();
    }
    _currentTempPlayFile = null;

    await _audioPlayer.play(DeviceFileSource(filePath));
    if (!mounted) {
      return;
    }
    setState(() {
      _playingSoundId = "pending";
      _playbackPaused = false;
      _pendingPlaybackConfirmed = true;
      _statusMessage =
          "Playback confirmed. Save this sample if it sounds clean.";
    });
  }

  Future<void> _savePendingSample() async {
    final String name = _nameController.text.trim();
    if (name.isEmpty) {
      return;
    }
    if (!_pendingPlaybackConfirmed) {
      _setStatus("Play the sample once before saving it.");
      return;
    }
    final String? filePath = _pendingRecordedFilePath;
    final DateTime? recordedAt = _pendingRecordedAt;
    if (filePath == null || filePath.isEmpty || recordedAt == null) {
      _setStatus("No recorded sound found to save.");
      return;
    }

    if (_playingSoundId == "pending") {
      await _audioPlayer.stop();
    }

    await widget.onSaveSound(
      name,
      _selectedClassSlug,
      _selectedClassLabel,
      filePath,
      recordedAt,
      _recordDuration.inSeconds,
    );
    if (!mounted) {
      return;
    }
    setState(() {
      _awaitingName = false;
      _nameController.clear();
      _pendingRecordedFilePath = null;
      _pendingRecordedAt = null;
      _pendingPlaybackConfirmed = false;
      _playingSoundId = null;
      _playbackPaused = false;
      _statusMessage = "Saved \"$name\".";
    });
  }

  Future<void> _saveEditedSound(CustomSoundSample sound) async {
    final String nextName = _editController.text.trim();
    if (nextName.isEmpty || nextName == sound.name) {
      setState(() {
        _editingSoundId = null;
        _editController.clear();
      });
      return;
    }
    await widget.onRenameSound(sound, nextName);
    if (!mounted) {
      return;
    }
    setState(() {
      _editingSoundId = null;
      _editController.clear();
    });
  }

  Future<void> _togglePlayback(CustomSoundSample sound) async {
    if (_playingSoundId == sound.id) {
      if (_playbackPaused) {
        await _audioPlayer.resume();
        if (!mounted) {
          return;
        }
        setState(() {
          _playbackPaused = false;
        });
      } else {
        await _audioPlayer.pause();
        if (!mounted) {
          return;
        }
        setState(() {
          _playbackPaused = true;
        });
      }
      return;
    }

    if (sound.filePath.isEmpty) {
      _setStatus("No audio file is attached to this saved sound.");
      return;
    }

    String playPath = sound.filePath;
    File? tempPlayFile;
    if (sound.filePath.endsWith(".enc") && widget.onDecryptFile != null) {
      try {
        playPath = await widget.onDecryptFile!(sound.filePath);
        tempPlayFile = File(playPath);
      } catch (e) {
        _setStatus("Failed to decrypt audio file: $e");
        return;
      }
    }

    final File file = File(playPath);
    if (!await file.exists()) {
      _setStatus("The recorded audio file is missing on this device.");
      if (tempPlayFile != null && await tempPlayFile.exists()) {
        await tempPlayFile.delete();
      }
      return;
    }

    await _audioPlayer.stop();
    final File? oldTempFile = _currentTempPlayFile;
    if (oldTempFile != null && await oldTempFile.exists()) {
      await oldTempFile.delete();
    }

    _currentTempPlayFile = tempPlayFile;
    await _audioPlayer.play(DeviceFileSource(playPath));
    if (!mounted) {
      return;
    }
    setState(() {
      _playingSoundId = sound.id;
      _playbackPaused = false;
      _statusMessage = null;
    });
  }

  void _setStatus(String message) {
    if (!mounted) {
      return;
    }
    setState(() {
      _statusMessage = message;
    });
  }

  Widget _buildSoundCard(CustomSoundSample sound) {
    final bool editing = _editingSoundId == sound.id;
    final bool playing = _playingSoundId == sound.id;
    return Container(
      margin: const EdgeInsets.only(top: 10),
      padding: const EdgeInsets.all(12),
      decoration: monitorPressed(radius: 16),
      child: Row(
        children: <Widget>[
          const Icon(Icons.audiotrack, color: monitorPrimary),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                if (editing)
                  TextField(
                    controller: _editController,
                    decoration: const InputDecoration(
                      isDense: true,
                      labelText: "Sound name",
                    ),
                  )
                else
                  Text(
                    sound.name,
                    style: const TextStyle(
                        color: monitorOnSurface, fontWeight: FontWeight.w600),
                  ),
                const SizedBox(height: 2),
                Text(
                  "${formatSoundLabel(sound.classLabel)} · ${monitorTimeAgo(sound.recordedAt)} · ${sound.durationSeconds}s",
                  style: const TextStyle(
                      color: monitorOnSurfaceVariant, fontSize: 12),
                ),
              ],
            ),
          ),
          IconButton(
            tooltip: playing
                ? (_playbackPaused ? "Resume playback" : "Pause playback")
                : "Play recording",
            onPressed: () async {
              await _togglePlayback(sound);
            },
            icon: Icon(
              (playing && !_playbackPaused)
                  ? Icons.pause_circle_outline
                  : Icons.play_circle_outline,
              color: monitorPrimary,
            ),
          ),
          if (editing)
            IconButton(
              tooltip: "Save",
              onPressed: () async {
                await _saveEditedSound(sound);
              },
              icon: const Icon(Icons.check, color: monitorPrimary),
            )
          else
            IconButton(
              tooltip: "Edit",
              onPressed: () {
                setState(() {
                  _editingSoundId = sound.id;
                  _editController.text = sound.name;
                });
              },
              icon: const Icon(Icons.edit, color: monitorOnSurfaceVariant),
            ),
          IconButton(
            tooltip: editing ? "Cancel edit" : "Delete",
            onPressed: () async {
              if (editing) {
                setState(() {
                  _editingSoundId = null;
                  _editController.clear();
                });
                return;
              }
              if (_playingSoundId == sound.id) {
                await _audioPlayer.stop();
                if (mounted) {
                  setState(() {
                    _playingSoundId = null;
                  });
                }
              }
              await widget.onDeleteSound(sound);
              if (!mounted) {
                return;
              }
              if (_editingSoundId == sound.id) {
                setState(() {
                  _editingSoundId = null;
                  _editController.clear();
                });
              }
              setState(() {
                _statusMessage = "Deleted \"${sound.name}\".";
              });
            },
            icon: Icon(
              editing ? Icons.close : Icons.delete_outline,
              color:
                  editing ? monitorOnSurfaceVariant : const Color(0xFFB00020),
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _recordTimer?.cancel();
    _playerCompleteSub?.cancel();
    final File? tempFile = _currentTempPlayFile;
    if (tempFile != null && tempFile.existsSync()) {
      try {
        tempFile.deleteSync();
      } catch (_) {}
    }
    unawaited(_audioRecorder.dispose());
    unawaited(_audioPlayer.dispose());
    _nameController.dispose();
    _editController.dispose();
    super.dispose();
  }
}
