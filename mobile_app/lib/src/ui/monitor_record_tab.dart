import "dart:async";
import "dart:io";

import "package:audioplayers/audioplayers.dart";
import "package:flutter/material.dart";
import "package:path_provider/path_provider.dart";
import "package:record/record.dart";

import "../models/custom_sound_sample.dart";
import "monitor_style.dart";

class MonitorRecordTab extends StatefulWidget {
  const MonitorRecordTab({
    super.key,
    required this.recordedSounds,
    required this.vibrationEnabled,
    required this.cameraFlashEnabled,
    required this.onVibrationToggle,
    required this.onCameraFlashToggle,
    required this.onSaveSound,
    required this.onRenameSound,
    required this.onDeleteSound,
  });

  final List<CustomSoundSample> recordedSounds;
  final bool vibrationEnabled;
  final bool cameraFlashEnabled;
  final ValueChanged<bool> onVibrationToggle;
  final ValueChanged<bool> onCameraFlashToggle;
  final Future<void> Function(
    String name,
    String filePath,
    DateTime recordedAt,
    int durationSeconds,
  ) onSaveSound;
  final Future<void> Function(CustomSoundSample sound, String name)
      onRenameSound;
  final Future<void> Function(CustomSoundSample sound) onDeleteSound;

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
  String? _statusMessage;
  final AudioRecorder _audioRecorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _editController = TextEditingController();
  StreamSubscription<void>? _playerCompleteSub;

  @override
  void initState() {
    super.initState();
    _playerCompleteSub = _audioPlayer.onPlayerComplete.listen((_) {
      if (!mounted) {
        return;
      }
      setState(() {
        _playingSoundId = null;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
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
          const Text(
            "Capture 5-second sound samples for customizing your on-device model.",
            style: TextStyle(color: monitorOnSurfaceVariant, fontSize: 12),
          ),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: monitorRaised(radius: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Row(
                  children: <Widget>[
                    Expanded(
                      child: Text(
                        _recording
                            ? "Recording in progress..."
                            : _awaitingName
                                ? "Name your new sample to save it"
                                : "Ready to record a new sample",
                        style: const TextStyle(
                            color: monitorOnSurface,
                            fontSize: 14,
                            fontWeight: FontWeight.w600),
                      ),
                    ),
                    ElevatedButton.icon(
                      onPressed:
                          _recording || _awaitingName ? null : _startRecording,
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
                LinearProgressIndicator(
                  value: _recording ? _progress : 0,
                  minHeight: 10,
                  borderRadius: BorderRadius.circular(8),
                  backgroundColor: monitorPrimarySoft.withValues(alpha: 0.2),
                  valueColor:
                      const AlwaysStoppedAnimation<Color>(monitorPrimary),
                ),
                const SizedBox(height: 8),
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
                  Row(
                    children: <Widget>[
                      OutlinedButton(
                        onPressed: _cancelPendingSample,
                        child: const Text("Discard"),
                      ),
                      const SizedBox(width: 8),
                      FilledButton(
                        onPressed: () async {
                          await _savePendingSample();
                        },
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

  Future<void> _startRecording() async {
    if (_recording || _awaitingName) {
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
      });
    }

    final Directory appDirectory = await getApplicationDocumentsDirectory();
    final Directory recordingsDirectory = Directory(
      "${appDirectory.path}${Platform.pathSeparator}recordings",
    );
    await recordingsDirectory.create(recursive: true);
    final String recordingPath =
        "${recordingsDirectory.path}${Platform.pathSeparator}soundpulse_${DateTime.now().millisecondsSinceEpoch}.m4a";

    final DateTime startedAt = DateTime.now();
    await _audioRecorder.start(
      const RecordConfig(
        encoder: AudioEncoder.aacLc,
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
      setState(() {
        _progress = nextProgress;
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
      _nameController.text = "Sound ${widget.recordedSounds.length + 1}";
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

    if (!mounted) {
      return;
    }

    setState(() {
      _awaitingName = false;
      _nameController.clear();
      _pendingRecordedFilePath = null;
      _pendingRecordedAt = null;
      _statusMessage = "Discarded recording.";
    });
  }

  Future<void> _savePendingSample() async {
    final String name = _nameController.text.trim();
    if (name.isEmpty) {
      return;
    }
    final String? filePath = _pendingRecordedFilePath;
    final DateTime? recordedAt = _pendingRecordedAt;
    if (filePath == null || filePath.isEmpty || recordedAt == null) {
      _setStatus("No recorded sound found to save.");
      return;
    }
    await widget.onSaveSound(
      name,
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
      await _audioPlayer.stop();
      if (!mounted) {
        return;
      }
      setState(() {
        _playingSoundId = null;
      });
      return;
    }

    if (sound.filePath.isEmpty) {
      _setStatus("No audio file is attached to this saved sound.");
      return;
    }

    final File file = File(sound.filePath);
    if (!await file.exists()) {
      _setStatus("The recorded audio file is missing on this device.");
      return;
    }

    await _audioPlayer.stop();
    await _audioPlayer.play(DeviceFileSource(sound.filePath));
    if (!mounted) {
      return;
    }
    setState(() {
      _playingSoundId = sound.id;
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
                  "${monitorTimeAgo(sound.recordedAt)} · ${sound.durationSeconds}s",
                  style: const TextStyle(
                      color: monitorOnSurfaceVariant, fontSize: 12),
                ),
              ],
            ),
          ),
          IconButton(
            tooltip: playing ? "Stop playback" : "Play recording",
            onPressed: () async {
              await _togglePlayback(sound);
            },
            icon: Icon(
              playing ? Icons.stop_circle_outlined : Icons.play_circle_outline,
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
    unawaited(_audioRecorder.dispose());
    unawaited(_audioPlayer.dispose());
    _nameController.dispose();
    _editController.dispose();
    super.dispose();
  }
}
