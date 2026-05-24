import "dart:async";
import "dart:io";
import "dart:math" as math;

import "package:audioplayers/audioplayers.dart";
import "package:flutter/foundation.dart";
import "package:flutter/material.dart";

import "../models/monitor_history_entry.dart";
import "monitor_style.dart";

class MonitorLiveFeedTab extends StatefulWidget {
  const MonitorLiveFeedTab({
    super.key,
    required this.mode,
    required this.backbone,
    required this.contextAware,
    required this.monitoring,
    required this.history,
    required this.onModeChanged,
    required this.onBackboneChanged,
    required this.onContextAwareChanged,
    required this.onViewAll,
  });

  final String mode;
  final String backbone;
  final bool contextAware;
  final bool monitoring;
  final List<MonitorHistoryEntry> history;
  final Future<void> Function(String mode) onModeChanged;
  final Future<void> Function(String backbone) onBackboneChanged;
  final Future<void> Function(bool enabled) onContextAwareChanged;
  final VoidCallback onViewAll;

  @override
  State<MonitorLiveFeedTab> createState() => _MonitorLiveFeedTabState();
}

class _MonitorLiveFeedTabState extends State<MonitorLiveFeedTab> {
  final AudioPlayer _audioPlayer = AudioPlayer();
  StreamSubscription<void>? _playerCompleteSub;
  String? _playingEntryId;
  String? _statusMessage;

  @override
  void initState() {
    super.initState();
    _playerCompleteSub = _audioPlayer.onPlayerComplete.listen((_) {
      if (!mounted) {
        return;
      }
      setState(() {
        _playingEntryId = null;
      });
    });
  }

  @override
  void didUpdateWidget(covariant MonitorLiveFeedTab oldWidget) {
    super.didUpdateWidget(oldWidget);
    final String? playingEntryId = _playingEntryId;
    if (playingEntryId == null) {
      return;
    }
    final bool stillExists = widget.history.any(
      (MonitorHistoryEntry entry) => entry.id == playingEntryId,
    );
    if (!stillExists) {
      unawaited(_audioPlayer.stop());
      _playingEntryId = null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Container(
            padding: const EdgeInsets.all(16),
            decoration: monitorRaised(radius: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const Text(
                  "Live Feed Settings",
                  style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: monitorOnSurface),
                ),
                const SizedBox(height: 12),
                const Text(
                  "Mode",
                  style: TextStyle(
                      color: monitorOnSurfaceVariant,
                      fontSize: 12,
                      fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 10),
                Row(
                  children: <Widget>[
                    _modeChip("general"),
                    const SizedBox(width: 8),
                    _modeChip("focus"),
                    const SizedBox(width: 8),
                    _modeChip("sleep"),
                  ],
                ),
                const SizedBox(height: 14),
                const Text(
                  "Active ML Model",
                  style: TextStyle(
                      color: monitorOnSurfaceVariant,
                      fontSize: 12,
                      fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 10),
                Row(
                  children: <Widget>[
                    _backboneChip("tiny_cnn", "TinyCNN (256-D)"),
                    const SizedBox(width: 8),
                    _backboneChip("yamnet", "YAMNet (1024-D)"),
                  ],
                ),
                const SizedBox(height: 14),
                Row(
                  children: <Widget>[
                    const Expanded(
                      child: Text(
                        "Context-aware routing",
                        style: TextStyle(
                            color: monitorOnSurface,
                            fontSize: 14,
                            fontWeight: FontWeight.w500),
                      ),
                    ),
                    Switch.adaptive(
                      value: widget.contextAware,
                      activeThumbColor: monitorPrimary,
                      activeTrackColor:
                          monitorPrimarySoft.withValues(alpha: 0.35),
                      onChanged: (bool enabled) async {
                        await widget.onContextAwareChanged(enabled);
                      },
                    ),
                  ],
                ),
              ],
            ),
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
          const SizedBox(height: 14),
          Row(
            children: <Widget>[
              const Text(
                "View Recent Sounds",
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: monitorOnSurface),
              ),
              const Spacer(),
              TextButton(
                onPressed: widget.onViewAll,
                child: const Text("View All",
                    style: TextStyle(color: monitorPrimary, fontSize: 12)),
              ),
            ],
          ),
          if (widget.history.isEmpty)
            Container(
              width: double.infinity,
              margin: const EdgeInsets.only(top: 8),
              padding: const EdgeInsets.all(20),
              decoration: monitorRaised(radius: 20),
              child: Text(
                widget.monitoring
                    ? "Listening for sounds..."
                    : "Start monitoring to detect sounds",
                style: const TextStyle(color: monitorOnSurfaceVariant),
              ),
            )
          else
            ...widget.history.take(6).map(_buildHistoryCard),
        ],
      ),
    );
  }

  Widget _modeChip(String candidateMode) {
    final bool selected = widget.mode == candidateMode;
    return Expanded(
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () async {
          await widget.onModeChanged(candidateMode);
        },
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 11),
          decoration:
              selected ? monitorPressed(radius: 12) : monitorRaised(radius: 12),
          child: Text(
            monitorLabelForMode(candidateMode),
            textAlign: TextAlign.center,
            style: TextStyle(
              color: selected ? monitorPrimary : monitorOnSurfaceVariant,
              fontWeight: FontWeight.w600,
              fontSize: 12,
            ),
          ),
        ),
      ),
    );
  }

  Widget _backboneChip(String candidateBackbone, String label) {
    final bool selected = widget.backbone == candidateBackbone;
    return Expanded(
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () async {
          await widget.onBackboneChanged(candidateBackbone);
        },
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 11),
          decoration:
              selected ? monitorPressed(radius: 12) : monitorRaised(radius: 12),
          child: Text(
            label,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: selected ? monitorPrimary : monitorOnSurfaceVariant,
              fontWeight: FontWeight.w600,
              fontSize: 12,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHistoryCard(MonitorHistoryEntry entry) {
    final String confidence = "${(entry.event.confidence * 100).round()}%";
    final bool playing = _playingEntryId == entry.id;
    return Container(
      margin: const EdgeInsets.only(top: 12),
      padding: const EdgeInsets.all(16),
      decoration: monitorRaised(radius: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              Expanded(
                child: Text(
                  entry.event.label,
                  style: const TextStyle(
                      color: monitorPrimary,
                      fontSize: 17,
                      fontWeight: FontWeight.w600),
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: <Widget>[
                  Text(
                    monitorTimeAgo(entry.capturedAt),
                    style: const TextStyle(
                        color: monitorOnSurfaceVariant,
                        fontSize: 10,
                        fontWeight: FontWeight.w500),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    "$confidence Certainty",
                    style: const TextStyle(
                        color: monitorOnSurface,
                        fontSize: 12,
                        fontWeight: FontWeight.w700),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: monitorPressed(radius: 14),
            child: SizedBox(
              height: 52,
              width: double.infinity,
              child: entry.waveformSamples.isEmpty
                  ? const Center(
                      child: Text(
                        "Waveform unavailable",
                        style: TextStyle(
                            color: monitorOnSurfaceVariant, fontSize: 11),
                      ),
                    )
                  : CustomPaint(
                      painter: _WaveformPainter(entry.waveformSamples),
                    ),
            ),
          ),
          const SizedBox(height: 6),
          Align(
            alignment: Alignment.centerRight,
            child: IconButton(
              tooltip: playing ? "Stop playback" : "Play detection",
              onPressed: () async {
                await _togglePlayback(entry);
              },
              icon: Icon(
                playing
                    ? Icons.stop_circle_outlined
                    : Icons.play_circle_outline,
                color: monitorPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _togglePlayback(MonitorHistoryEntry entry) async {
    if (_playingEntryId == entry.id) {
      await _audioPlayer.stop();
      if (!mounted) {
        return;
      }
      setState(() {
        _playingEntryId = null;
      });
      return;
    }

    if (entry.audioFilePath.isEmpty) {
      _setStatus("No audio clip is attached to this detection.");
      return;
    }

    final File file = File(entry.audioFilePath);
    if (!await file.exists()) {
      _setStatus("The saved audio clip for this detection is missing.");
      return;
    }

    await _audioPlayer.stop();
    await _audioPlayer.play(DeviceFileSource(entry.audioFilePath));
    if (!mounted) {
      return;
    }
    setState(() {
      _playingEntryId = entry.id;
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

  @override
  void dispose() {
    _playerCompleteSub?.cancel();
    unawaited(_audioPlayer.dispose());
    super.dispose();
  }
}

class _WaveformPainter extends CustomPainter {
  const _WaveformPainter(this.samples);

  final List<double> samples;

  @override
  void paint(Canvas canvas, Size size) {
    if (samples.isEmpty || size.width <= 0 || size.height <= 0) {
      return;
    }

    final Paint baselinePaint = Paint()
      ..color = monitorOnSurfaceVariant.withValues(alpha: 0.25)
      ..strokeWidth = 1;
    final Paint waveformPaint = Paint()
      ..color = monitorPrimary.withValues(alpha: 0.85)
      ..strokeWidth = 1.8
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    final double centerY = size.height / 2;

    canvas.drawLine(
        Offset(0, centerY), Offset(size.width, centerY), baselinePaint);

    final int points = math.max(2, size.width.floor());
    final Path path = Path();
    for (int i = 0; i < points; i += 1) {
      final double x = (i / (points - 1)) * size.width;
      final int sampleIndex =
          ((i / (points - 1)) * (samples.length - 1)).round();
      final double sample = samples[sampleIndex].clamp(-1.0, 1.0);
      final double y = centerY - (sample * centerY * 0.92);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(path, waveformPaint);
  }

  @override
  bool shouldRepaint(covariant _WaveformPainter oldDelegate) {
    return !listEquals(oldDelegate.samples, samples);
  }
}
