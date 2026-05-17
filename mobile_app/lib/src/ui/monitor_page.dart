import "dart:async";
import "dart:io";

import "package:flutter/material.dart";

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

  StreamSubscription<InferenceEvent>? _eventsSub;

  bool _monitoring = false;
  bool _contextAware = false;
  String _mode = "general";
  int _tabIndex = 0;

  @override
  void initState() {
    super.initState();
    _loadPreferences();
  }

  Future<void> _loadPreferences() async {
    String storedMode = "general";
    bool storedContextAware = false;
    List<CustomSoundSample> storedSounds = const <CustomSoundSample>[];

    try {
      storedMode = await _storageService.readMode();
      storedContextAware = await _storageService.readContextAware();
      storedSounds = await _storageService.readRecordedSounds();
    } catch (_) {
      storedMode = "general";
      storedContextAware = false;
      storedSounds = const <CustomSoundSample>[];
    }

    if (!mounted) {
      return;
    }

    setState(() {
      _mode = storedMode;
      _contextAware = storedContextAware;
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

    await _runtimeBridge.startMonitoring(mode: _mode);
    _eventsSub = _runtimeBridge.streamEvents().listen(_onEvent);

    if (!mounted) {
      return;
    }

    setState(() {
      _monitoring = true;
    });
  }

  void _onEvent(InferenceEvent event) {
    if (!mounted) {
      return;
    }

    setState(() {
      if (!event.isBackground && !event.isUnknown) {
        _history.insert(
            0, MonitorHistoryEntry(event: event, capturedAt: DateTime.now()));
        if (_history.length > 20) {
          _history.removeLast();
        }
      }
    });
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
                  contextAware: _contextAware,
                  monitoring: _monitoring,
                  history: _history,
                  onModeChanged: _changeMode,
                  onContextAwareChanged: _setContextAware,
                  onViewAll: () => setState(() => _tabIndex = 1),
                ),
                MonitorHistoryTab(history: _history),
                MonitorRecordTab(
                  recordedSounds: _recordedSounds,
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
