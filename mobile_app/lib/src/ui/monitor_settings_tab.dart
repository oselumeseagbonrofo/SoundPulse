import "dart:async";
import "package:flutter/material.dart";
import "../storage/local_storage_service.dart";
import "../runtime/runtime_bridge.dart";

import "monitor_style.dart";

class MonitorSettingsTab extends StatefulWidget {
  const MonitorSettingsTab({
    super.key,
    required this.activeMode,
    required this.contextAware,
    required this.onModeChanged,
    required this.onContextAwareChanged,
    required this.onBack,
  });

  final String activeMode;
  final bool contextAware;
  final Future<void> Function(String mode) onModeChanged;
  final Future<void> Function(bool enabled) onContextAwareChanged;
  final VoidCallback onBack;

  @override
  State<MonitorSettingsTab> createState() => _MonitorSettingsTabState();
}

class _MonitorSettingsTabState extends State<MonitorSettingsTab> {
  final LocalStorageService _storageService = LocalStorageService();
  final RuntimeBridge _runtimeBridge = RuntimeBridge();

  List<Map<String, dynamic>> _configs = <Map<String, dynamic>>[];
  bool _loading = true;

  // Track which sound cards are expanded
  final Set<String> _expandedSlugs = <String>{};

  // vibration options
  static const List<Map<String, String>> _vibrationOptions = [
    {"value": "single", "label": "Single Short Buzz"},
    {"value": "double", "label": "Double Pulse Buzz"},
    {"value": "triple", "label": "Triple Pulse Buzz"},
    {"value": "long", "label": "Long Continuous Buzz"},
    {"value": "none", "label": "No Haptic Vibration"},
  ];

  // flash options
  static const List<Map<String, String>> _flashOptions = [
    {"value": "single", "label": "Single Light Flash"},
    {"value": "double", "label": "Double Light Flash"},
    {"value": "strobe", "label": "Rapid Strobe Flashes"},
    {"value": "none", "label": "No Flash Light"},
  ];

  // System Modes List
  static const List<Map<String, String>> _systemModes = [
    {"slug": "general", "label": "General"},
    {"slug": "focus", "label": "Focus"},
    {"slug": "sleep", "label": "Sleep"},
    {"slug": "cooking", "label": "Cooking"},
    {"slug": "bathroom", "label": "Bathroom"},
    {"slug": "no_context", "label": "No Context"},
  ];

  @override
  void initState() {
    super.initState();
    unawaited(_loadConfigs());
  }

  Map<String, bool> _getInitialAllowValues(String slug) {
    if (slug == "fire_alarm" || slug == "clock_alarm") {
      return {
        "allow_in_sleep_mode": true,
        "allow_in_focus_mode": true,
        "allow_in_cooking_mode": true,
        "allow_in_bathroom_mode": true,
      };
    } else if (slug == "water_running") {
      return {
        "allow_in_sleep_mode": false,
        "allow_in_focus_mode": false,
        "allow_in_cooking_mode": true,
        "allow_in_bathroom_mode": false,
      };
    } else if (slug == "door_knock" || slug == "doorbell" || slug == "speech") {
      return {
        "allow_in_sleep_mode": true,
        "allow_in_focus_mode": false,
        "allow_in_cooking_mode": true,
        "allow_in_bathroom_mode": true,
      };
    } else if (slug == "crying_baby" || slug == "dog_barking" || slug == "door_wood_creaking") {
      return {
        "allow_in_sleep_mode": true,
        "allow_in_focus_mode": false,
        "allow_in_cooking_mode": false,
        "allow_in_bathroom_mode": true,
      };
    } else {
      // Custom classes defaults
      return {
        "allow_in_sleep_mode": true,
        "allow_in_focus_mode": false,
        "allow_in_cooking_mode": false,
        "allow_in_bathroom_mode": true,
      };
    }
  }

  Future<void> _loadConfigs() async {
    final List<Map<String, dynamic>> loadedConfigs = <Map<String, dynamic>>[];

    // 1. Core/Standard sound classes
    final List<Map<String, String>> standards = [
      {"slug": "fire_alarm", "label": "Fire alarm"},
      {"slug": "clock_alarm", "label": "Clock alarm"},
      {"slug": "door_knock", "label": "Door knock"},
      {"slug": "doorbell", "label": "Doorbell"},
      {"slug": "crying_baby", "label": "Crying baby"},
      {"slug": "water_running", "label": "Water running"},
      {"slug": "dog_barking", "label": "Dog barking"},
      {"slug": "door_wood_creaking", "label": "Door opening"},
      {"slug": "speech", "label": "Speech"},
    ];

    // 2. Custom sound classes created by user
    final List<Map<String, dynamic>> customConfigs =
        await _storageService.readAllClassConfigs();

    // 3. Sync and build consolidated settings model
    for (final standard in standards) {
      final String slug = standard["slug"]!;
      final String label = standard["label"]!;
      Map<String, dynamic>? config = await _storageService.readClassConfig(slug);
      bool needsSave = false;

      if (config == null) {
        config = <String, dynamic>{
          "slug": slug,
          "label": label,
          "enabled": true,
          "is_custom": false,
          "modes": <String, String>{},
          "custom_vibration": "single",
          "custom_flash": "single",
          ..._getInitialAllowValues(slug),
        };
        needsSave = true;
      } else {
        // Enforce fallback structure just in case
        config["enabled"] ??= true;
        config["is_custom"] = false;
        config["modes"] ??= <String, String>{};
        config["custom_vibration"] ??= "single";
        config["custom_flash"] ??= "single";

        final defaults = _getInitialAllowValues(slug);
        defaults.forEach((key, val) {
          if (config![key] == null) {
            config[key] = val;
            needsSave = true;
          }
        });
      }

      if (needsSave) {
        await _storageService.saveClassConfig(slug, config);
      }
      loadedConfigs.add(config);
    }

    for (final custom in customConfigs) {
      final String slug = custom["slug"] as String? ?? "";
      if (slug.isEmpty) continue;
      if (standards.any((s) => s["slug"] == slug)) continue;

      Map<String, dynamic>? config = await _storageService.readClassConfig(slug);
      bool needsSave = false;

      if (config == null) {
        config = <String, dynamic>{
          "slug": slug,
          "label": custom["label"] as String? ?? formatSoundLabel(slug),
          "enabled": true,
          "is_custom": true,
          "modes": <String, String>{},
          "custom_vibration": "single",
          "custom_flash": "single",
          ..._getInitialAllowValues(slug),
        };
        needsSave = true;
      } else {
        config["enabled"] ??= true;
        config["is_custom"] = true;
        config["modes"] ??= <String, String>{};
        config["custom_vibration"] ??= "single";
        config["custom_flash"] ??= "single";

        final defaults = _getInitialAllowValues(slug);
        defaults.forEach((key, val) {
          if (config![key] == null) {
            config[key] = val;
            needsSave = true;
          }
        });
      }

      if (needsSave) {
        await _storageService.saveClassConfig(slug, config);
      }
      loadedConfigs.add(config);
    }

    if (mounted) {
      setState(() {
        _configs = loadedConfigs;
        _loading = false;
      });
    }
  }

  Future<void> _updateConfig(String slug, Map<String, dynamic> updated) async {
    await _storageService.saveClassConfig(slug, updated);
    // Reload dynamically to ensure parity
    await _loadConfigs();
  }

  Widget _buildModeChip(String modeSlug, String label) {
    final bool active = widget.activeMode == modeSlug;
    return GestureDetector(
      onTap: () => widget.onModeChanged(modeSlug),
      child: Container(
        margin: const EdgeInsets.only(right: 8),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: active ? monitorPressed(radius: 20) : monitorRaised(radius: 20),
        child: Text(
          label,
          style: TextStyle(
            color: active ? monitorPrimary : monitorOnSurfaceVariant,
            fontWeight: active ? FontWeight.w700 : FontWeight.w600,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  IconData _iconForSound(String slug) {
    if (slug.contains("fire") || slug.contains("smoke")) {
      return Icons.local_fire_department;
    }
    if (slug.contains("clock") || slug.contains("alarm")) {
      return Icons.alarm;
    }
    if (slug.contains("door") || slug.contains("knock") || slug.contains("bell")) {
      return Icons.door_sliding;
    }
    if (slug.contains("baby") || slug.contains("cry")) {
      return Icons.child_care;
    }
    if (slug.contains("water")) {
      return Icons.water_drop;
    }
    if (slug.contains("dog") || slug.contains("bark")) {
      return Icons.pets;
    }
    if (slug.contains("speech") || slug.contains("voice")) {
      return Icons.record_voice_over;
    }
    return Icons.waves;
  }

  Widget _buildModeAllowSwitch({
    required String modeSlug,
    required bool allowed,
    required void Function(bool newValue) onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: <Widget>[
          Text(
            "Enable in ${monitorLabelForMode(modeSlug)} Mode",
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: monitorOnSurface,
            ),
          ),
          Switch.adaptive(
            value: allowed,
            activeThumbColor: monitorPrimary,
            activeTrackColor: monitorPrimarySoft.withValues(alpha: 0.35),
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }

  Widget _buildPatternSelector({
    required String label,
    required String activeValue,
    required List<Map<String, String>> options,
    required void Function(String newValue) onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(
          label,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: monitorOnSurfaceVariant,
          ),
        ),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          decoration: monitorPressed(radius: 12),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: activeValue,
              isExpanded: true,
              icon: const Icon(Icons.arrow_drop_down, color: monitorOnSurfaceVariant),
              dropdownColor: monitorSurface,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: monitorOnSurface,
              ),
              items: options.map((opt) {
                return DropdownMenuItem<String>(
                  value: opt["value"],
                  child: Text(opt["label"]!),
                );
              }).toList(),
              onChanged: (String? val) {
                if (val != null) onChanged(val);
              },
            ),
          ),
        ),
        const SizedBox(height: 12),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: monitorSurface,
      appBar: AppBar(
        backgroundColor: monitorSurface,
        elevation: 0,
        title: const Text(
          "Settings & Context Rules",
          style: TextStyle(
            color: monitorOnSurface,
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  // 1. Context Mode Routing Neumorphic Card
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: monitorRaised(radius: 20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: <Widget>[
                        Row(
                          children: <Widget>[
                            const Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: <Widget>[
                                  Text(
                                    "Context-Aware Routing",
                                    style: TextStyle(
                                      color: monitorOnSurface,
                                      fontSize: 14,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                  SizedBox(height: 2),
                                  Text(
                                    "Dynamically filters non-urgent alert noise based on what you are doing.",
                                    style: TextStyle(
                                      color: monitorOnSurfaceVariant,
                                      fontSize: 11,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            Switch.adaptive(
                              value: widget.contextAware,
                              activeThumbColor: monitorPrimary,
                              activeTrackColor: monitorPrimarySoft.withValues(alpha: 0.35),
                              onChanged: widget.onContextAwareChanged,
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          "Active Context Mode",
                          style: TextStyle(
                            color: monitorOnSurfaceVariant,
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 8),
                        SingleChildScrollView(
                          scrollDirection: Axis.horizontal,
                          child: Row(
                            children: _systemModes.map((mode) {
                              return _buildModeChip(mode["slug"]!, mode["label"]!);
                            }).toList(),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),

                  // 2. Sound Detections Gating Header
                  const Text(
                    "Sound Detection Preferences",
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: monitorOnSurface,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    "Select which sounds the app should listen for, choose your custom alert modalities, and define specific haptic/visual patterns.",
                    style: TextStyle(
                      fontSize: 11,
                      color: monitorOnSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 14),

                  // 3. Sound List Cards
                  ..._configs.map((config) {
                    final String slug = config["slug"] as String;
                    final String label = config["label"] as String;
                    final bool enabled = config["enabled"] as bool? ?? true;
                    final bool isCustom = config["is_custom"] as bool? ?? false;
                    final String customVib = config["custom_vibration"] as String? ?? "single";
                    final String customFls = config["custom_flash"] as String? ?? "single";
                    final bool expanded = _expandedSlugs.contains(slug);

                    return Container(
                      margin: const EdgeInsets.only(bottom: 16),
                      decoration: monitorRaised(radius: 20),
                      child: Column(
                        children: <Widget>[
                          // Sound Card Header Row
                          InkWell(
                            onTap: () {
                              setState(() {
                                if (expanded) {
                                  _expandedSlugs.remove(slug);
                                } else {
                                  _expandedSlugs.add(slug);
                                }
                              });
                            },
                            borderRadius: BorderRadius.circular(20),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                              child: Row(
                                children: <Widget>[
                                  Container(
                                    width: 40,
                                    height: 40,
                                    decoration: monitorPressed(radius: 12),
                                    child: Icon(
                                      _iconForSound(slug),
                                      color: enabled ? monitorPrimary : monitorOnSurfaceVariant,
                                      size: 20,
                                    ),
                                  ),
                                  const SizedBox(width: 14),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: <Widget>[
                                        Row(
                                          children: <Widget>[
                                            Text(
                                              label,
                                              style: TextStyle(
                                                color: enabled ? monitorOnSurface : monitorOnSurfaceVariant,
                                                fontSize: 14,
                                                fontWeight: FontWeight.w700,
                                              ),
                                            ),
                                            if (isCustom) ...[
                                              const SizedBox(width: 6),
                                              Container(
                                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                                decoration: BoxDecoration(
                                                  color: monitorPrimarySoft.withValues(alpha: 0.15),
                                                  borderRadius: BorderRadius.circular(6),
                                                ),
                                                child: const Text(
                                                  "CUSTOM",
                                                  style: TextStyle(
                                                    color: monitorPrimary,
                                                    fontSize: 8,
                                                    fontWeight: FontWeight.w700,
                                                  ),
                                                ),
                                              ),
                                            ],
                                          ],
                                        ),
                                        const SizedBox(height: 2),
                                        Text(
                                          enabled ? "Listening active" : "Detection off",
                                          style: TextStyle(
                                            color: enabled ? const Color(0xFF10B981) : monitorOnSurfaceVariant,
                                            fontSize: 10,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  Switch.adaptive(
                                    value: enabled,
                                    activeThumbColor: monitorPrimary,
                                    activeTrackColor: monitorPrimarySoft.withValues(alpha: 0.35),
                                    onChanged: (bool nextVal) {
                                      final Map<String, dynamic> nextConfig =
                                          Map<String, dynamic>.from(config);
                                      nextConfig["enabled"] = nextVal;
                                      unawaited(_updateConfig(slug, nextConfig));
                                    },
                                  ),
                                ],
                              ),
                            ),
                          ),

                          // Sound Card Expanded Section
                          if (expanded) ...[
                            const Divider(height: 1, color: Color(0x1F000000)),
                            Padding(
                              padding: const EdgeInsets.all(16),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: <Widget>[
                                  const Text(
                                    "ENABLE SOUND IN CONTEXT MODES",
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.w800,
                                      letterSpacing: 1.0,
                                      color: monitorPrimary,
                                    ),
                                  ),
                                  const SizedBox(height: 12),
                                  _buildModeAllowSwitch(
                                    modeSlug: "sleep",
                                    allowed: config["allow_in_sleep_mode"] as bool? ?? true,
                                    onChanged: (bool nextVal) {
                                      final Map<String, dynamic> nextConfig =
                                          Map<String, dynamic>.from(config);
                                      nextConfig["allow_in_sleep_mode"] = nextVal;
                                      unawaited(_updateConfig(slug, nextConfig));
                                    },
                                  ),
                                  _buildModeAllowSwitch(
                                    modeSlug: "focus",
                                    allowed: config["allow_in_focus_mode"] as bool? ?? true,
                                    onChanged: (bool nextVal) {
                                      final Map<String, dynamic> nextConfig =
                                          Map<String, dynamic>.from(config);
                                      nextConfig["allow_in_focus_mode"] = nextVal;
                                      unawaited(_updateConfig(slug, nextConfig));
                                    },
                                  ),
                                  _buildModeAllowSwitch(
                                    modeSlug: "cooking",
                                    allowed: config["allow_in_cooking_mode"] as bool? ?? true,
                                    onChanged: (bool nextVal) {
                                      final Map<String, dynamic> nextConfig =
                                          Map<String, dynamic>.from(config);
                                      nextConfig["allow_in_cooking_mode"] = nextVal;
                                      unawaited(_updateConfig(slug, nextConfig));
                                    },
                                  ),
                                  _buildModeAllowSwitch(
                                    modeSlug: "bathroom",
                                    allowed: config["allow_in_bathroom_mode"] as bool? ?? true,
                                    onChanged: (bool nextVal) {
                                      final Map<String, dynamic> nextConfig =
                                          Map<String, dynamic>.from(config);
                                      nextConfig["allow_in_bathroom_mode"] = nextVal;
                                      unawaited(_updateConfig(slug, nextConfig));
                                    },
                                  ),
                                  const SizedBox(height: 8),

                                  const Text(
                                    "CUSTOM HAPTIC & VISUAL CUES",
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.w800,
                                      letterSpacing: 1.0,
                                      color: monitorPrimary,
                                    ),
                                  ),
                                  const SizedBox(height: 12),
                                  _buildPatternSelector(
                                    label: "Custom Vibration Pattern",
                                    activeValue: customVib,
                                    options: _vibrationOptions,
                                    onChanged: (String newVal) {
                                      final Map<String, dynamic> nextConfig =
                                          Map<String, dynamic>.from(config);
                                      nextConfig["custom_vibration"] = newVal;
                                      unawaited(_updateConfig(slug, nextConfig));
                                    },
                                  ),
                                  _buildPatternSelector(
                                    label: "Custom Camera Flash Pattern",
                                    activeValue: customFls,
                                    options: _flashOptions,
                                    onChanged: (String newVal) async {
                                      final Map<String, dynamic> nextConfig =
                                          Map<String, dynamic>.from(config);
                                      nextConfig["custom_flash"] = newVal;
                                      await _updateConfig(slug, nextConfig);
                                      if (newVal != "none") {
                                        final bool hasCam = await _runtimeBridge.hasCameraPermission();
                                        if (!hasCam) {
                                          await _runtimeBridge.requestCameraPermission();
                                        }
                                      }
                                    },
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ],
                      ),
                    );
                  }),
                ],
              ),
            ),
    );
  }
}
