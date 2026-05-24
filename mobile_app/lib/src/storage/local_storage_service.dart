import "package:hive/hive.dart";

import "../models/custom_sound_sample.dart";

/// Persists user preferences and user-recorded sounds in Hive.
class LocalStorageService {
  LocalStorageService();

  static const String _settingsBoxName = "soundpulse.settings";
  static const String _recordingsBoxName = "soundpulse.recordings";
  static const String _modeKey = "soundpulse.mode";
  static const String _contextAwareKey = "soundpulse.context_aware";
  static const String _backboneKey = "soundpulse.backbone";
  static const String _recordedSoundsKey = "soundpulse.recorded_sounds";

  Box<dynamic>? _settingsBox() {
    if (!Hive.isBoxOpen(_settingsBoxName)) {
      return null;
    }

    return Hive.box<dynamic>(_settingsBoxName);
  }

  Box<dynamic>? _recordingsBox() {
    if (!Hive.isBoxOpen(_recordingsBoxName)) {
      return null;
    }

    return Hive.box<dynamic>(_recordingsBoxName);
  }

  Future<void> saveMode(String mode) {
    return Future.sync(() => _settingsBox()?.put(_modeKey, mode));
  }

  Future<String> readMode() async {
    final box = _settingsBox();
    return box?.get(_modeKey, defaultValue: "general") as String? ?? "general";
  }

  Future<void> saveContextAware(bool enabled) {
    return Future.sync(() => _settingsBox()?.put(_contextAwareKey, enabled));
  }

  Future<bool> readContextAware() async {
    final box = _settingsBox();
    return box?.get(_contextAwareKey, defaultValue: false) as bool? ?? false;
  }

  Future<void> saveBackbone(String backbone) {
    return Future.sync(() => _settingsBox()?.put(_backboneKey, backbone));
  }

  Future<String> readBackbone() async {
    final box = _settingsBox();
    return box?.get(_backboneKey, defaultValue: "yamnet") as String? ?? "yamnet";
  }

  Future<List<CustomSoundSample>> readRecordedSounds() async {
    final Box<dynamic>? box = _recordingsBox();
    final dynamic rawSounds =
        box?.get(_recordedSoundsKey, defaultValue: const <dynamic>[]);
    final List<dynamic> sounds =
        rawSounds is List<dynamic> ? rawSounds : const <dynamic>[];
    return sounds
        .whereType<Map<dynamic, dynamic>>()
        .map(CustomSoundSample.fromMap)
        .toList(growable: false);
  }

  Future<void> saveRecordedSounds(List<CustomSoundSample> sounds) {
    final List<Map<String, dynamic>> payload = sounds
        .map((CustomSoundSample sound) => sound.toMap())
        .toList(growable: false);
    return Future.sync(
        () => _recordingsBox()?.put(_recordedSoundsKey, payload));
  }
}
