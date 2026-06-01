import "dart:io";
import "dart:typed_data";
import "package:hive/hive.dart";
import "package:encrypt/encrypt.dart" as enc;
import "../models/custom_sound_sample.dart";

/// Persists preferences, custom sound configs, thresholds, context policies,
/// event history, feedback, and user recorded sound metadata in Hive with AES-256 encryption.
class LocalStorageService {
  LocalStorageService();

  // Hive Box Names
  static const String _settingsBoxName = "soundpulse.settings";
  static const String _classConfigBoxName = "soundpulse.class_config";
  static const String _prototypeBoxName = "soundpulse.prototype";
  static const String _thresholdBoxName = "soundpulse.threshold";
  static const String _modePolicyBoxName = "soundpulse.mode_policy";
  static const String _eventHistoryBoxName = "soundpulse.event_history";
  static const String _feedbackBoxName = "soundpulse.feedback";
  static const String _recordingMetadataBoxName = "soundpulse.recording_metadata";

  // Setting Keys
  static const String _modeKey = "soundpulse.mode";
  static const String _contextAwareKey = "soundpulse.context_aware";
  static const String _backboneKey = "soundpulse.backbone";
  static const String _vibrationKey = "soundpulse.vibration_enabled";
  static const String _cameraFlashKey = "soundpulse.camera_flash_enabled";
  static const String _personalisationLevelKey = "soundpulse.personalisation_level";

  // ----------------------------------------------------
  // Box Getters
  // ----------------------------------------------------
  Box<dynamic>? _settingsBox() =>
      Hive.isBoxOpen(_settingsBoxName) ? Hive.box<dynamic>(_settingsBoxName) : null;

  Box<dynamic>? _classConfigBox() =>
      Hive.isBoxOpen(_classConfigBoxName) ? Hive.box<dynamic>(_classConfigBoxName) : null;

  Box<dynamic>? _prototypeBox() =>
      Hive.isBoxOpen(_prototypeBoxName) ? Hive.box<dynamic>(_prototypeBoxName) : null;

  Box<dynamic>? _thresholdBox() =>
      Hive.isBoxOpen(_thresholdBoxName) ? Hive.box<dynamic>(_thresholdBoxName) : null;

  Box<dynamic>? _modePolicyBox() =>
      Hive.isBoxOpen(_modePolicyBoxName) ? Hive.box<dynamic>(_modePolicyBoxName) : null;

  Box<dynamic>? _eventHistoryBox() =>
      Hive.isBoxOpen(_eventHistoryBoxName) ? Hive.box<dynamic>(_eventHistoryBoxName) : null;

  Box<dynamic>? _feedbackBox() =>
      Hive.isBoxOpen(_feedbackBoxName) ? Hive.box<dynamic>(_feedbackBoxName) : null;

  Box<dynamic>? _recordingMetadataBox() =>
      Hive.isBoxOpen(_recordingMetadataBoxName)
          ? Hive.box<dynamic>(_recordingMetadataBoxName)
          : null;

  // ----------------------------------------------------
  // AES-256 Symmetric File Encryption / Decryption
  // ----------------------------------------------------
  enc.Key _getOrGenerateEncryptionKey() {
    final box = _settingsBox();
    final storedKeyBase64 = box?.get("soundpulse.encryption_key");
    if (storedKeyBase64 is String) {
      return enc.Key.fromBase64(storedKeyBase64);
    }
    // Generate a unique secure 256-bit (32-byte) key per device
    final newKey = enc.Key.fromSecureRandom(32);
    box?.put("soundpulse.encryption_key", newKey.base64);
    return newKey;
  }

  /// Encrypts a raw on-device WAV file using standard AES-256 in CBC mode.
  /// The initialization vector (IV) is generated dynamically and prepended to the cipher payload.
  Future<void> encryptFile(String inputPath, String outputPath) async {
    final key = _getOrGenerateEncryptionKey();
    final iv = enc.IV.fromSecureRandom(16);
    final encrypter = enc.Encrypter(enc.AES(key, mode: enc.AESMode.cbc));

    final fileBytes = await File(inputPath).readAsBytes();
    final encrypted = encrypter.encryptBytes(fileBytes, iv: iv);

    // Concat: IV (16 bytes) + Encrypted Bytes
    final outputBytes = Uint8List(iv.bytes.length + encrypted.bytes.length);
    outputBytes.setAll(0, iv.bytes);
    outputBytes.setAll(iv.bytes.length, encrypted.bytes);

    await File(outputPath).writeAsBytes(outputBytes, flush: true);
  }

  /// Decrypts an AES-256 encrypted local audio sample back to raw WAV byte data in-memory.
  Future<Uint8List> decryptFile(String inputPath) async {
    final key = _getOrGenerateEncryptionKey();
    final encrypter = enc.Encrypter(enc.AES(key, mode: enc.AESMode.cbc));

    final encryptedBytes = await File(inputPath).readAsBytes();
    if (encryptedBytes.length < 16) {
      throw const FormatException("Encrypted WAV file is corrupt or invalid.");
    }

    final ivBytes = encryptedBytes.sublist(0, 16);
    final cipherBytes = encryptedBytes.sublist(16);

    final iv = enc.IV(ivBytes);
    final decrypted = encrypter.decryptBytes(enc.Encrypted(cipherBytes), iv: iv);
    return Uint8List.fromList(decrypted);
  }

  /// Helper to temporarily decrypt a WAV file to a secure directory for playback.
  Future<String> decryptToTempFile(String encryptedPath) async {
    final decryptedBytes = await decryptFile(encryptedPath);
    final tempDir = Directory.systemTemp;
    final tempPath =
        "${tempDir.path}/soundpulse_play_${DateTime.now().millisecondsSinceEpoch}.wav";
    await File(tempPath).writeAsBytes(decryptedBytes, flush: true);
    return tempPath;
  }

  // ----------------------------------------------------
  // Box 1: Settings (soundpulse.settings)
  // ----------------------------------------------------
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

  Future<void> saveVibrationEnabled(bool enabled) {
    return Future.sync(() => _settingsBox()?.put(_vibrationKey, enabled));
  }

  Future<bool> readVibrationEnabled() async {
    final box = _settingsBox();
    return box?.get(_vibrationKey, defaultValue: false) as bool? ?? false;
  }

  Future<void> saveCameraFlashEnabled(bool enabled) {
    return Future.sync(() => _settingsBox()?.put(_cameraFlashKey, enabled));
  }

  Future<bool> readCameraFlashEnabled() async {
    final box = _settingsBox();
    return box?.get(_cameraFlashKey, defaultValue: false) as bool? ?? false;
  }

  Future<void> savePersonalisationLevel(String level) {
    return Future.sync(() => _settingsBox()?.put(_personalisationLevelKey, level));
  }

  Future<String> readPersonalisationLevel() async {
    final box = _settingsBox();
    return box?.get(_personalisationLevelKey, defaultValue: "default") as String? ?? "default";
  }

  // ----------------------------------------------------
  // Box 2: Class Config (soundpulse.class_config)
  // ----------------------------------------------------
  Future<void> saveClassConfig(String classSlug, Map<String, dynamic> config) {
    return Future.sync(() => _classConfigBox()?.put(classSlug, config));
  }

  Future<Map<String, dynamic>?> readClassConfig(String classSlug) async {
    final raw = _classConfigBox()?.get(classSlug);
    return raw is Map ? raw.cast<String, dynamic>() : null;
  }

  Future<List<Map<String, dynamic>>> readAllClassConfigs() async {
    final box = _classConfigBox();
    if (box == null) return const [];
    return box.values
        .whereType<Map>()
        .map((m) => m.cast<String, dynamic>())
        .toList(growable: false);
  }

  // ----------------------------------------------------
  // Box 3: Prototype (soundpulse.prototype)
  // ----------------------------------------------------
  Future<void> savePrototype(String classSlug, Map<String, dynamic> prototype) {
    return Future.sync(() => _prototypeBox()?.put(classSlug, prototype));
  }

  Future<Map<String, dynamic>?> readPrototype(String classSlug) async {
    final raw = _prototypeBox()?.get(classSlug);
    return raw is Map ? raw.cast<String, dynamic>() : null;
  }

  Future<void> savePrototypeCentroids(Map<String, List<double>> centroids, {String sourceType = "advanced_personalisation"}) async {
    final box = _prototypeBox();
    if (box == null) return;
    await box.clear();
    for (final entry in centroids.entries) {
      await savePrototype(entry.key, {
        "class_id": entry.key,
        "centroid_vector": entry.value,
        "updated_at": DateTime.now().toIso8601String(),
        "source_type": sourceType
      });
    }
  }

  Future<Map<String, List<double>>> readPrototypeCentroids() async {
    final box = _prototypeBox();
    if (box == null) return const {};

    final Map<String, List<double>> centroids = {};
    for (final key in box.keys) {
      final rawVal = box.get(key);
      if (rawVal is Map) {
        final vec = rawVal["centroid_vector"];
        if (vec is List) {
          centroids[key.toString()] = vec.whereType<num>().map((e) => e.toDouble()).toList();
        }
      }
    }
    return centroids;
  }

  // ----------------------------------------------------
  // Box 4: Thresholds (soundpulse.threshold)
  // ----------------------------------------------------
  Future<void> saveThresholdConfig(String classSlug, Map<String, dynamic> threshold) {
    return Future.sync(() => _thresholdBox()?.put(classSlug, threshold));
  }

  Future<Map<String, dynamic>?> readThresholdConfig(String classSlug) async {
    final raw = _thresholdBox()?.get(classSlug);
    return raw is Map ? raw.cast<String, dynamic>() : null;
  }

  // ----------------------------------------------------
  // Box 5: Mode Policy (soundpulse.mode_policy)
  // ----------------------------------------------------
  Future<void> saveModePolicy(String modeName, Map<String, dynamic> policy) {
    return Future.sync(() => _modePolicyBox()?.put(modeName, policy));
  }

  Future<Map<String, dynamic>?> readModePolicy(String modeName) async {
    final raw = _modePolicyBox()?.get(modeName);
    return raw is Map ? raw.cast<String, dynamic>() : null;
  }

  // ----------------------------------------------------
  // Box 6: Event History (soundpulse.event_history)
  // ----------------------------------------------------
  Future<void> saveEventHistory(List<Map<String, dynamic>> events) async {
    final box = _eventHistoryBox();
    if (box == null) return;
    for (final event in events) {
      final id = event["event_id"] as String?;
      if (id != null) {
        await box.put(id, event);
      }
    }
  }

  Future<List<Map<String, dynamic>>> readEventHistory() async {
    final box = _eventHistoryBox();
    if (box == null) return const [];
    return box.values
        .whereType<Map>()
        .map((e) => e.cast<String, dynamic>())
        .toList(growable: false);
  }

  Future<void> clearEventHistory() async {
    await _eventHistoryBox()?.clear();
  }

  // ----------------------------------------------------
  // Box 7: Feedback (soundpulse.feedback)
  // ----------------------------------------------------
  Future<void> saveFeedback(String eventId, Map<String, dynamic> feedback) {
    return Future.sync(() => _feedbackBox()?.put(eventId, feedback));
  }

  Future<Map<String, dynamic>?> readFeedback(String eventId) async {
    final raw = _feedbackBox()?.get(eventId);
    return raw is Map ? raw.cast<String, dynamic>() : null;
  }

  Future<Map<String, Map<String, dynamic>>> readAllFeedback() async {
    final box = _feedbackBox();
    if (box == null) return const {};
    final Map<String, Map<String, dynamic>> results = {};
    for (final key in box.keys) {
      final val = box.get(key);
      if (val is Map) {
        results[key.toString()] = val.cast<String, dynamic>();
      }
    }
    return results;
  }

  // ----------------------------------------------------
  // Box 8: Recording Metadata (soundpulse.recording_metadata)
  // ----------------------------------------------------
  Future<void> saveRecordingMetadata(String recordingId, Map<String, dynamic> metadata) {
    return Future.sync(() => _recordingMetadataBox()?.put(recordingId, metadata));
  }

  Future<Map<String, dynamic>?> readRecordingMetadata(String recordingId) async {
    final raw = _recordingMetadataBox()?.get(recordingId);
    return raw is Map ? raw.cast<String, dynamic>() : null;
  }

  Future<List<CustomSoundSample>> readRecordedSounds() async {
    final box = _recordingMetadataBox();
    if (box == null) return const [];
    return box.values
        .whereType<Map>()
        .map((m) => CustomSoundSample.fromMap(m))
        .toList(growable: false);
  }

  Future<void> saveRecordedSounds(List<CustomSoundSample> sounds) async {
    final box = _recordingMetadataBox();
    if (box == null) return;
    await box.clear();
    for (final sound in sounds) {
      await box.put(sound.id, sound.toMap());
    }
  }
}
