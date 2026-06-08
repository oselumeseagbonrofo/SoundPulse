import "dart:async";
import "dart:io";

import "package:flutter/services.dart";
import "package:path_provider/path_provider.dart";

import "../models/inference_event.dart";

class RuntimeBridge {
  RuntimeBridge()
      : _events = const EventChannel("soundpulse.runtime/events"),
        _methods = const MethodChannel("soundpulse.runtime/methods");

  final EventChannel _events;
  final MethodChannel _methods;

  Stream<InferenceEvent> streamEvents() {
    return _events.receiveBroadcastStream().map((dynamic raw) {
      final map = raw as Map<dynamic, dynamic>;
      return InferenceEvent.fromMap(map);
    });
  }

  Future<String> _copyAssetToLocalFile(
      String assetPath, String localFileName) async {
    final directory = await getApplicationDocumentsDirectory();
    final localFile = File("${directory.path}/$localFileName");
    final data = await rootBundle.load(assetPath);
    final bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await localFile.writeAsBytes(bytes, flush: true);
    return localFile.path;
  }

  Future<Map<String, dynamic>> _runtimeAssetArgs(String backbone) async {
    final String classLabelsPath = await _copyAssetToLocalFile(
      "assets/config/class_labels.json",
      "class_labels.json",
    );
    final String preprocessingConfigPath = await _copyAssetToLocalFile(
      "assets/config/preprocessing_config.json",
      "preprocessing_config.json",
    );
    final String thresholdDefaultsPath = await _copyAssetToLocalFile(
      "assets/config/threshold_defaults.json",
      "threshold_defaults.json",
    );

    // 2. Copy backbone-specific files to local files
    final String modelAsset = backbone == "tiny_cnn"
        ? "assets/models/tiny_cnn/tinycnn_embedding.tflite"
        : "assets/models/yamnet/yamnet_embedding.tflite";
    final String protoAsset = backbone == "tiny_cnn"
        ? "assets/models/tiny_cnn/prototype_store.json"
        : "assets/models/yamnet/prototype_store.json";

    final String modelPath = await _copyAssetToLocalFile(
      modelAsset,
      "${backbone}_embedding.tflite",
    );
    final String prototypeStorePath = await _copyAssetToLocalFile(
      protoAsset,
      "prototype_store_$backbone.json",
    );

    return <String, dynamic>{
      "backbone": backbone,
      "modelPath": modelPath,
      "classLabelsPath": classLabelsPath,
      "preprocessingConfigPath": preprocessingConfigPath,
      "thresholdDefaultsPath": thresholdDefaultsPath,
      "prototypeStorePath": prototypeStorePath,
    };
  }

  Future<void> initializeRuntime({required String backbone}) async {
    return _methods.invokeMethod<void>(
      "initializeRuntime",
      await _runtimeAssetArgs(backbone),
    );
  }

  Future<void> startMonitoring(
      {required String mode, required String backbone}) async {
    final Map<String, dynamic> args = await _runtimeAssetArgs(backbone);
    args["mode"] = mode;
    return _methods.invokeMethod<void>("startMonitoring", args);
  }

  Future<void> stopMonitoring() {
    return _methods.invokeMethod<void>("stopMonitoring");
  }

  Future<bool> isMonitoringActive() async {
    return await _methods.invokeMethod<bool>("isMonitoringActive") ?? false;
  }

  Future<void> updateMode(String mode) {
    return _methods
        .invokeMethod<void>("updateMode", <String, dynamic>{"mode": mode});
  }

  Future<void> setContextAware(bool enabled) {
    return _methods.invokeMethod<void>(
        "setContextAware", <String, dynamic>{"enabled": enabled});
  }

  Future<List<InferenceEvent>> pushAudioPcm(List<double> samples) async {
    final response = await _methods.invokeMethod<List<dynamic>>(
      "pushAudioPcm",
      <String, dynamic>{"samples": samples},
    );
    if (response == null) {
      return const <InferenceEvent>[];
    }
    return response.map((dynamic raw) {
      final map = raw as Map<dynamic, dynamic>;
      return InferenceEvent.fromMap(map);
    }).toList(growable: false);
  }

  Future<List<double>> extractEmbeddingFromPcm(List<double> samples) async {
    final response = await _methods.invokeMethod<List<dynamic>>(
      "extractEmbeddingFromPcm",
      <String, dynamic>{"samples": samples},
    );
    return (response ?? const <dynamic>[])
        .map((dynamic value) => (value as num).toDouble())
        .toList(growable: false);
  }

  Future<void> updatePrototype({
    required String label,
    required List<double> embedding,
  }) {
    return _methods.invokeMethod<void>("updatePrototype", <String, dynamic>{
      "label": label,
      "embedding": embedding,
    });
  }

  Future<void> resetPrototype(String label) {
    return _methods.invokeMethod<void>(
      "resetPrototype",
      <String, dynamic>{"label": label},
    );
  }

  Future<void> showNotification({required String title, required String body}) {
    return _methods.invokeMethod<void>("showNotification", <String, dynamic>{
      "title": title,
      "body": body,
    });
  }

  Future<void> vibrate() {
    return _methods.invokeMethod<void>("vibrate");
  }

  Future<void> triggerCameraFlash() {
    return _methods.invokeMethod<void>("triggerCameraFlash");
  }

  Future<void> requestNotificationPermission() {
    return _methods.invokeMethod<void>("requestNotificationPermission");
  }

  Future<bool> isBatteryOptimizationIgnored() async {
    return await _methods.invokeMethod<bool>("isBatteryOptimizationIgnored") ?? true;
  }

  Future<void> requestIgnoreBatteryOptimizations() {
    return _methods.invokeMethod<void>("requestIgnoreBatteryOptimizations");
  }
}
