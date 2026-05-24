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

  Future<String> _copyAssetToLocalFile(String assetPath, String localFileName) async {
    final directory = await getApplicationDocumentsDirectory();
    final localFile = File("${directory.path}/$localFileName");
    final data = await rootBundle.load(assetPath);
    final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await localFile.writeAsBytes(bytes, flush: true);
    return localFile.path;
  }

  Future<void> startMonitoring({required String mode, required String backbone}) async {
    // 1. Copy common configs to local files
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

    // 3. Invoke native monitoring with all the paths
    return _methods.invokeMethod<void>("startMonitoring", <String, dynamic>{
      "mode": mode,
      "backbone": backbone,
      "modelPath": modelPath,
      "classLabelsPath": classLabelsPath,
      "preprocessingConfigPath": preprocessingConfigPath,
      "thresholdDefaultsPath": thresholdDefaultsPath,
      "prototypeStorePath": prototypeStorePath,
    });
  }

  Future<void> stopMonitoring() {
    return _methods.invokeMethod<void>("stopMonitoring");
  }

  Future<void> updateMode(String mode) {
    return _methods.invokeMethod<void>("updateMode", <String, dynamic>{"mode": mode});
  }

  Future<void> setContextAware(bool enabled) {
    return _methods.invokeMethod<void>("setContextAware", <String, dynamic>{"enabled": enabled});
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
}
