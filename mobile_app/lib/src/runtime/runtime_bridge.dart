import "dart:async";

import "package:flutter/services.dart";

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

  Future<void> startMonitoring({required String mode}) {
    return _methods.invokeMethod<void>("startMonitoring", <String, dynamic>{"mode": mode});
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
