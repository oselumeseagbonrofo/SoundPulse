import "inference_event.dart";

class MonitorHistoryEntry {
  const MonitorHistoryEntry({
    required this.event,
    required this.capturedAt,
  });

  final InferenceEvent event;
  final DateTime capturedAt;
}
