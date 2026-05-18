import "inference_event.dart";

class MonitorHistoryEntry {
  const MonitorHistoryEntry({
    required this.id,
    required this.event,
    required this.capturedAt,
    required this.waveformSamples,
    required this.audioFilePath,
  });

  final String id;
  final InferenceEvent event;
  final DateTime capturedAt;
  final List<double> waveformSamples;
  final String audioFilePath;
}
