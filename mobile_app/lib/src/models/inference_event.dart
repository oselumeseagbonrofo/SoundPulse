class InferenceEvent {
  InferenceEvent({
    required this.label,
    required this.confidenceTier,
    required this.confidence,
    required this.isUnknown,
    required this.isBackground,
    this.waveformSamples = const <double>[],
  });

  final String label;
  final String confidenceTier;
  final double confidence;
  final bool isUnknown;
  final bool isBackground;
  final List<double> waveformSamples;

  factory InferenceEvent.fromMap(Map<dynamic, dynamic> map) {
    final flags =
        (map["flags"] as Map<dynamic, dynamic>? ?? <dynamic, dynamic>{});
    final dynamic waveformRaw = map["waveform_samples"];
    final List<double> waveformSamples =
        (waveformRaw is List<dynamic> ? waveformRaw : const <dynamic>[])
            .map((dynamic sample) => (sample as num).toDouble())
            .toList(growable: false);
    return InferenceEvent(
      label: (map["label"] as String?) ?? "unknown",
      confidenceTier: (map["confidence_tier"] as String?) ?? "low",
      confidence: ((map["confidence"] as num?) ?? 0.0).toDouble(),
      isUnknown: (flags["is_unknown"] as bool?) ?? true,
      isBackground: (flags["is_background"] as bool?) ?? false,
      waveformSamples: waveformSamples,
    );
  }
}
