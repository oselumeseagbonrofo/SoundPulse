class CustomSoundSample {
  CustomSoundSample({
    required this.id,
    required this.name,
    required this.classSlug,
    required this.classLabel,
    required this.recordedAt,
    required this.durationSeconds,
    required this.filePath,
    this.embedding = const <double>[],
    this.audit = const <String, dynamic>{},
  });

  final String id;
  final String name;
  final String classSlug;
  final String classLabel;
  final DateTime recordedAt;
  final int durationSeconds;
  final String filePath;
  final List<double> embedding;
  final Map<String, dynamic> audit;

  CustomSoundSample copyWith({
    String? name,
    String? filePath,
    List<double>? embedding,
  }) {
    return CustomSoundSample(
      id: id,
      name: name ?? this.name,
      classSlug: classSlug,
      classLabel: classLabel,
      recordedAt: recordedAt,
      durationSeconds: durationSeconds,
      filePath: filePath ?? this.filePath,
      embedding: embedding ?? this.embedding,
      audit: audit,
    );
  }

  factory CustomSoundSample.fromMap(Map<dynamic, dynamic> map) {
    final String? timestamp = map["recorded_at"] as String?;
    final dynamic rawEmbedding = map["embedding"];
    final dynamic rawAudit = map["audit"];
    return CustomSoundSample(
      id: (map["id"] as String?) ?? "",
      name: (map["name"] as String?) ?? "Untitled sound",
      classSlug: (map["class_slug"] as String?) ?? "unknown",
      classLabel: (map["class_label"] as String?) ??
          (map["name"] as String?) ??
          "Untitled sound",
      recordedAt: DateTime.tryParse(timestamp ?? "") ?? DateTime.now(),
      durationSeconds: ((map["duration_seconds"] as num?) ?? 5).toInt(),
      filePath: (map["file_path"] as String?) ?? "",
      embedding:
          (rawEmbedding is List<dynamic> ? rawEmbedding : const <dynamic>[])
              .whereType<num>()
              .map((num value) => value.toDouble())
              .toList(growable: false),
      audit: rawAudit is Map<dynamic, dynamic>
          ? rawAudit.map((dynamic key, dynamic value) =>
              MapEntry<String, dynamic>(key.toString(), value))
          : const <String, dynamic>{},
    );
  }

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      "id": id,
      "name": name,
      "class_slug": classSlug,
      "class_label": classLabel,
      "recorded_at": recordedAt.toIso8601String(),
      "duration_seconds": durationSeconds,
      "file_path": filePath,
      "embedding": embedding,
      "audit": audit,
    };
  }
}
