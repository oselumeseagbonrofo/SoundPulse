class CustomSoundSample {
  CustomSoundSample({
    required this.id,
    required this.name,
    required this.recordedAt,
    required this.durationSeconds,
    required this.filePath,
  });

  final String id;
  final String name;
  final DateTime recordedAt;
  final int durationSeconds;
  final String filePath;

  CustomSoundSample copyWith({
    String? name,
    String? filePath,
  }) {
    return CustomSoundSample(
      id: id,
      name: name ?? this.name,
      recordedAt: recordedAt,
      durationSeconds: durationSeconds,
      filePath: filePath ?? this.filePath,
    );
  }

  factory CustomSoundSample.fromMap(Map<dynamic, dynamic> map) {
    final String? timestamp = map["recorded_at"] as String?;
    return CustomSoundSample(
      id: (map["id"] as String?) ?? "",
      name: (map["name"] as String?) ?? "Untitled sound",
      recordedAt: DateTime.tryParse(timestamp ?? "") ?? DateTime.now(),
      durationSeconds: ((map["duration_seconds"] as num?) ?? 5).toInt(),
      filePath: (map["file_path"] as String?) ?? "",
    );
  }

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      "id": id,
      "name": name,
      "recorded_at": recordedAt.toIso8601String(),
      "duration_seconds": durationSeconds,
      "file_path": filePath,
    };
  }
}
