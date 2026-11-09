class AppConfig {
  /// The active ML backbone model.
  /// Supported values:
  /// - "yamnet" (YAMNet backbone pipeline)
  /// - "tiny_cnn" (Tiny CNN / MobileNetV4 pipeline)
  static const String activeBackbone = "yamnet";
}
