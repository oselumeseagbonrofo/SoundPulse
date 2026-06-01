import "../models/inference_event.dart";

/// The structural alert modality routing decisions.
enum AlertModality {
  none,
  silentLog,
  textOnly,
  standardVibrationText,
  vibrationFlashText,
  criticalFullScreen,
}

/// Helper extension to serialize/deserialize AlertModality to String.
extension AlertModalityExtension on AlertModality {
  String toStr() {
    switch (this) {
      case AlertModality.none:
        return "none";
      case AlertModality.silentLog:
        return "silent_log";
      case AlertModality.textOnly:
        return "text_only";
      case AlertModality.standardVibrationText:
        return "standard_vibration_text";
      case AlertModality.vibrationFlashText:
        return "vibration_flash_text";
      case AlertModality.criticalFullScreen:
        return "critical_full_screen";
    }
  }

  static AlertModality fromStr(String val) {
    switch (val) {
      case "none":
        return AlertModality.none;
      case "silent_log":
        return AlertModality.silentLog;
      case "text_only":
        return AlertModality.textOnly;
      case "standard_vibration_text":
        return AlertModality.standardVibrationText;
      case "vibration_flash_text":
        return AlertModality.vibrationFlashText;
      case "critical_full_screen":
        return AlertModality.criticalFullScreen;
      default:
        return AlertModality.silentLog;
    }
  }
}

/// The decided priority of an event after context and policy evaluation.
enum EventPriority {
  critical,
  important,
  useful,
  background,
  low,
}

/// Represents the final decision of the Alert Policy Decision Engine.
class AlertDecision {
  AlertDecision({
    required this.classSlug,
    required this.priority,
    required this.modality,
    required this.confidence,
    required this.suppressed,
    required this.suppressionReason,
    this.chainBoostApplied = false,
  });

  final String classSlug;
  final EventPriority priority;
  final AlertModality modality;
  final double confidence;
  final bool suppressed;
  final String suppressionReason;
  final bool chainBoostApplied;

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      "class_slug": classSlug,
      "priority": priority.toString().split(".").last,
      "modality": modality.toStr(),
      "confidence": confidence,
      "suppressed": suppressed,
      "suppression_reason": suppressionReason,
      "chain_boost_applied": chainBoostApplied,
    };
  }
}

/// Represents an entry in the rolling temporal event buffer.
class TemporalEventEntry {
  TemporalEventEntry({
    required this.slug,
    required this.timestamp,
    required this.confidence,
  });

  final String slug;
  final DateTime timestamp;
  final double confidence;
}

/// ContextManager coordinates dynamic alert routing, haptic/visual modality decisions,
/// temporal sequence boosts, and deduplication controls based on active context modes.
class ContextManager {
  ContextManager();

  // In-memory rolling 15-second history buffer of recent events.
  final List<TemporalEventEntry> _recentEventsBuffer = <TemporalEventEntry>[];

  // In-memory tracker of when a notification was last successfully dispatched per sound class.
  final Map<String, DateTime> _lastAlertTimestamps = <String, DateTime>{};

  // 15-second window parameter from the policy
  static const Duration _historyWindow = Duration(seconds: 15);

  /// Default baseline sound class mappings to Priorities and Modalities.
  static const Map<String, EventPriority> _defaultPriorities = {
    "fire_alarm": EventPriority.critical,
    "clock_alarm": EventPriority.critical,
    "door_knock": EventPriority.important,
    "doorbell": EventPriority.important,
    "door_wood_creaking": EventPriority.important,
    "crying_baby": EventPriority.important,
    "dog_barking": EventPriority.important,
    "water_running": EventPriority.useful,
    "speech": EventPriority.important,
    "background": EventPriority.background,
    "unknown": EventPriority.low,
  };

  static const Map<String, AlertModality> _defaultModalities = {
    "fire_alarm": AlertModality.criticalFullScreen,
    "clock_alarm": AlertModality.criticalFullScreen,
    "door_knock": AlertModality.vibrationFlashText,
    "doorbell": AlertModality.vibrationFlashText,
    "door_wood_creaking": AlertModality.vibrationFlashText,
    "crying_baby": AlertModality.vibrationFlashText,
    "dog_barking": AlertModality.standardVibrationText,
    "water_running": AlertModality.standardVibrationText,
    "speech": AlertModality.standardVibrationText,
    "background": AlertModality.none,
    "unknown": AlertModality.silentLog,
  };

  /// Decides which Alert Modality to employ based on the event and active context mode.
  AlertDecision decideAlert({
    required InferenceEvent event,
    required String activeMode,
    Map<String, dynamic>? userOverrides, // Custom overrides from class_config Hive box if any
  }) {
    final String rawSlug = event.label;
    final double rawConfidence = event.confidence;

    // Suppress if explicitly disabled by user
    if (userOverrides != null && userOverrides["enabled"] == false) {
      return AlertDecision(
        classSlug: rawSlug,
        priority: _defaultPriorities[rawSlug] ?? EventPriority.low,
        modality: AlertModality.none,
        confidence: rawConfidence,
        suppressed: true,
        suppressionReason: "Suppressed because sound detection is disabled in settings.",
      );
    }

    // 1. Garbage collect the rolling event queue to strictly keep only last 15 seconds.
    final DateTime now = DateTime.now();
    _recentEventsBuffer.removeWhere((entry) => now.difference(entry.timestamp) > _historyWindow);

    // 2. Suppress Ambient Background hum immediately
    if (rawSlug == "background" || event.isBackground) {
      return AlertDecision(
        classSlug: "background",
        priority: EventPriority.background,
        modality: AlertModality.none,
        confidence: rawConfidence,
        suppressed: true,
        suppressionReason: "Suppressed ambient background hum.",
      );
    }

    // 3. Resolve base configuration (priority and modality)
    EventPriority priority = _defaultPriorities[rawSlug] ?? EventPriority.low;
    AlertModality modality = _defaultModalities[rawSlug] ?? AlertModality.silentLog;

    // Load any user overrides if supplied
    if (userOverrides != null) {
      if (userOverrides["priority_level"] != null) {
        priority = EventPriority.values.firstWhere(
          (e) => e.toString().split(".").last == userOverrides["priority_level"],
          orElse: () => priority,
        );
      }
      if (userOverrides["modes"] is Map) {
        final Map<dynamic, dynamic> modesMap = userOverrides["modes"] as Map;
        if (modesMap[activeMode] != null) {
          modality = AlertModalityExtension.fromStr(modesMap[activeMode] as String);
        }
      } else if (userOverrides["user_alert_modality"] != null) {
        modality = AlertModalityExtension.fromStr(userOverrides["user_alert_modality"] as String);
      }
    }

    // 4. Temporal Event-Chain Sequence Boosts: knock_then_speech
    // Sequence check: If current is speech, and we have door_knock or doorbell in last 15s.
    double finalConfidence = rawConfidence;
    bool chainBoostApplied = false;

    if (rawSlug == "speech") {
      final bool hasRecentKnock = _recentEventsBuffer.any(
        (entry) => entry.slug == "door_knock" || entry.slug == "doorbell"
      );
      if (hasRecentKnock) {
        finalConfidence = (rawConfidence + 0.05).clamp(0.0, 1.0);
        priority = EventPriority.important;
        modality = AlertModality.vibrationFlashText;
        chainBoostApplied = true;
      }
    }

    // Add current event to in-memory history buffer
    _recentEventsBuffer.add(TemporalEventEntry(
      slug: rawSlug,
      timestamp: now,
      confidence: finalConfidence,
    ));

    // 5. Active Context Mode Policy Rules
    bool suppressed = false;
    String suppressionReason = "";
    AlertModality decidedModality = modality;

    final bool hasModeOverride = userOverrides != null &&
        userOverrides["modes"] is Map &&
        (userOverrides["modes"] as Map)[activeMode] != null;

    if (!hasModeOverride) {
      // Sleep Mode Rules
      if (activeMode == "sleep") {
        if (priority == EventPriority.useful || priority == EventPriority.low) {
          suppressed = true;
          decidedModality = AlertModality.silentLog;
          suppressionReason = "Suppressed low-value $priority sound in Sleep Mode.";
        } else if (priority == EventPriority.important) {
          decidedModality = AlertModality.vibrationFlashText; // Sleep mode raises important to vibration+flash
        } else if (priority == EventPriority.critical) {
          decidedModality = AlertModality.criticalFullScreen; // Critical sounds remain full screen
        }
      }
      // Focus Mode Rules
      else if (activeMode == "focus") {
        if (priority != EventPriority.critical) {
          suppressed = true;
          decidedModality = AlertModality.silentLog;
          suppressionReason = "Suppressed non-critical sound in Focus Mode.";
        } else {
          decidedModality = AlertModality.criticalFullScreen;
        }
      }
      // Cooking Mode Rules
      else if (activeMode == "cooking") {
        final bool isCookingRelevant = rawSlug == "fire_alarm" ||
            rawSlug == "clock_alarm" ||
            rawSlug == "door_knock" ||
            rawSlug == "doorbell" ||
            rawSlug == "water_running" ||
            rawSlug == "speech";
        if (!isCookingRelevant) {
          suppressed = true;
          decidedModality = AlertModality.silentLog;
          suppressionReason = "Suppressed non-cooking sound in Cooking Mode.";
        }
      }
      // Bathroom Mode Rules
      else if (activeMode == "bathroom") {
        if (rawSlug == "water_running") {
          suppressed = true;
          decidedModality = AlertModality.silentLog;
          suppressionReason = "Suppressed running water in Bathroom Mode.";
        } else if (priority == EventPriority.important) {
          decidedModality = AlertModality.vibrationFlashText;
        }
      }
    }

    // 6. Notification Overload & Rate-Limiting Deduplication
    if (!suppressed && decidedModality != AlertModality.none && decidedModality != AlertModality.silentLog) {
      final DateTime? lastAlert = _lastAlertTimestamps[rawSlug];
      if (lastAlert != null) {
        final Duration elapsed = now.difference(lastAlert);
        Duration minWindow;
        switch (priority) {
          case EventPriority.critical:
            minWindow = const Duration(seconds: 10);
            break;
          case EventPriority.important:
            minWindow = const Duration(seconds: 45);
            break;
          case EventPriority.useful:
            minWindow = const Duration(seconds: 120);
            break;
          case EventPriority.background:
          case EventPriority.low:
            minWindow = const Duration(seconds: 300);
            break;
        }

        if (elapsed < minWindow) {
          suppressed = true;
          decidedModality = AlertModality.silentLog;
          suppressionReason = "Rate-limited ($priority): Triggered within ${minWindow.inSeconds}s window (elapsed: ${elapsed.inSeconds}s).";
        }
      }
    }

    // Update last alert timestamp if successfully alert-routed
    if (!suppressed && decidedModality != AlertModality.none && decidedModality != AlertModality.silentLog) {
      _lastAlertTimestamps[rawSlug] = now;
    }

    return AlertDecision(
      classSlug: rawSlug,
      priority: priority,
      modality: decidedModality,
      confidence: finalConfidence,
      suppressed: suppressed,
      suppressionReason: suppressionReason,
      chainBoostApplied: chainBoostApplied,
    );
  }

  /// Clean up memory on dispose or reset.
  void reset() {
    _recentEventsBuffer.clear();
    _lastAlertTimestamps.clear();
  }
}
