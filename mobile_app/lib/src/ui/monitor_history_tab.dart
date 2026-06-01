import "package:flutter/material.dart";
import "../models/monitor_history_entry.dart";
import "monitor_style.dart";

class MonitorHistoryTab extends StatelessWidget {
  const MonitorHistoryTab({
    super.key,
    required this.history,
    required this.submittedFeedback,
    required this.onSubmitFeedback,
  });

  final List<MonitorHistoryEntry> history;
  final Map<String, String> submittedFeedback;
  final void Function(String eventId, String label, String feedbackValue) onSubmitFeedback;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 100),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          const Text(
            "All Detections",
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: monitorOnSurface),
          ),
          const SizedBox(height: 8),
          if (history.isEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: monitorRaised(radius: 18),
              child: const Text("No detections yet", style: TextStyle(color: monitorOnSurfaceVariant)),
            )
          else
            ...history.map(_buildHistoryTile),
        ],
      ),
    );
  }

  Widget _buildHistoryTile(MonitorHistoryEntry entry) {
    final String label = entry.event.label;
    final String? currentFeedback = submittedFeedback[entry.id];
    return Container(
      margin: const EdgeInsets.only(top: 10),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: monitorRaised(radius: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              Expanded(
                child: Text(
                  formatSoundLabel(label),
                  style: const TextStyle(fontWeight: FontWeight.w600, color: monitorOnSurface),
                ),
              ),
              Text(
                "${(entry.event.confidence * 100).round()}%",
                style: const TextStyle(fontWeight: FontWeight.w600, color: monitorOnSurfaceVariant),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: <Widget>[
              const Text(
                "Feedback: ",
                style: TextStyle(fontSize: 11, color: monitorOnSurfaceVariant, fontWeight: FontWeight.w500),
              ),
              const Spacer(),
              _feedbackButton(entry.id, label, "correct", Icons.check_circle_outline, Colors.green, currentFeedback == "correct"),
              const SizedBox(width: 6),
              _feedbackButton(entry.id, label, "wrong", Icons.cancel_outlined, Colors.red, currentFeedback == "wrong"),
              const SizedBox(width: 6),
              _feedbackButton(entry.id, label, "not_sure", Icons.help_outline, Colors.orange, currentFeedback == "not_sure"),
            ],
          ),
        ],
      ),
    );
  }

  Widget _feedbackButton(
    String eventId,
    String label,
    String val,
    IconData icon,
    Color activeColor,
    bool isSelected,
  ) {
    return InkWell(
      onTap: () => onSubmitFeedback(eventId, label, val),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: isSelected
            ? monitorPressed(radius: 8).copyWith(color: activeColor.withValues(alpha: 0.15))
            : monitorRaised(radius: 8),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Icon(
              icon,
              size: 12,
              color: isSelected ? activeColor : monitorOnSurfaceVariant,
            ),
            const SizedBox(width: 4),
            Text(
              val == "not_sure" ? "Not Sure" : val[0].toUpperCase() + val.substring(1),
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.w600,
                color: isSelected ? activeColor : monitorOnSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
