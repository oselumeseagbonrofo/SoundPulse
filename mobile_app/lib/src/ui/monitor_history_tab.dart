import "package:flutter/material.dart";

import "../models/monitor_history_entry.dart";
import "monitor_style.dart";

class MonitorHistoryTab extends StatelessWidget {
  const MonitorHistoryTab({
    super.key,
    required this.history,
  });

  final List<MonitorHistoryEntry> history;

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
    return Container(
      margin: const EdgeInsets.only(top: 10),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: monitorRaised(radius: 14),
      child: Row(
        children: <Widget>[
          Expanded(
            child: Text(
              entry.event.label,
              style: const TextStyle(fontWeight: FontWeight.w600, color: monitorOnSurface),
            ),
          ),
          Text(
            "${(entry.event.confidence * 100).round()}%",
            style: const TextStyle(fontWeight: FontWeight.w600, color: monitorOnSurfaceVariant),
          ),
        ],
      ),
    );
  }
}
