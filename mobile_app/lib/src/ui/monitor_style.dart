import "package:flutter/material.dart";

const Color monitorSurface = Color(0xFFE8EAF0);
const Color monitorSurfaceLow = Color(0xFFF0F2F8);
const Color monitorOnSurface = Color(0xFF2E3040);
const Color monitorOnSurfaceVariant = Color(0xFF585A68);
const Color monitorPrimary = Color(0xFF6366F1);
const Color monitorPrimarySoft = Color(0xFF818CF8);

BoxDecoration monitorRaised({double radius = 16}) {
  return BoxDecoration(
    color: monitorSurface,
    borderRadius: BorderRadius.circular(radius),
    boxShadow: const <BoxShadow>[
      BoxShadow(
        color: Color(0x14000000),
        offset: Offset(6, 6),
        blurRadius: 12,
      ),
      BoxShadow(
        color: Color(0x99FFFFFF),
        offset: Offset(-6, -6),
        blurRadius: 12,
      ),
    ],
  );
}

BoxDecoration monitorPressed({double radius = 16}) {
  return BoxDecoration(
    color: monitorSurfaceLow,
    borderRadius: BorderRadius.circular(radius),
    border: Border.all(color: const Color(0x66FFFFFF)),
    boxShadow: const <BoxShadow>[
      BoxShadow(
        color: Color(0x0D000000),
        offset: Offset(2, 2),
        blurRadius: 4,
      ),
    ],
  );
}

String monitorTimeAgo(DateTime timestamp) {
  final Duration diff = DateTime.now().difference(timestamp);
  if (diff.inMinutes < 1) {
    return "just now";
  }
  if (diff.inHours < 1) {
    return "${diff.inMinutes}m ago";
  }
  return "${diff.inHours}h ago";
}

String monitorLabelForMode(String mode) {
  return "${mode[0].toUpperCase()}${mode.substring(1)}";
}
