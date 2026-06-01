import "package:flutter/material.dart";
import "package:hive_flutter/hive_flutter.dart";

import "src/ui/monitor_page.dart";

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Hive.initFlutter();
  await Hive.openBox<dynamic>("soundpulse.settings");
  await Hive.openBox<dynamic>("soundpulse.class_config");
  await Hive.openBox<dynamic>("soundpulse.prototype");
  await Hive.openBox<dynamic>("soundpulse.threshold");
  await Hive.openBox<dynamic>("soundpulse.mode_policy");
  await Hive.openBox<dynamic>("soundpulse.event_history");
  await Hive.openBox<dynamic>("soundpulse.feedback");
  await Hive.openBox<dynamic>("soundpulse.recording_metadata");
  runApp(const SoundPulseApp());
}

class SoundPulseApp extends StatelessWidget {
  const SoundPulseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "SoundPulse",
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6366F1),
          surface: const Color(0xFFE8EAF0),
        ),
        scaffoldBackgroundColor: const Color(0xFFE8EAF0),
        fontFamily: "Plus Jakarta Sans",
        useMaterial3: true,
      ),
      home: const MonitorPage(),
    );
  }
}
