import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:soundpulse/main.dart';
import 'package:soundpulse/src/services/context_manager.dart';
import 'package:soundpulse/src/models/inference_event.dart';

void main() {
  testWidgets('renders monitoring shell', (WidgetTester tester) async {
    await tester.pumpWidget(const SoundPulseApp());

    expect(find.text('Sound Pulse'), findsOneWidget);
    expect(find.text('Live Feed'), findsOneWidget);
    expect(find.text('View Recent Sounds'), findsOneWidget);
  });

  testWidgets('personalisation settings card and gating limits work', (WidgetTester tester) async {
    await tester.pumpWidget(const SoundPulseApp());

    // Navigate to Record tab
    await tester.tap(find.text('Record'));
    await tester.pumpAndSettle();

    // Verify Personalisation Mode is displayed
    expect(find.text('Personalisation Mode'), findsOneWidget);
    expect(find.text('Default Setup'), findsOneWidget);
    expect(find.text('Quick Personalisation'), findsOneWidget);
    expect(find.text('Advanced Personalisation'), findsOneWidget);

    // Default setup is active on start, check gating banner
    expect(find.text('Personalisation is Disabled'), findsOneWidget);

    // Tap on Quick Personalisation
    await tester.tap(find.text('Quick Personalisation'));
    await tester.pumpAndSettle();

    // Verify that the gating banner is gone and record button is present
    expect(find.text('Personalisation is Disabled'), findsNothing);
    expect(find.text('Ready to record a new sample'), findsOneWidget);

    // Verify dynamic limit is displayed for Quick
    expect(find.text('0/3 samples saved for Crying baby · minimum 1'), findsOneWidget);

    // Tap on Advanced Personalisation
    await tester.tap(find.text('Advanced Personalisation'));
    await tester.pumpAndSettle();

    // Verify dynamic limit is displayed for Advanced
    expect(find.text('0/5 samples saved for Crying baby · minimum 3'), findsOneWidget);
  });

  testWidgets('adding custom sound class works', (WidgetTester tester) async {
    await tester.pumpWidget(const SoundPulseApp());

    // Navigate to Record tab
    await tester.tap(find.text('Record'));
    await tester.pumpAndSettle();

    // Tap on Quick Personalisation to enable controls
    await tester.tap(find.text('Quick Personalisation'));
    await tester.pumpAndSettle();

    // Verify dynamic limit is displayed for built-in Crying baby first
    expect(find.text('0/3 samples saved for Crying baby · minimum 1'), findsOneWidget);

    // Ensure the add custom class button is visible by scrolling it into view
    await tester.ensureVisible(find.byIcon(Icons.add));
    await tester.pumpAndSettle();

    // Tap Add Custom Sound Class button
    await tester.tap(find.byIcon(Icons.add));
    await tester.pumpAndSettle();

    // Verify Add Dialog is open
    expect(find.text('Add Custom Sound'), findsOneWidget);

    // Enter name "Microwave beep"
    await tester.enterText(find.byType(TextField), 'Microwave beep');
    await tester.pumpAndSettle();

    // Click Add
    await tester.tap(find.text('Add'));
    await tester.pumpAndSettle();

    // Verify that the dropdown selection has switched to the new Microwave beep class
    expect(find.text('0/3 samples saved for Microwave beep · minimum 1'), findsOneWidget);
  });

  group('ContextManager & Alert Policy Tests', () {
    late ContextManager manager;

    setUp(() {
      manager = ContextManager();
    });

    test('sound mapping settings work', () {
      // Clock alarm matches fire alarm settings (Critical, criticalFullScreen)
      final fireDec = manager.decideAlert(
        event: InferenceEvent(label: "fire_alarm", confidenceTier: "high", confidence: 0.9, isUnknown: false, isBackground: false),
        activeMode: "general",
      );
      final clockDec = manager.decideAlert(
        event: InferenceEvent(label: "clock_alarm", confidenceTier: "high", confidence: 0.9, isUnknown: false, isBackground: false),
        activeMode: "general",
      );
      expect(fireDec.priority, EventPriority.critical);
      expect(fireDec.modality, AlertModality.criticalFullScreen);
      expect(clockDec.priority, EventPriority.critical);
      expect(clockDec.modality, AlertModality.criticalFullScreen);

      // Doorbell & Door wood creaking match door_gate_knock settings (Important, vibrationFlashText)
      final knockDec = manager.decideAlert(
        event: InferenceEvent(label: "door_knock", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "general",
      );
      final bellDec = manager.decideAlert(
        event: InferenceEvent(label: "doorbell", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "general",
      );
      final creakDec = manager.decideAlert(
        event: InferenceEvent(label: "door_wood_creaking", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "general",
      );
      expect(knockDec.priority, EventPriority.important);
      expect(knockDec.modality, AlertModality.vibrationFlashText);
      expect(bellDec.priority, EventPriority.important);
      expect(bellDec.modality, AlertModality.vibrationFlashText);
      expect(creakDec.priority, EventPriority.important);
      expect(creakDec.modality, AlertModality.vibrationFlashText);

      // Dog barking uses medium_high sound importance settings (Important, standardVibrationText)
      final dogDec = manager.decideAlert(
        event: InferenceEvent(label: "dog_barking", confidenceTier: "high", confidence: 0.85, isUnknown: false, isBackground: false),
        activeMode: "general",
      );
      expect(dogDec.priority, EventPriority.important);
      expect(dogDec.modality, AlertModality.standardVibrationText);
    });

    test('Sleep Mode suppresses useful and low priority alerts', () {
      // Useful sound like water_running should be suppressed in Sleep Mode
      final sleepUseful = manager.decideAlert(
        event: InferenceEvent(label: "water_running", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "sleep",
      );
      expect(sleepUseful.suppressed, isTrue);
      expect(sleepUseful.modality, AlertModality.silentLog);

      // Critical sound like fire_alarm should NOT be suppressed in Sleep Mode
      final sleepCritical = manager.decideAlert(
        event: InferenceEvent(label: "fire_alarm", confidenceTier: "high", confidence: 0.9, isUnknown: false, isBackground: false),
        activeMode: "sleep",
      );
      expect(sleepCritical.suppressed, isFalse);
      expect(sleepCritical.modality, AlertModality.criticalFullScreen);
    });

    test('Focus Mode mutes non-critical alerts', () {
      // Important sound like door_knock should be suppressed in Focus Mode
      final focusImportant = manager.decideAlert(
        event: InferenceEvent(label: "door_knock", confidenceTier: "high", confidence: 0.85, isUnknown: false, isBackground: false),
        activeMode: "focus",
      );
      expect(focusImportant.suppressed, isTrue);
      expect(focusImportant.modality, AlertModality.silentLog);

      // Critical sound like clock_alarm should NOT be suppressed in Focus Mode
      final focusCritical = manager.decideAlert(
        event: InferenceEvent(label: "clock_alarm", confidenceTier: "high", confidence: 0.95, isUnknown: false, isBackground: false),
        activeMode: "focus",
      );
      expect(focusCritical.suppressed, isFalse);
      expect(focusCritical.modality, AlertModality.criticalFullScreen);
    });

    test('Deduplication Rate Limiting limits repeated alerts', () {
      // First occurrence of door_knock triggers haptic vibration
      final first = manager.decideAlert(
        event: InferenceEvent(label: "door_knock", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "general",
      );
      expect(first.suppressed, isFalse);

      // Immediate second occurrence within rate limit window (45s) should be suppressed
      final second = manager.decideAlert(
        event: InferenceEvent(label: "door_knock", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "general",
      );
      expect(second.suppressed, isTrue);
      expect(second.modality, AlertModality.silentLog);
    });

    test('Temporal Event-Chain Sequence Boosts work', () {
      // Event 1: door_knock is added to recent buffer
      manager.decideAlert(
        event: InferenceEvent(label: "door_knock", confidenceTier: "high", confidence: 0.7, isUnknown: false, isBackground: false),
        activeMode: "general",
      );

      // Event 2: speech immediately follows door_knock
      final speechDec = manager.decideAlert(
        event: InferenceEvent(label: "speech", confidenceTier: "medium", confidence: 0.6, isUnknown: false, isBackground: false),
        activeMode: "general",
      );

      // Verify that speech gets a +0.05 confidence boost and important priority with vibrationFlashText
      expect(speechDec.chainBoostApplied, isTrue);
      expect(speechDec.confidence, closeTo(0.65, 0.001));
      expect(speechDec.priority, EventPriority.important);
      expect(speechDec.modality, AlertModality.vibrationFlashText);
    });
  });

  group('Stage G Settings, Enablement Gating & Overrides Tests', () {
    late ContextManager manager;

    setUp(() {
      manager = ContextManager();
    });

    test('sound enablement gating suppresses alerts completely', () {
      final dec = manager.decideAlert(
        event: InferenceEvent(label: "crying_baby", confidenceTier: "high", confidence: 0.9, isUnknown: false, isBackground: false),
        activeMode: "general",
        userOverrides: {"enabled": false},
      );
      expect(dec.suppressed, isTrue);
      expect(dec.modality, AlertModality.none);
      expect(dec.suppressionReason, contains("disabled"));
    });

    test('mode-specific user override bypasses default mode suppression rules', () {
      // Sleep mode normally suppresses useful sounds like water_running
      final sleepDefault = manager.decideAlert(
        event: InferenceEvent(label: "water_running", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "sleep",
      );
      expect(sleepDefault.suppressed, isTrue);
      expect(sleepDefault.modality, AlertModality.silentLog);

      // User adds mode-specific override to force sound to alert in Sleep Mode
      final sleepOverridden = manager.decideAlert(
        event: InferenceEvent(label: "water_running", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "sleep",
        userOverrides: {
          "enabled": true,
          "modes": {"sleep": "standard_vibration_text"}
        },
      );
      expect(sleepOverridden.suppressed, isFalse);
      expect(sleepOverridden.modality, AlertModality.standardVibrationText);
    });

    test('Cooking Mode baseline context policy', () {
      // Dog barking is not cooking-relevant, should be suppressed in Cooking Mode
      final dogCooking = manager.decideAlert(
        event: InferenceEvent(label: "dog_barking", confidenceTier: "high", confidence: 0.85, isUnknown: false, isBackground: false),
        activeMode: "cooking",
      );
      expect(dogCooking.suppressed, isTrue);
      expect(dogCooking.modality, AlertModality.silentLog);

      // Water running is cooking-relevant, should NOT be suppressed in Cooking Mode
      final waterCooking = manager.decideAlert(
        event: InferenceEvent(label: "water_running", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "cooking",
      );
      expect(waterCooking.suppressed, isFalse);
      expect(waterCooking.modality, AlertModality.standardVibrationText);
    });

    test('Bathroom Mode baseline context policy', () {
      // Water running is expected in Bathroom Mode, should be suppressed
      final waterBathroom = manager.decideAlert(
        event: InferenceEvent(label: "water_running", confidenceTier: "high", confidence: 0.8, isUnknown: false, isBackground: false),
        activeMode: "bathroom",
      );
      expect(waterBathroom.suppressed, isTrue);
      expect(waterBathroom.modality, AlertModality.silentLog);

      // Crying baby should still be heard in Bathroom Mode
      final babyBathroom = manager.decideAlert(
        event: InferenceEvent(label: "crying_baby", confidenceTier: "high", confidence: 0.9, isUnknown: false, isBackground: false),
        activeMode: "bathroom",
      );
      expect(babyBathroom.suppressed, isFalse);
      expect(babyBathroom.modality, AlertModality.vibrationFlashText);
    });

    testWidgets('settings UI navigation toggle works', (WidgetTester tester) async {
      await tester.pumpWidget(const SoundPulseApp());

      // Tap settings button in top bar to open settings view
      await tester.tap(find.byIcon(Icons.settings));
      await tester.pumpAndSettle();

      // Verify settings view is rendered
      expect(find.text('Settings & Context Rules'), findsOneWidget);
      expect(find.text('Context-Aware Routing'), findsOneWidget);

      // Tap settings back button to return to previous tab
      await tester.tap(find.byIcon(Icons.arrow_back).first);
      await tester.pumpAndSettle();

      // Verify we are back to main monitoring view
      expect(find.text('Settings & Context Rules'), findsNothing);
      expect(find.text('Sound Pulse'), findsOneWidget);
    });
  });
}

