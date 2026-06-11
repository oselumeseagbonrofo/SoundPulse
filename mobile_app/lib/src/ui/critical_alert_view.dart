import "dart:async";
import "package:flutter/material.dart";
import "../runtime/runtime_bridge.dart";
import "monitor_style.dart";

class CriticalAlertView extends StatefulWidget {
  const CriticalAlertView({
    super.key,
    required this.soundLabel,
    this.onDismiss,
  });

  final String soundLabel;
  final VoidCallback? onDismiss;

  @override
  State<CriticalAlertView> createState() => _CriticalAlertViewState();
}

class _CriticalAlertViewState extends State<CriticalAlertView> with SingleTickerProviderStateMixin {
  final RuntimeBridge _runtimeBridge = RuntimeBridge();
  late AnimationController _pulseController;
  Timer? _vibrationTimer;

  @override
  void initState() {
    super.initState();

    // Set up pulsing alert animation
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat(reverse: true);

    // Trigger initial vibration and start periodic timer for continuous haptics
    _triggerVibration();
    _vibrationTimer = Timer.periodic(const Duration(seconds: 2), (_) {
      _triggerVibration();
    });
  }

  void _triggerVibration() {
    unawaited(_runtimeBridge.vibrate("long"));
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _vibrationTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final String cleanLabel = formatSoundLabel(widget.soundLabel);

    return PopScope(
      canPop: false,
      child: Scaffold(
        body: AnimatedBuilder(
          animation: _pulseController,
          builder: (context, child) {
            final double t = _pulseController.value;
            return Container(
              width: double.infinity,
              height: double.infinity,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: <Color>[
                    Color.lerp(const Color(0xFFEF4444), const Color(0xFFF97316), t)!,
                    Color.lerp(const Color(0xFFB91C1C), const Color(0xFFC2410C), t)!,
                  ],
                ),
              ),
              child: child,
            );
          },
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 36.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: <Widget>[
                  const SizedBox(height: 24),
                  // Alert Info
                  Column(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      const Icon(
                        Icons.warning_amber_rounded,
                        color: Colors.white,
                        size: 96,
                      ),
                      const SizedBox(height: 24),
                      const Text(
                        "CRITICAL ALERT DETECTED",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 2.0,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        cleanLabel.toUpperCase(),
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 32,
                          fontWeight: FontWeight.w900,
                        ),
                      ),
                    ],
                  ),
                  // Dismiss Button
                  GestureDetector(
                    onTap: () {
                      if (widget.onDismiss != null) {
                        widget.onDismiss!();
                      } else {
                        Navigator.of(context).pop();
                      }
                    },
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 20),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(30),
                        boxShadow: const <BoxShadow>[
                          BoxShadow(
                            color: Color(0x33000000),
                            blurRadius: 10,
                            offset: Offset(0, 5),
                          ),
                        ],
                      ),
                      alignment: Alignment.center,
                      child: const Text(
                        "DISMISS ALARM",
                        style: TextStyle(
                          color: Color(0xFFB91C1C),
                          fontSize: 16,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 1.0,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
