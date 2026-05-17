import 'package:flutter_test/flutter_test.dart';

import 'package:soundpulse_mobile/main.dart';

void main() {
  testWidgets('renders monitoring shell', (WidgetTester tester) async {
    await tester.pumpWidget(const SoundPulseApp());

    expect(find.text('Sound Pulse'), findsOneWidget);
    expect(find.text('Live Feed'), findsOneWidget);
    expect(find.text('View Recent Sounds'), findsOneWidget);
  });
}
