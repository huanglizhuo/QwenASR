// ignore_for_file: avoid_print
import 'dart:io';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:qwen_asr/qwen_asr.dart';
import 'package:qwen_asr_example/main.dart';
import 'package:qwen_asr_example/record_screen.dart';
import 'package:qwen_asr_example/streaming_screen.dart';
import 'package:qwen_asr_example/wav_utils.dart';

/// Widget integration tests for the R14 example-app features:
///   * three-tab shell (Live / Record / File)
///   * File tab: bundled-sample + pick-file buttons render
///   * Record tab: push-to-talk button + state machine (recording → transcribing
///     → done), driven by real pointer gestures against the real engine.
///
/// Runs on a device or the macOS host:
///   flutter test integration_test/features_test.dart -d `device|macos`
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const projectRoot = '/Users/lizhuo/owork/q-asr';
  const modelDir = '$projectRoot/qwen3-asr-0.6b';
  const wavPath = '$projectRoot/bench/samples/audio.wav';
  const mixedWavPath = '$projectRoot/bench/samples/mixed_zh_en.wav';

  bool hasCjk(String s) => s.runes.any((r) => r >= 0x4E00 && r <= 0x9FFF);
  bool hasAsciiWord(String s) => RegExp(r'[A-Za-z]{2,}').hasMatch(s);
  // The degeneration artifact = the same short phrase repeated many times.
  bool hasRepeatLoop(String s) {
    final words = s.toLowerCase().split(RegExp(r'\s+'));
    var maxRun = 0, run = 0;
    for (var i = 1; i < words.length; i++) {
      if (words[i] == words[i - 1] && words[i].isNotEmpty) {
        run++;
        if (run > maxRun) maxRun = run;
      } else {
        run = 0;
      }
    }
    return maxRun >= 4;
  }

  late QAsrEngine engine;

  setUpAll(() async {
    engine = await QAsrEngine.load(modelDir, verbosity: 0);
  });

  tearDownAll(() => engine.dispose());

  Future<void> pumpHome(WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(home: HomeTabs(engine: engine)));
    await tester.pumpAndSettle();
  }

  testWidgets('three tabs render (Live / Record / File)', (tester) async {
    await pumpHome(tester);
    expect(find.text('Live'), findsOneWidget);
    expect(find.text('Record'), findsOneWidget);
    expect(find.text('File'), findsOneWidget);
    expect(find.byType(Tab), findsNWidgets(3));
  });

  testWidgets('File tab exposes bundled-sample + pick-file buttons', (
    tester,
  ) async {
    await pumpHome(tester);
    await tester.tap(find.text('File'));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('transcribe_asset_button')), findsOneWidget);
    expect(find.byKey(const Key('pick_file_button')), findsOneWidget);
  });

  testWidgets('Live tab exposes the VAD-Live mode selector', (tester) async {
    await pumpHome(tester);
    // Live is the default tab.
    expect(find.byKey(const Key('live_mode_dropdown')), findsOneWidget);
    expect(find.byKey(const Key('live_mode_description')), findsOneWidget);
  });

  testWidgets('Record tab: PTT button renders and accepts a hold gesture', (
    tester,
  ) async {
    await pumpHome(tester);
    await tester.tap(find.text('Record'));
    await tester.pumpAndSettle();

    final ptt = find.byKey(const Key('ptt_button'));
    expect(ptt, findsOneWidget);
    expect(find.byKey(const Key('ptt_status')), findsOneWidget);

    // Press-and-release must not throw even where the mic platform channel is
    // unavailable (e.g. the host test harness). On a real device this begins a
    // recording; here we only assert the gesture wiring does not crash.
    final center = tester.getCenter(ptt);
    final gesture = await tester.startGesture(
      center,
      kind: PointerDeviceKind.touch,
    );
    await tester.pump(const Duration(milliseconds: 120));
    await gesture.up();
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });

  testWidgets(
    'Record tab: state machine transcribing → done (synthetic audio)',
    (tester) async {
      await pumpHome(tester);
      await tester.tap(find.text('Record'));
      await tester.pumpAndSettle();

      // Inject a real speech buffer straight into the transcribe path — this
      // exercises the recording→transcribing→done state machine and confirms
      // `transcribePcm` is invoked, without depending on live mic capture.
      final samples = parseWavTo16kMono(File(wavPath).readAsBytesSync());
      expect(samples, isNotEmpty);

      // `transcribeSamples` is a @visibleForTesting method on the private State;
      // reach it via dynamic dispatch (no access to the private type needed).
      final dynamic state = tester.state(find.byType(RecordScreen));
      final future = state.transcribeSamples(samples) as Future<void>;
      await tester.pump(); // let setState(_transcribing = true) flush

      final statusFinder = find.byKey(const Key('ptt_status'));
      final midStatus = (tester.widget<Text>(statusFinder)).data ?? '';
      print('PTT status mid-transcribe: $midStatus');
      expect(midStatus, contains('Transcribing'));

      await future;
      await tester.pumpAndSettle();

      final endStatus = (tester.widget<Text>(statusFinder)).data ?? '';
      final transcript =
          (tester.widget<SelectableText>(
            find.byKey(const Key('ptt_transcript')),
          )).data ??
          '';
      print('PTT status after transcribe: $endStatus');
      print('PTT transcript: $transcript');
      expect(endStatus, contains('Done'));
      // The bench clip transcribes to a known phrase — confirm real ASR output.
      expect(transcript.toLowerCase(), contains('shenyang'));
    },
  );

  testWidgets(
    'VAD Live: single English clip segments, decodes, appends (no degeneration)',
    (tester) async {
      await pumpHome(tester);
      // Live is the default tab; drive the mode-aware sim path (Task B) that the
      // real mic also feeds, exercising segmentation → queue → one-shot → append.
      final samples = parseWavTo16kMono(File(wavPath).readAsBytesSync());
      expect(samples, isNotEmpty);

      final dynamic state = tester.state(find.byType(StreamingScreen));
      final future = state.runVadLiveOnSamplesForTest(samples) as Future<void>;
      await tester.pump();
      await future;
      await tester.pumpAndSettle();

      final transcript = state.transcriptForTest as String;
      final done = state.utterancesDoneForTest as int;
      print('VAD-Live (audio.wav) utterances=$done transcript: $transcript');
      expect(done, greaterThanOrEqualTo(1));
      // Assert on unambiguous mid/late English content. (We deliberately do NOT
      // assert on the very first word "Shenyang": in auto-multilingual mode the
      // short leading fragment can be language-mis-detected on its own — an
      // inherent property of per-utterance auto-detection on a tiny fragment,
      // not a pipeline defect; forcing language='en' avoids it.)
      final lower = transcript.toLowerCase();
      expect(lower, contains('secrets'));
      expect(lower, contains('decision'));
      // The whole point of the redesign: no rolling-window repeat-loop artifact.
      expect(
        hasRepeatLoop(transcript),
        isFalse,
        reason: 'VAD-Live one-shot decode must not degenerate',
      );
    },
  );

  testWidgets('VAD Live: mixed zh/en clip switches language per utterance', (
    tester,
  ) async {
    if (!File(mixedWavPath).existsSync()) {
      print('mixed_zh_en.wav absent — skipping multilingual VAD-Live test');
      return;
    }
    await pumpHome(tester);
    final samples = parseWavTo16kMono(File(mixedWavPath).readAsBytesSync());
    expect(samples, isNotEmpty);

    final dynamic state = tester.state(find.byType(StreamingScreen));
    final future = state.runVadLiveOnSamplesForTest(samples) as Future<void>;
    await tester.pump();
    await future;
    await tester.pumpAndSettle();

    final transcript = state.transcriptForTest as String;
    final done = state.utterancesDoneForTest as int;
    print('VAD-Live (mixed) utterances=$done transcript: $transcript');
    // The clip has ~10 speech segments; expect several independent utterances.
    expect(done, greaterThanOrEqualTo(3));
    // Per-utterance auto-detect (multilingual, no forced preamble) must yield
    // BOTH Chinese characters and English words — proving per-utterance
    // language switching without vad_segment_reset.
    expect(
      hasCjk(transcript),
      isTrue,
      reason: 'Chinese utterances should render as Chinese, not translated',
    );
    expect(hasAsciiWord(transcript), isTrue);
    // And no repeat-loop degeneration (which the single-pass offline decode of
    // this same clip DOES exhibit).
    expect(hasRepeatLoop(transcript), isFalse);
  });
}
