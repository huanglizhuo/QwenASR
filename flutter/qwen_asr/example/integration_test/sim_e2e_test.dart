// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:qwen_asr/qwen_asr.dart';
import 'package:qwen_asr_example/download_screen.dart';
import 'package:qwen_asr_example/model_manager.dart';
import 'package:qwen_asr_example/streaming_screen.dart';
import 'package:qwen_asr_example/wav_utils.dart';

/// End-to-end simulator test: real download UI against a local server, then
/// the real simulated-mic streaming UI, asserting the transcript matches the
/// bench reference.
///
/// A local HTTP server must serve the model on the port passed via
/// `--dart-define=MODEL_SERVER=http://127.0.0.1:PORT/` and the reference
/// transcript + sim WAV bytes via `--dart-define` paths (host FS, shared with
/// the simulator).
///
/// Run:
///   flutter test integration_test/sim_e2e_test.dart -d SIM_UDID \
///     --dart-define=MODEL_SERVER=http://127.0.0.1:8000/ \
///     --dart-define=REF_TXT=/abs/path/audio.txt \
///     --dart-define=SIM_WAV=/abs/path/audio.wav
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const serverUrl = String.fromEnvironment('MODEL_SERVER');
  const refTxtPath = String.fromEnvironment('REF_TXT');
  const simWavPath = String.fromEnvironment('SIM_WAV');

  late QAsrEngine engine;

  testWidgets('A. download model via real DownloadScreen UI (local server)', (
    tester,
  ) async {
    expect(serverUrl, isNotEmpty, reason: 'pass --dart-define=MODEL_SERVER');
    await ModelManager.deleteModel();

    var done = false;
    await tester.pumpWidget(
      MaterialApp(
        home: DownloadScreen(
          initialBaseUrl: serverUrl,
          autoStart: true,
          onComplete: () => done = true,
        ),
      ),
    );

    // Poll up to 5 minutes for the download to finish (localhost is fast; the
    // model is ~1.8 GB).
    final deadline = DateTime.now().add(const Duration(minutes: 5));
    while (!done && DateTime.now().isBefore(deadline)) {
      await tester.pump(const Duration(milliseconds: 500));
      await Future<void>.delayed(const Duration(milliseconds: 200));
    }
    expect(done, isTrue, reason: 'download did not complete in time');
    expect(await ModelManager.isModelReady(), isTrue);
    print('DOWNLOAD: model ready in app documents');
  });

  testWidgets('B. model loads from app documents', (tester) async {
    final dir = await ModelManager.modelDir();
    engine = await QAsrEngine.load(dir, verbosity: 0);
    expect(engine, isNotNull);
    print('LOAD: engine loaded from $dir');
  });

  testWidgets('C. simulated-mic streaming matches reference (real UI)', (
    tester,
  ) async {
    // Provision sim.wav into app documents from the host-shared path.
    final docs = await ModelManager.modelDir();
    final docsRoot = Directory(docs).parent.path;
    final simDest = File('$docsRoot/sim.wav');
    expect(File(simWavPath).existsSync(), isTrue, reason: 'SIM_WAV not found');
    await simDest.writeAsBytes(File(simWavPath).readAsBytesSync());

    final refText = File(refTxtPath).readAsStringSync().trim();

    await tester.pumpWidget(
      MaterialApp(
        home: StreamingScreen(engine: engine, autoSimWavPath: simDest.path),
      ),
    );

    // The screen auto-runs the simulated-mic path (real-time paced ~28s).
    final transcriptFinder = find.byKey(const Key('transcript_text'));
    final deadline = DateTime.now().add(const Duration(minutes: 3));
    String current = '';
    while (DateTime.now().isBefore(deadline)) {
      await tester.pump(const Duration(milliseconds: 500));
      await Future<void>.delayed(const Duration(milliseconds: 300));
      final widget = tester.widget<SelectableText>(transcriptFinder);
      // The transcript is a SelectableText.rich: stable text + grey provisional
      // tail. At "Done" the provisional tail is empty, so the flattened plain
      // text equals the final stable transcript.
      current = widget.textSpan?.toPlainText() ?? widget.data ?? '';
      final statusDone = find
          .textContaining('Done (simulated)')
          .evaluate()
          .isNotEmpty;
      if (statusDone && current.trim() == refText) break;
    }
    print('STREAM final: "$current"');
    expect(current.trim(), equals(refText));
  });

  testWidgets('D. real-mic sanity: engine survives live-like noise', (
    tester,
  ) async {
    await engine.streamReset();
    engine.setStreamChunkSec(kSimEngineChunkSec);
    engine.setStreamUnfixedChunks(kUnfixedChunks);
    engine.setPastTextConditioning(true);
    // Push 3s of low-level noise in 0.5s chunks like the mic path would.
    final rnd = List<double>.generate(
      (0.5 * kEngineSampleRate).round(),
      (i) => ((i * 31 % 200) - 100) / 3000.0,
    );
    String out = '';
    for (var c = 0; c < 6; c++) {
      final res = await engine.streamPush(
        Float32List.fromList(rnd),
        finalize: c == 5,
      );
      out = res.text;
    }
    expect(out, isA<String>());
    print('REAL-MIC SANITY: no crash, out.len=${out.length}');
  });
}
