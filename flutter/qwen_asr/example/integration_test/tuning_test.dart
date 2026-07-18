// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:qwen_asr/qwen_asr.dart';
import 'package:qwen_asr_example/wav_utils.dart';

/// On-simulator tuning sweep: chunk size, thread count, realtime factor,
/// first-partial latency. Requires the model already provisioned in app docs
/// (run sim_e2e_test.dart first) and SIM_WAV pointing at the host bench clip.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  const simWavPath = String.fromEnvironment('SIM_WAV');
  const refTxtPath = String.fromEnvironment('REF_TXT');
  // Host model dir (simulator shares the Mac filesystem).
  const modelDirPath = String.fromEnvironment('MODEL_DIR');

  Future<(String, double, int?)> run(
    QAsrEngine engine,
    Float32List samples,
    double micChunkSec,
    double engineChunkSec,
  ) async {
    await engine.streamReset();
    engine.setStreamChunkSec(engineChunkSec);
    engine.setStreamUnfixedChunks(2);
    engine.setPastTextConditioning(true);
    final chunk = (micChunkSec * kEngineSampleRate).round();
    final sw = Stopwatch()..start();
    var last = '';
    int? firstMs;
    for (var i = 0; i < samples.length; i += chunk) {
      final end = (i + chunk) > samples.length ? samples.length : i + chunk;
      last = await engine.streamPush(
        Float32List.sublistView(samples, i, end),
        finalize: end >= samples.length,
      );
      if (last.isNotEmpty && firstMs == null) firstMs = sw.elapsedMilliseconds;
    }
    final audioSec = samples.length / kEngineSampleRate;
    return (last, audioSec / (sw.elapsedMilliseconds / 1000.0), firstMs);
  }

  testWidgets('tuning sweep on simulator', (tester) async {
    final samples = parseWavTo16kMono(File(simWavPath).readAsBytesSync());
    final refText = File(refTxtPath).readAsStringSync().trim();
    final dir = modelDirPath;

    for (final threads in [0, 4]) {
      final engine = await QAsrEngine.load(dir, threads: threads);
      for (final ec in [1.0, 4.0, 8.0]) {
        final (text, rtf, firstMs) = await run(engine, samples, 0.5, ec);
        print(
          'THREADS=$threads engineChunk=${ec}s rtf=${rtf.toStringAsFixed(2)}x '
          'firstPartial=${firstMs}ms match=${text.trim() == refText}',
        );
      }
      engine.dispose();
    }
  });
}
