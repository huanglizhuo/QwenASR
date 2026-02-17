import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:qwen_asr/qwen_asr.dart';

/// Pure API integration tests — no widgets, no UI.
/// Tests the Rust bridge directly via QAsrEngine.
///
/// Run:  flutter test integration_test/qwen_asr_test.dart -d macos
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Paths — adjust if your layout differs.
  final projectRoot = '/Users/lizhuo/owork/q-asr';
  final modelDir = '$projectRoot/qwen3-asr-0.6b';
  final wavPath = '$projectRoot/bench/samples/audio.wav';
  final refPath = '$projectRoot/bench/samples/audio.txt';

  late QAsrEngine engine;

  setUpAll(() async {
    engine = await QAsrEngine.load(modelDir, verbosity: 1);
  });

  tearDownAll(() {
    engine.dispose();
  });

  // --- Core API tests ---

  test('load model succeeds', () {
    expect(engine, isNotNull);
  });

  test('transcribeFile returns non-empty result', () async {
    final result = await engine.transcribeFile(wavPath);
    expect(result, isNotEmpty);
    print('transcribeFile result: $result');
  });

  test('transcribeWavBuffer matches reference text', () async {
    final wavBytes = File(wavPath).readAsBytesSync();
    final refText = File(refPath).readAsStringSync().trim();

    final result = await engine.transcribeWavBuffer(Uint8List.fromList(wavBytes));
    expect(result, isNotEmpty);
    print('transcribeWavBuffer result: $result');

    // Key phrases check (robust against minor wording differences)
    expect(result.toLowerCase(), contains('shenyang'));
    expect(result.toLowerCase(), contains('disappointing'));

    // Exact match
    expect(result, equals(refText));
  });

  test('transcribePcm works with raw samples', () async {
    // Read WAV, parse to PCM via the engine's wav buffer path,
    // but here we just verify the API accepts Float32List.
    // Use a short silence buffer to keep it fast.
    final silence = Float32List(16000); // 1 second of silence
    final result = await engine.transcribePcm(silence);
    // Silence should produce empty or very short output
    expect(result, isA<String>());
    print('transcribePcm (silence) result: "$result"');
  });

  // --- Configuration tests ---

  test('setLanguage accepts valid language', () {
    expect(engine.setLanguage('English'), isTrue);
  });

  test('setLanguage rejects invalid language', () {
    expect(engine.setLanguage('InvalidLanguage'), isFalse);
  });

  test('setLanguage empty resets', () {
    expect(engine.setLanguage(''), isTrue);
  });

  test('setSegmentSec + transcribe works', () async {
    engine.setSegmentSec(30);
    final wavBytes = File(wavPath).readAsBytesSync();
    final result = await engine.transcribeWavBuffer(Uint8List.fromList(wavBytes));
    expect(result, isNotEmpty);
    print('segmented result: $result');
    engine.setSegmentSec(0); // reset
  });

  test('perfStats returns formatted string', () {
    final stats = engine.perfStats();
    expect(stats, contains('audio='));
    expect(stats, contains('encode='));
    expect(stats, contains('decode='));
    expect(stats, contains('tokens='));
    print('perfStats: $stats');
  });
}
