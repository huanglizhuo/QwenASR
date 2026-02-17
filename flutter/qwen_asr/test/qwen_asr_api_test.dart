import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:qwen_asr/qwen_asr.dart';

/// Pure Dart API tests — no app, no UI.
///
/// Prerequisites:
///   cd flutter/qwen_asr/rust && cargo build --release
///
/// Run:
///   cd flutter/qwen_asr && flutter test test/qwen_asr_api_test.dart
void main() {
  const projectRoot = '/Users/lizhuo/owork/q-asr';
  const modelDir = '$projectRoot/qwen3-asr-0.6b';
  const wavPath = '$projectRoot/bench/samples/audio.wav';
  const refPath = '$projectRoot/bench/samples/audio.txt';
  const dylibPath =
      '$projectRoot/flutter/qwen_asr/rust/target/release/librust_lib_qwen_asr.dylib';

  late QAsrEngine engine;

  setUpAll(() async {
    await QAsrEngine.initWith(dylibPath: dylibPath);
    engine = await QAsrEngine.load(modelDir, verbosity: 1);
  });

  tearDownAll(() {
    engine.dispose();
  });

  test('load model succeeds', () {
    expect(engine, isNotNull);
  });

  test('transcribeFile returns non-empty result', () async {
    final result = await engine.transcribeFile(wavPath);
    expect(result, isNotEmpty);
    print('transcribeFile: $result');
  });

  test('transcribeWavBuffer matches reference text', () async {
    final wavBytes = File(wavPath).readAsBytesSync();
    final refText = File(refPath).readAsStringSync().trim();

    final result =
        await engine.transcribeWavBuffer(Uint8List.fromList(wavBytes));
    expect(result, isNotEmpty);
    print('transcribeWavBuffer: $result');

    expect(result.toLowerCase(), contains('shenyang'));
    expect(result.toLowerCase(), contains('disappointing'));
    expect(result, equals(refText));
  });

  test('transcribePcm accepts raw samples', () async {
    final silence = Float32List(16000); // 1s silence
    final result = await engine.transcribePcm(silence);
    expect(result, isA<String>());
    print('transcribePcm (silence): "$result"');
  });

  test('setLanguage valid', () {
    expect(engine.setLanguage('English'), isTrue);
  });

  test('setLanguage invalid', () {
    expect(engine.setLanguage('InvalidLanguage'), isFalse);
  });

  test('setLanguage empty resets', () {
    expect(engine.setLanguage(''), isTrue);
  });

  test('setSegmentSec + transcribe', () async {
    engine.setSegmentSec(30);
    final wavBytes = File(wavPath).readAsBytesSync();
    final result =
        await engine.transcribeWavBuffer(Uint8List.fromList(wavBytes));
    expect(result, isNotEmpty);
    print('segmented: $result');
    engine.setSegmentSec(0);
  });

  test('perfStats format', () {
    final stats = engine.perfStats();
    expect(stats, contains('audio='));
    expect(stats, contains('encode='));
    expect(stats, contains('decode='));
    expect(stats, contains('tokens='));
    print('perfStats: $stats');
  });
}
