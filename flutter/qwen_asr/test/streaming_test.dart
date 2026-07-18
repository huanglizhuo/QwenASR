// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:qwen_asr/qwen_asr.dart';

/// Streaming-API tests + a chunk-size / thread tuning sweep.
///
/// Prerequisites:
///   cd flutter/qwen_asr/rust && cargo build --release
///
/// Run:
///   cd flutter/qwen_asr && flutter test test/streaming_test.dart
void main() {
  const projectRoot = '/Users/lizhuo/owork/q-asr';
  const modelDir = '$projectRoot/qwen3-asr-0.6b';
  const wavPath = '$projectRoot/bench/samples/audio.wav';
  const refPath = '$projectRoot/bench/samples/audio.txt';
  const dylibPath =
      '$projectRoot/flutter/qwen_asr/rust/target/release/librust_lib_qwen_asr.dylib';
  const sampleRate = 16000;

  late QAsrEngine engine;
  late Float32List samples;
  late String refText;

  // Thread count for the loaded engine (0 = auto). Override per run with
  // --dart-define=THREADS=N to measure thread scaling.
  const threads = int.fromEnvironment('THREADS', defaultValue: 0);

  setUpAll(() async {
    await QAsrEngine.initWith(dylibPath: dylibPath);
    engine = await QAsrEngine.load(modelDir, threads: threads, verbosity: 0);
    samples = _parseWavPcm16(File(wavPath).readAsBytesSync());
    refText = File(refPath).readAsStringSync().trim();
  });

  tearDownAll(() => engine.dispose());

  /// Feed the whole clip through the streaming pipeline in [micChunkSec]-sized
  /// pushes. Returns (finalTranscript, realtimeFactor, firstPartialMs).
  Future<(String, double, int?)> runStream({
    required double micChunkSec,
    required double engineChunkSec,
    int unfixedChunks = 2,
    bool pastText = true,
  }) async {
    await engine.streamReset();
    engine.setStreamChunkSec(engineChunkSec);
    engine.setStreamUnfixedChunks(unfixedChunks);
    engine.setPastTextConditioning(pastText);
    final chunk = (micChunkSec * sampleRate).round();
    final sw = Stopwatch()..start();
    var last = '';
    int? firstPartialMs;
    for (var i = 0; i < samples.length; i += chunk) {
      final end = (i + chunk) > samples.length ? samples.length : i + chunk;
      final finalize = end >= samples.length;
      final res = await engine.streamPush(
        Float32List.sublistView(samples, i, end),
        finalize: finalize,
      );
      last = res.text;
      // First visible text = stable OR provisional becomes non-empty.
      if ((res.text.isNotEmpty || res.provisional.isNotEmpty) &&
          firstPartialMs == null) {
        firstPartialMs = sw.elapsedMilliseconds;
      }
    }
    sw.stop();
    final audioSec = samples.length / sampleRate;
    final rtf = audioSec / (sw.elapsedMilliseconds / 1000.0);
    return (last, rtf, firstPartialMs);
  }

  test(
    'streaming final transcript matches reference (default tuning)',
    () async {
      final (text, rtf, firstMs) = await runStream(
        micChunkSec: 0.5,
        engineChunkSec: 8.0,
      );
      print(
        'STREAM default: rtf=${rtf.toStringAsFixed(2)}x '
        'firstPartial=${firstMs}ms',
      );
      print('STREAM text: $text');
      expect(text.trim(), equals(refText));
    },
  );

  test('THREADS measurement (release, one engine per run)', () async {
    final (text, rtf, firstMs) = await runStream(
      micChunkSec: 0.5,
      engineChunkSec: 8.0,
    );
    print(
      'THREADS=${threads == 0 ? "auto" : threads} engineChunk=8.0s '
      'rtf=${rtf.toStringAsFixed(2)}x firstPartial=${firstMs}ms '
      'match=${text.trim() == refText}',
    );
  });

  test('TUNING SWEEP: engine chunk size', () async {
    print('\n=== TUNING SWEEP (macOS desktop, functional) ===');
    for (final engineChunk in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]) {
      final (text, rtf, firstMs) = await runStream(
        micChunkSec: 0.5,
        engineChunkSec: engineChunk,
      );
      final match = text.trim() == refText;
      print(
        'engineChunk=${engineChunk}s micChunk=0.5s '
        'rtf=${rtf.toStringAsFixed(2)}x firstPartial=${firstMs}ms '
        'match=$match len=${text.length}',
      );
      print('   text="$text"');
    }
  });
}

// Parse a PCM16 WAV to 16 kHz mono floats (downmix + linear resample).
Float32List _parseWavPcm16(Uint8List bytes) {
  const dstRate = 16000;
  final bd = ByteData.view(
    bytes.buffer,
    bytes.offsetInBytes,
    bytes.lengthInBytes,
  );
  var channels = 1;
  var srcRate = dstRate;
  var pos = 12;
  Float32List? mono;
  while (pos + 8 <= bytes.lengthInBytes) {
    final id = bd.getUint32(pos, Endian.big);
    final size = bd.getUint32(pos + 4, Endian.little);
    final body = pos + 8;
    if (id == 0x666d7420) {
      channels = bd.getUint16(body + 2, Endian.little);
      srcRate = bd.getUint32(body + 4, Endian.little);
    } else if (id == 0x64617461) {
      final frames = (size ~/ 2) ~/ channels;
      mono = Float32List(frames);
      for (var f = 0; f < frames; f++) {
        var acc = 0.0;
        for (var c = 0; c < channels; c++) {
          acc +=
              bd.getInt16(body + (f * channels + c) * 2, Endian.little) /
              32768.0;
        }
        mono[f] = acc / channels;
      }
      break;
    }
    pos = body + size + (size & 1);
  }
  if (mono == null) return Float32List(0);
  if (srcRate == dstRate) return mono;
  final outLen = (mono.length * dstRate / srcRate).floor();
  final out = Float32List(outLen);
  final ratio = srcRate / dstRate;
  for (var i = 0; i < outLen; i++) {
    final sp = i * ratio;
    final i0 = sp.floor();
    final i1 = (i0 + 1 < mono.length) ? i0 + 1 : i0;
    final frac = sp - i0;
    out[i] = mono[i0] * (1 - frac) + mono[i1] * frac;
  }
  return out;
}
