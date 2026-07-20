import 'package:flutter_test/flutter_test.dart';
import 'package:qwen_asr_example/vad_live_pipeline.dart';

const int _sr = 16000;

/// One frame of `seconds` of samples (values irrelevant to the pipeline, which
/// only counts them — Silero's own detection is the package's test surface).
List<double> _frame(double seconds) =>
    List<double>.filled((seconds * _sr).round(), 0.1);

void main() {
  test('one utterance: start → frames → end appends once, in order', () async {
    final appended = <String>[];
    var n = 0;
    final p = VadLivePipeline(
      decode: (s) async => 'utt${n++}',
      onAppend: appended.add,
    );
    p.onSpeechStart();
    p.onFrame(_frame(0.5));
    p.onFrame(_frame(0.5));
    p.onSpeechEnd();
    await p.drainAll();
    expect(appended, ['utt0']);
    expect(p.appendedCount, 1);
  });

  test('empty decode result (too-short) is not appended', () async {
    final appended = <String>[];
    final p = VadLivePipeline(
      decode: (s) async => '', // e.g. OneShotTranscriber too-short guard
      onAppend: appended.add,
    );
    p.onSpeechStart();
    p.onFrame(_frame(0.1));
    p.onSpeechEnd();
    await p.drainAll();
    expect(appended, isEmpty);
    expect(p.appendedCount, 0);
  });

  test('misfire discards the in-progress buffer', () async {
    final appended = <String>[];
    final p = VadLivePipeline(decode: (s) async => 'x', onAppend: appended.add);
    p.onSpeechStart();
    p.onFrame(_frame(0.4));
    p.onMisfire();
    await p.drainAll();
    expect(appended, isEmpty);
  });

  test('max-duration safety cap force-closes an unbroken utterance', () async {
    final emittedLens = <int>[];
    final p = VadLivePipeline(
      decode: (s) async {
        emittedLens.add(s.length);
        return 'seg';
      },
      onAppend: (_) {},
      maxUtteranceSec: 2.0, // small cap for the test
    );
    p.onSpeechStart();
    // 5 s of continuous speech, no boundary → expect ~2 forced closes...
    for (var i = 0; i < 10; i++) {
      p.onFrame(_frame(0.5));
    }
    await p.drainAll(); // flush the tail
    // Two full 2 s caps (32000 samples) plus a 1 s tail.
    expect(emittedLens.length, 3);
    expect(emittedLens[0], (2.0 * _sr).round());
    expect(emittedLens[1], (2.0 * _sr).round());
    expect(emittedLens[2], (1.0 * _sr).round());
  });

  test(
    'FIFO order preserved and nothing dropped under a slow consumer',
    () async {
      final appended = <String>[];
      var idx = 0;
      final p = VadLivePipeline(
        decode: (s) async {
          // Simulate a slow consumer so utterances back up in the queue.
          await Future<void>.delayed(const Duration(milliseconds: 5));
          return 'u${idx++}';
        },
        onAppend: appended.add,
      );
      // Fire 5 complete utterances faster than the consumer drains.
      for (var k = 0; k < 5; k++) {
        p.onSpeechStart();
        p.onFrame(_frame(0.5));
        p.onSpeechEnd();
      }
      await p.drainAll();
      expect(appended, ['u0', 'u1', 'u2', 'u3', 'u4']);
      expect(p.appendedCount, 5);
    },
  );

  test(
    'stop() lets in-flight finish and drops unstarted queued work',
    () async {
      final appended = <String>[];
      var idx = 0;
      final started = <int>[];
      final p = VadLivePipeline(
        decode: (s) async {
          final id = idx++;
          started.add(id);
          await Future<void>.delayed(const Duration(milliseconds: 30));
          return 'u$id';
        },
        onAppend: appended.add,
      );
      // Enqueue 3 utterances quickly; consumer starts #0 immediately.
      for (var k = 0; k < 3; k++) {
        p.onSpeechStart();
        p.onFrame(_frame(0.5));
        p.onSpeechEnd();
      }
      // Let the first decode begin, then stop.
      await Future<void>.delayed(const Duration(milliseconds: 5));
      await p.stop();
      // Only the in-flight #0 completed and appended; #1/#2 were dropped.
      expect(appended, ['u0']);
      expect(started, [0]);
    },
  );

  test('stop() also drops the in-progress (un-ended) buffer', () async {
    final appended = <String>[];
    final p = VadLivePipeline(decode: (s) async => 'x', onAppend: appended.add);
    p.onSpeechStart();
    p.onFrame(_frame(0.6)); // speaking, never ended
    await p.stop();
    expect(appended, isEmpty);
  });

  test(
    'drainAll flushes an in-progress utterance that never got onSpeechEnd',
    () async {
      final appended = <String>[];
      final p = VadLivePipeline(
        decode: (s) async => 'tail',
        onAppend: appended.add,
      );
      p.onSpeechStart();
      p.onFrame(_frame(0.6)); // sim feed hits EOF here, no onSpeechEnd
      await p.drainAll();
      expect(appended, ['tail']);
    },
  );
}
