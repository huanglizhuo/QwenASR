import 'package:flutter/foundation.dart';
import 'package:qwen_asr/qwen_asr.dart';

/// Sample rate the engine expects (16 kHz mono). Shared by every capture path.
const int kSampleRate = 16000;

/// Minimum captured audio (seconds) before a buffer is worth a one-shot decode.
///
/// Shorter buffers are treated as accidental taps / VAD blips and skipped. This
/// is the single source of truth for the "too-short" guard, reused by BOTH the
/// push-to-talk Record tab and the VAD-Live utterance segmenter's min-speech
/// gate so the two stay consistent.
const double kMinUtteranceSec = 0.3;

/// Outcome of a one-shot offline decode attempt.
class OneShotOutcome {
  /// True when the buffer was below [kMinUtteranceSec] and no decode ran.
  final bool tooShort;

  /// Duration of the input buffer in seconds.
  final double durationSec;

  /// The transcript ('' when [tooShort] or the decode returned nothing).
  final String transcript;

  /// Engine perf stats string ('' when [tooShort]).
  final String perf;

  const OneShotOutcome({
    required this.tooShort,
    required this.durationSec,
    required this.transcript,
    required this.perf,
  });

  const OneShotOutcome.tooShort(this.durationSec)
    : tooShort = true,
      transcript = '',
      perf = '';
}

/// Runs the shared one-shot offline decode: the too-short guard, the
/// [QAsrEngine.transcribePcm] call (the project's most WER-validated decode
/// path — the same one the Record tab uses), and the `QASR_METRIC` /
/// `QASR_TRANSCRIPT` automation log lines.
///
/// Behavior is identical to what the Record tab did inline before this was
/// extracted; the VAD-Live consumer reuses it verbatim so both paths share one
/// guard threshold and one logging format. [metricTag] labels the log lines
/// (e.g. `ptt`, `vad`).
class OneShotTranscriber {
  final QAsrEngine engine;
  final String metricTag;
  final double minSec;

  const OneShotTranscriber(
    this.engine, {
    required this.metricTag,
    this.minSec = kMinUtteranceSec,
  });

  /// Decode [samples] (16 kHz mono float, -1..1). Applies the too-short guard,
  /// then transcribes and logs. Never throws for the guard case; a decode
  /// failure propagates to the caller.
  Future<OneShotOutcome> transcribe(Float32List samples) async {
    final sec = samples.length / kSampleRate;
    if (sec < minSec) {
      return OneShotOutcome.tooShort(sec);
    }
    final text = await engine.transcribePcm(samples);
    final perf = engine.perfStats();
    // Test/automation hook: emit final result to logcat (grep QASR_).
    debugPrint('QASR_METRIC ${metricTag}_perf | $perf');
    debugPrint('QASR_TRANSCRIPT $metricTag | $text');
    return OneShotOutcome(
      tooShort: false,
      durationSec: sec,
      transcript: text,
      perf: perf,
    );
  }
}
