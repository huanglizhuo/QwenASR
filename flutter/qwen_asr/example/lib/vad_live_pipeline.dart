import 'dart:async';
import 'dart:collection';
import 'dart:typed_data';

/// Decodes one completed utterance's samples into a transcript string.
///
/// Injected so the pipeline is unit-testable without the Rust engine. The
/// production implementation wraps `OneShotTranscriber` (too-short guard +
/// `transcribePcm` + `QASR_` logging). Returns '' for a too-short / empty
/// utterance, which the pipeline simply does not append.
typedef UtteranceDecoder = Future<String> Function(Float32List samples);

/// The queue + single sequential one-shot consumer for VAD-Live mode.
///
/// This layer is deliberately independent of the specific VAD detector: it is
/// fed discrete VAD events (`onSpeechStart` / `onFrame` / `onSpeechEnd`) and
/// owns everything after boundary detection — utterance buffering (so it can
/// enforce a deterministic safety cap regardless of the detector's own duration
/// handling), a FIFO queue that never blocks or drops during capture, and a
/// single consumer that decodes utterances one at a time (never overlapping
/// engine calls) while capture continues.
///
/// Utterance samples are assembled from the detector's per-frame output rather
/// than its convenience "speech-end" payload, specifically so the max-duration
/// safety cap ([maxUtteranceSec]) can force-close an utterance the detector
/// would otherwise let grow unbounded.
class VadLivePipeline {
  /// Injected decode function (production: [OneShotTranscriber]).
  final UtteranceDecoder decode;

  /// Called with each completed utterance's transcript, in completion order.
  final void Function(String text) onAppend;

  /// Optional status updates ('Listening...', 'Transcribing utterance N...').
  final void Function(String status)? onStatus;

  final int sampleRate;

  /// Defensive max-utterance-duration cap (belt-and-suspenders on top of the
  /// detector's own silence handling; also makes tests deterministic). An
  /// utterance reaching this many seconds without a detected boundary is
  /// force-closed and emitted; the next audio starts a fresh utterance.
  final double maxUtteranceSec;

  VadLivePipeline({
    required this.decode,
    required this.onAppend,
    this.onStatus,
    this.sampleRate = 16000,
    this.maxUtteranceSec = 20.0,
  });

  int get _maxSamples => (sampleRate * maxUtteranceSec).round();

  // Utterance assembly (owned here so the cap can bound it).
  final List<double> _buf = <double>[];
  bool _inUtterance = false;

  // FIFO queue + single sequential consumer.
  final Queue<Float32List> _queue = Queue<Float32List>();
  bool _consumerBusy = false;
  bool _accepting = true;
  int _appended = 0;

  /// Utterances decoded and appended (non-empty) so far this session.
  int get appendedCount => _appended;

  /// Utterances waiting in the FIFO (not yet started).
  int get queueLength => _queue.length;

  /// True while the consumer is inside a decode.
  bool get isBusy => _consumerBusy;

  /// The detector reports an utterance has begun. Start a fresh buffer.
  void onSpeechStart() {
    _inUtterance = true;
    _buf.clear();
    onStatus?.call('Listening...');
  }

  /// The detector delivered one processed frame's float samples (16 kHz mono,
  /// -1..1). Accumulate while inside an utterance and enforce the safety cap.
  void onFrame(List<double> frame) {
    if (!_inUtterance) return;
    _buf.addAll(frame);
    if (_buf.length >= _maxSamples) {
      // Force-close this over-long utterance and keep listening: a new buffer
      // accumulates the continuation immediately.
      _emitCurrent();
    }
  }

  /// The detector reports the utterance ended at a silence boundary.
  void onSpeechEnd() {
    if (!_inUtterance) return;
    _emitCurrent();
    _inUtterance = false;
  }

  /// The detector reports a false positive: drop the in-progress buffer.
  void onMisfire() {
    _buf.clear();
    _inUtterance = false;
  }

  void _emitCurrent() {
    if (_buf.isEmpty) return;
    final u = Float32List.fromList(_buf);
    _buf.clear();
    if (!_accepting) return;
    _queue.add(u);
    _pump();
  }

  Future<void> _pump() async {
    if (_consumerBusy) return;
    _consumerBusy = true;
    try {
      while (_queue.isNotEmpty) {
        final u = _queue.removeFirst();
        onStatus?.call('Transcribing utterance ${_appended + 1}...');
        final text = (await decode(u)).trim();
        if (text.isNotEmpty) {
          _appended++;
          onAppend(text);
        }
      }
    } finally {
      _consumerBusy = false;
    }
  }

  /// Finalize by decoding EVERYTHING captured so far: flush the in-progress
  /// utterance (so trailing audio with no closing silence — e.g. a sim feed
  /// hitting EOF — is not lost) and wait for the queue to fully drain. Used by
  /// the simulated-mic path and any "finish and keep all results" flow.
  Future<void> drainAll() async {
    if (_inUtterance && _buf.isNotEmpty) {
      _emitCurrent();
    }
    _inUtterance = false;
    _pump();
    while (_consumerBusy || _queue.isNotEmpty) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
  }

  /// Stop capture semantics: let any IN-FLIGHT decode finish and append its
  /// result, but DROP the in-progress buffer and any not-yet-started queued
  /// utterances. Used by the real-mic Stop button.
  Future<void> stop() async {
    _accepting = false;
    _inUtterance = false;
    _buf.clear();
    _queue.clear(); // drop unstarted queued work
    while (_consumerBusy) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
  }
}
