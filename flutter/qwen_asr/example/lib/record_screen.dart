import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:qwen_asr/qwen_asr.dart';
import 'package:record/record.dart';

import 'one_shot_transcribe.dart';

/// Push-to-talk ("PTT") recording screen.
///
/// Hold the button to record: PCM16 mic samples are accumulated in memory
/// (NOT pushed to the streaming engine). Release to run the accumulated audio
/// through the one-shot offline [QAsrEngine.transcribePcm] API and show the
/// result. The state machine is idle → recording → transcribing → idle.
class RecordScreen extends StatefulWidget {
  final QAsrEngine engine;
  const RecordScreen({super.key, required this.engine});

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _micSub;
  Timer? _ticker;

  late final OneShotTranscriber _oneShot = OneShotTranscriber(
    widget.engine,
    metricTag: 'ptt',
  );

  final List<double> _samples = [];
  bool _recording = false;
  bool _transcribing = false;
  int _recordMs = 0;

  String _status = 'Hold the button and speak';
  String _transcript = '';
  String _perf = '';

  final Stopwatch _holdWatch = Stopwatch();

  @override
  void dispose() {
    _ticker?.cancel();
    _micSub?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _startRecording() async {
    // Idempotent: raw pointer events (or overlapping gestures) may fire twice.
    if (_recording || _transcribing) return;
    if (!await _recorder.hasPermission()) {
      if (!mounted) return;
      setState(() => _status = 'Microphone permission denied');
      return;
    }
    _samples.clear();
    _holdWatch
      ..reset()
      ..start();
    setState(() {
      _recording = true;
      _recordMs = 0;
      _transcript = '';
      _perf = '';
      _status = 'Recording... (hold)';
    });
    _ticker = Timer.periodic(const Duration(milliseconds: 100), (_) {
      if (!mounted || !_recording) return;
      setState(() => _recordMs = _holdWatch.elapsedMilliseconds);
    });
    try {
      final stream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: kSampleRate,
          numChannels: 1,
        ),
      );
      _micSub = stream.listen(
        (bytes) => _samples.addAll(_pcm16ToFloat(bytes)),
        onError: (Object e) {
          if (mounted) setState(() => _status = 'Mic error: $e');
        },
      );
    } catch (e) {
      _ticker?.cancel();
      _holdWatch.stop();
      if (!mounted) return;
      setState(() {
        _recording = false;
        _status = 'Failed to start mic: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    if (!_recording) return;
    _recording = false;
    _ticker?.cancel();
    _holdWatch.stop();
    await _micSub?.cancel();
    _micSub = null;
    try {
      await _recorder.stop();
    } catch (_) {
      // Best-effort; ignore stop errors.
    }

    final captured = Float32List.fromList(_samples);
    _samples.clear();
    final capturedSec = captured.length / kSampleRate;
    if (capturedSec < kMinUtteranceSec) {
      if (!mounted) return;
      setState(() {
        _status =
            'Too short (${(capturedSec * 1000).round()} ms) — hold longer and speak';
      });
      return;
    }
    await _transcribeSamples(captured);
  }

  /// Run accumulated samples through the one-shot offline API and show the
  /// result. Exposed for integration testing (synthetic-buffer injection).
  @visibleForTesting
  Future<void> transcribeSamples(Float32List samples) =>
      _transcribeSamples(samples);

  Future<void> _transcribeSamples(Float32List samples) async {
    if (!mounted) return;
    setState(() {
      _transcribing = true;
      _status = 'Transcribing...';
    });
    try {
      // Shared one-shot decode: too-short guard + transcribePcm + QASR_ logs.
      final outcome = await _oneShot.transcribe(samples);
      if (!mounted) return;
      if (outcome.tooShort) {
        setState(() {
          _transcribing = false;
          _status =
              'Too short (${(outcome.durationSec * 1000).round()} ms) — hold longer and speak';
        });
        return;
      }
      setState(() {
        _transcript = outcome.transcript;
        _perf = outcome.perf;
        _transcribing = false;
        _status = 'Done (${outcome.durationSec.toStringAsFixed(1)}s)';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _transcribing = false;
        _status = 'Failed: $e';
      });
    }
  }

  Float32List _pcm16ToFloat(Uint8List bytes) {
    final count = bytes.lengthInBytes ~/ 2;
    final view = ByteData.view(bytes.buffer, bytes.offsetInBytes, count * 2);
    final out = Float32List(count);
    for (var i = 0; i < count; i++) {
      out[i] = view.getInt16(i * 2, Endian.little) / 32768.0;
    }
    return out;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final active = _recording;
    // While transcribing, the button is disabled to prevent overlapping sessions.
    final enabled = !_transcribing;
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: Center(
              child: Listener(
                onPointerDown: enabled ? (_) => _startRecording() : null,
                onPointerUp: enabled ? (_) => _stopRecording() : null,
                onPointerCancel: enabled ? (_) => _stopRecording() : null,
                child: Container(
                  key: const Key('ptt_button'),
                  width: 200,
                  height: 200,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: !enabled
                        ? theme.disabledColor.withValues(alpha: 0.2)
                        : (active
                              ? theme.colorScheme.error
                              : theme.colorScheme.primary),
                    boxShadow: active
                        ? [
                            BoxShadow(
                              color: theme.colorScheme.error.withValues(
                                alpha: 0.4,
                              ),
                              blurRadius: 24,
                              spreadRadius: 4,
                            ),
                          ]
                        : null,
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        _transcribing
                            ? Icons.hourglass_top
                            : (active ? Icons.mic : Icons.mic_none),
                        size: 56,
                        color: theme.colorScheme.onPrimary,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _transcribing
                            ? 'Transcribing'
                            : (active ? 'Recording' : 'Hold to talk'),
                        style: TextStyle(color: theme.colorScheme.onPrimary),
                      ),
                      if (active)
                        Text(
                          '${(_recordMs / 1000).toStringAsFixed(1)}s',
                          key: const Key('ptt_duration'),
                          style: TextStyle(
                            color: theme.colorScheme.onPrimary,
                            fontFeatures: const [FontFeature.tabularFigures()],
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text('Status: $_status', key: const Key('ptt_status')),
          if (_perf.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text('Perf: $_perf', style: theme.textTheme.bodySmall),
            ),
          const SizedBox(height: 12),
          const Text(
            'Transcript:',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          SizedBox(
            height: 120,
            child: SingleChildScrollView(
              child: SelectableText(
                _transcript.isEmpty ? '(none)' : _transcript,
                key: const Key('ptt_transcript'),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
