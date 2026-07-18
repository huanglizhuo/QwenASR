import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:qwen_asr/qwen_asr.dart';
import 'package:record/record.dart';

import 'wav_utils.dart';

/// Sample rate the engine expects (16 kHz mono).
const int kSampleRate = 16000;

/// Dart-side mic push chunk in seconds. Audio is buffered to this size before
/// each `streamPush`. Tunable: smaller = more frequent pushes / lower latency,
/// larger = fewer FFI round-trips. 0.5 s is a good default (see report).
const double kMicChunkSec = 0.5;

/// Engine internal streaming chunk in seconds. This is the accuracy/latency
/// knob. 8.0 s is the engine default and the ONLY value that reproduces the
/// offline transcript exactly for the bench clip; smaller values emit partials
/// sooner but introduce punctuation/duplication drift (see streaming_test.dart
/// tuning sweep). Kept at 8.0 for correctness; the finalize (Stop) pass still
/// produces the full accurate transcript.
const double kEngineChunkSec = 8.0;

/// Leading chunks kept "unfixed" before committing tokens to the stable
/// transcript. Small value = progressive live partials. The engine default
/// (99) holds everything until finalize, which truncates a fast live stream.
const int kUnfixedChunks = 2;

/// Live microphone + simulated-mic streaming ASR screen.
class StreamingScreen extends StatefulWidget {
  final QAsrEngine engine;

  /// Optional WAV path to auto-run through the simulated-mic path on launch
  /// (test hook, injected via --dart-define). null disables auto-run.
  final String? autoSimWavPath;

  const StreamingScreen({super.key, required this.engine, this.autoSimWavPath});

  @override
  State<StreamingScreen> createState() => _StreamingScreenState();
}

class _StreamingScreenState extends State<StreamingScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _micSub;

  String _transcript = '';
  String _status = 'Idle';
  String _perf = '';
  bool _running = false;
  bool _simulating = false;

  // Push pipeline state.
  final List<double> _pending = [];
  bool _draining = false;
  int _samplesPushed = 0;
  int? _firstPushMs;
  int? _firstPartialMs;

  int get _chunkSamples => (kMicChunkSec * kSampleRate).round();

  @override
  void initState() {
    super.initState();
    // Configure the engine streaming session for correct progressive partials.
    widget.engine.setStreamChunkSec(kEngineChunkSec);
    widget.engine.setStreamUnfixedChunks(kUnfixedChunks);
    widget.engine.setPastTextConditioning(true);
    if (widget.autoSimWavPath != null) {
      WidgetsBinding.instance.addPostFrameCallback(
        (_) => _startSimulated(widget.autoSimWavPath!),
      );
    }
  }

  @override
  void dispose() {
    _micSub?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Shared push pipeline
  // ---------------------------------------------------------------------------

  void _resetSession() {
    _pending.clear();
    _samplesPushed = 0;
    _firstPushMs = null;
    _firstPartialMs = null;
    _transcript = '';
    _perf = '';
  }

  /// Feed float samples (-1..1) into the buffer and drain full chunks.
  void _feed(Float32List samples) {
    _pending.addAll(samples);
    _drain();
  }

  /// Serialized drain: pushes one chunk at a time, awaiting each so engine
  /// calls never overlap and partial results arrive in order.
  Future<void> _drain({bool finalize = false}) async {
    if (_draining) return;
    _draining = true;
    try {
      while (_pending.length >= _chunkSamples) {
        final chunk = Float32List.fromList(_pending.sublist(0, _chunkSamples));
        _pending.removeRange(0, _chunkSamples);
        await _pushChunk(chunk, finalize: false);
      }
      if (finalize) {
        final tail = Float32List.fromList(_pending);
        _pending.clear();
        await _pushChunk(tail, finalize: true);
      }
    } finally {
      _draining = false;
    }
  }

  Future<void> _pushChunk(Float32List chunk, {required bool finalize}) async {
    _firstPushMs ??= DateTime.now().millisecondsSinceEpoch;
    _samplesPushed += chunk.length;
    final text = await widget.engine.streamPush(chunk, finalize: finalize);
    if (!mounted) return;
    if (text.isNotEmpty && _firstPartialMs == null) {
      _firstPartialMs = DateTime.now().millisecondsSinceEpoch;
    }
    setState(() {
      if (text.isNotEmpty) _transcript = text;
    });
  }

  String _computePerf() {
    final now = DateTime.now().millisecondsSinceEpoch;
    final audioSec = _samplesPushed / kSampleRate;
    final wallSec = _firstPushMs == null ? 0.0 : (now - _firstPushMs!) / 1000.0;
    final rtf = wallSec > 0 ? audioSec / wallSec : 0.0;
    final firstLatency = (_firstPushMs != null && _firstPartialMs != null)
        ? (_firstPartialMs! - _firstPushMs!)
        : null;
    return 'audio=${audioSec.toStringAsFixed(1)}s '
        'wall=${wallSec.toStringAsFixed(1)}s '
        'rtf=${rtf.toStringAsFixed(2)}x'
        '${firstLatency != null ? ' firstPartial=${firstLatency}ms' : ''}\n'
        '${widget.engine.perfStats()}';
  }

  // ---------------------------------------------------------------------------
  // Live microphone
  // ---------------------------------------------------------------------------

  Future<void> _startMic() async {
    if (_running || _simulating) return;
    if (!await _recorder.hasPermission()) {
      setState(() => _status = 'Microphone permission denied');
      return;
    }
    await widget.engine.streamReset();
    setState(() {
      _running = true;
      _status = 'Listening...';
      _resetSession();
    });

    final stream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: kSampleRate,
        numChannels: 1,
      ),
    );
    _micSub = stream.listen(
      (bytes) => _feed(_pcm16ToFloat(bytes)),
      onError: (Object e) => setState(() => _status = 'Mic error: $e'),
    );
  }

  Future<void> _stopMic() async {
    if (!_running) return;
    setState(() => _status = 'Finalizing...');
    await _micSub?.cancel();
    _micSub = null;
    await _recorder.stop();
    await _drain(finalize: true);
    if (!mounted) return;
    setState(() {
      _running = false;
      _status = 'Done';
      _perf = _computePerf();
    });
  }

  // ---------------------------------------------------------------------------
  // Simulated microphone (WAV file through the SAME push pipeline)
  // ---------------------------------------------------------------------------

  Future<void> _startSimulated(String wavPath) async {
    if (_running || _simulating) return;
    final file = File(wavPath);
    if (!await file.exists()) {
      setState(() => _status = 'Sim WAV not found: $wavPath');
      return;
    }
    final samples = _parseWavPcm16(await file.readAsBytes());
    if (samples.isEmpty) {
      setState(() => _status = 'Sim WAV parse failed');
      return;
    }

    await widget.engine.streamReset();
    setState(() {
      _simulating = true;
      _status = 'Simulating mic from WAV...';
      _resetSession();
    });

    // Feed at real-time pacing: sleep chunk-duration between pushes.
    final chunk = _chunkSamples;
    for (var i = 0; i < samples.length && mounted; i += chunk) {
      final end = (i + chunk).clamp(0, samples.length);
      _feed(Float32List.fromList(samples.sublist(i, end)));
      await Future<void>.delayed(
        const Duration(milliseconds: (kMicChunkSec * 1000) ~/ 1),
      );
    }
    await _drain(finalize: true);
    if (!mounted) return;
    setState(() {
      _simulating = false;
      _status = 'Done (simulated)';
      _perf = _computePerf();
    });
  }

  // ---------------------------------------------------------------------------
  // Audio helpers
  // ---------------------------------------------------------------------------

  Float32List _pcm16ToFloat(Uint8List bytes) {
    final count = bytes.lengthInBytes ~/ 2;
    final view = ByteData.view(bytes.buffer, bytes.offsetInBytes, count * 2);
    final out = Float32List(count);
    for (var i = 0; i < count; i++) {
      out[i] = view.getInt16(i * 2, Endian.little) / 32768.0;
    }
    return out;
  }

  /// Parse a 16-bit PCM WAV to 16 kHz mono float samples (-1..1), downmixing
  /// channels and linearly resampling to the engine's expected rate. Mirrors
  /// what the Rust `audio::parse_wav_buffer` does so the simulated-mic path
  /// works with arbitrary-rate WAVs.
  Float32List _parseWavPcm16(Uint8List bytes) => parseWavTo16kMono(bytes);

  // ---------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final busy = _running || _simulating;
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  key: const Key('mic_button'),
                  onPressed: _simulating
                      ? null
                      : (_running ? _stopMic : _startMic),
                  icon: Icon(_running ? Icons.stop : Icons.mic),
                  label: Text(_running ? 'Stop' : 'Start Mic'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  key: const Key('sim_button'),
                  onPressed: busy
                      ? null
                      : () async {
                          final path = await _defaultSimWavPath();
                          if (path != null) await _startSimulated(path);
                        },
                  icon: const Icon(Icons.play_circle_outline),
                  label: const Text('Simulated Mic'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text('Status: $_status'),
          if (_perf.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                'Perf: $_perf',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ),
          const SizedBox(height: 12),
          const Text(
            'Live transcript:',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Expanded(
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                border: Border.all(color: Theme.of(context).dividerColor),
                borderRadius: BorderRadius.circular(8),
              ),
              child: SingleChildScrollView(
                child: SelectableText(
                  _transcript.isEmpty ? '(listening...)' : _transcript,
                  key: const Key('transcript_text'),
                  style: const TextStyle(fontSize: 16),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// A simulated-mic WAV provisioned into app documents during testing.
  /// Test tooling drops `sim.wav` in the app documents directory.
  Future<String?> _defaultSimWavPath() async {
    try {
      final docs = await getApplicationDocumentsDirectory();
      final candidate = '${docs.path}/sim.wav';
      if (await File(candidate).exists()) return candidate;
      setState(
        () => _status =
            'No sim.wav in app documents. Provision one to use this button.',
      );
    } catch (e) {
      if (kDebugMode) debugPrint('sim path error: $e');
    }
    return null;
  }
}
