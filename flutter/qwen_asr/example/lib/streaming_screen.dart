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

/// Engine internal streaming chunk in seconds — the accuracy/latency knob.
/// The engine only decodes once a full chunk has accumulated, so no partial
/// can appear before the first chunk boundary.
///
/// 8.0 s is the ONLY value that reproduces the offline transcript exactly for
/// the bench clip (see streaming_test.dart tuning sweep); the simulated-mic
/// verification path uses it. It is the wrong trade-off for a live mic:
/// speech shorter than 8 s would show nothing until Stop/finalize.
const double kSimEngineChunkSec = 8.0;

/// Live-mic engine chunk: partials appear every ~2 s of speech. Costs minor
/// punctuation/duplication drift in intermediate partials vs the 8 s setting;
/// the finalize (Stop) pass still re-decodes the tail for the final text.
const double kLiveEngineChunkSec = 2.0;

/// Optional override for the simulated-mic engine chunk, in seconds, injected
/// via `--dart-define=SIM_ENGINE_CHUNK_SEC=2.0`. Lets automation exercise the
/// live-mic (short-chunk) latency profile through the sim path when no real
/// microphone input is available. Empty string = use [kSimEngineChunkSec].
const String kSimEngineChunkOverride = String.fromEnvironment(
  'SIM_ENGINE_CHUNK_SEC',
);

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
  String _provisional = '';
  String _status = 'Idle';
  String _perf = '';
  bool _running = false;
  bool _simulating = false;

  // Push pipeline state.
  final List<double> _pending = [];
  bool _draining = false;
  int _samplesPushed = 0;
  int? _firstPushMs;
  // First moment ANY text (stable OR provisional) is visible.
  int? _firstPartialMs;
  // First moment committed STABLE text is visible.
  int? _firstStableMs;

  int get _chunkSamples => (kMicChunkSec * kSampleRate).round();

  @override
  void initState() {
    super.initState();
    // Session-independent streaming config; the engine chunk size is set
    // per-session (live vs simulated) in _startMic/_startSimulated.
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
    _firstStableMs = null;
    _transcript = '';
    _provisional = '';
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
    final res = await widget.engine.streamPush(chunk, finalize: finalize);
    if (!mounted) return;
    final now = DateTime.now().millisecondsSinceEpoch;
    // First VISIBLE text = stable OR provisional becomes non-empty. This is the
    // latency the provisional-tail change is meant to slash.
    if ((res.text.isNotEmpty || res.provisional.isNotEmpty) &&
        _firstPartialMs == null) {
      _firstPartialMs = now;
    }
    // First COMMITTED stable text (the old firstPartial semantics).
    if (res.text.isNotEmpty && _firstStableMs == null) {
      _firstStableMs = now;
    }
    setState(() {
      if (res.text.isNotEmpty) _transcript = res.text;
      _provisional = res.provisional;
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
    final firstStableLatency = (_firstPushMs != null && _firstStableMs != null)
        ? (_firstStableMs! - _firstPushMs!)
        : null;
    return 'audio=${audioSec.toStringAsFixed(1)}s '
        'wall=${wallSec.toStringAsFixed(1)}s '
        'rtf=${rtf.toStringAsFixed(2)}x'
        '${firstLatency != null ? ' firstPartial=${firstLatency}ms' : ''}'
        '${firstStableLatency != null ? ' firstStable=${firstStableLatency}ms' : ''}\n'
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
    widget.engine.setStreamChunkSec(kLiveEngineChunkSec);
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
    final perf = _computePerf();
    setState(() {
      _running = false;
      _status = 'Done';
      _perf = perf;
    });
    // Test/automation hook: emit final result to logcat (grep QASR_).
    debugPrint('QASR_METRIC mic_perf | $perf');
    debugPrint('QASR_TRANSCRIPT mic | $_transcript');
  }

  // ---------------------------------------------------------------------------
  // Simulated microphone (WAV file through the SAME push pipeline)
  // ---------------------------------------------------------------------------

  Future<void> _startSimulated(String wavPath) async {
    if (_running || _simulating) return;
    // Automation hook: an http(s) AUTO_SIM_WAV is fetched into app documents
    // first — release builds have no adb access to the app sandbox, so
    // on-device test tooling provisions the clip over the network instead.
    if (wavPath.startsWith('http://') || wavPath.startsWith('https://')) {
      setState(() => _status = 'Fetching sim WAV...');
      try {
        final docs = await getApplicationDocumentsDirectory();
        final dest = File('${docs.path}/sim.wav');
        final client = HttpClient();
        final req = await client.getUrl(Uri.parse(wavPath));
        final resp = await req.close();
        if (resp.statusCode != 200) {
          throw HttpException('status ${resp.statusCode}');
        }
        await resp.pipe(dest.openWrite());
        client.close();
        wavPath = dest.path;
      } catch (e) {
        setState(() => _status = 'Sim WAV fetch failed: $e');
        return;
      }
    }
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

    // Automation may force the short-chunk (live-mic) profile through the sim
    // path via --dart-define=SIM_ENGINE_CHUNK_SEC so first-provisional latency
    // can be measured without a real microphone.
    final simEngineChunk =
        double.tryParse(kSimEngineChunkOverride) ?? kSimEngineChunkSec;
    widget.engine.setStreamChunkSec(simEngineChunk);
    await widget.engine.streamReset();
    setState(() {
      _simulating = true;
      _status =
          'Simulating mic from WAV (engineChunk='
          '${simEngineChunk.toStringAsFixed(1)}s)...';
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
    final perf = _computePerf();
    setState(() {
      _simulating = false;
      _status = 'Done (simulated)';
      _perf = perf;
    });
    // Test/automation hook: emit final result to logcat (grep QASR_).
    debugPrint('QASR_METRIC sim_perf | $perf');
    debugPrint('QASR_TRANSCRIPT sim | $_transcript');
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
              child: SingleChildScrollView(child: _buildTranscript(context)),
            ),
          ),
        ],
      ),
    );
  }

  /// Live transcript: committed stable text in normal weight, followed by the
  /// provisional (unfixed) tail in grey italic. The provisional tail is a
  /// lower-confidence hypothesis that later pushes may revise or promote to
  /// stable, so it is styled distinctly. Keeps the `transcript_text` key on the
  /// SelectableText for automation.
  Widget _buildTranscript(BuildContext context) {
    final hasAny = _transcript.isNotEmpty || _provisional.isNotEmpty;
    final baseStyle = const TextStyle(fontSize: 16);
    if (!hasAny) {
      return SelectableText(
        '(listening...)',
        key: const Key('transcript_text'),
        style: baseStyle,
      );
    }
    final provisionalColor =
        Theme.of(context).textTheme.bodySmall?.color?.withValues(alpha: 0.55) ??
        Colors.grey;
    final needsSpace =
        _transcript.isNotEmpty &&
        _provisional.isNotEmpty &&
        !_transcript.endsWith(' ') &&
        !_provisional.startsWith(' ');
    return SelectableText.rich(
      TextSpan(
        style: baseStyle,
        children: [
          if (_transcript.isNotEmpty) TextSpan(text: _transcript),
          if (_provisional.isNotEmpty)
            TextSpan(
              text: (needsSpace ? ' ' : '') + _provisional,
              style: TextStyle(
                color: provisionalColor,
                fontStyle: FontStyle.italic,
              ),
            ),
        ],
      ),
      key: const Key('transcript_text'),
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
