import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:qwen_asr/qwen_asr.dart';
import 'package:record/record.dart';
import 'package:vad/vad.dart';

import 'one_shot_transcribe.dart';
import 'streaming_settings.dart';
import 'vad_live_pipeline.dart';
import 'wav_utils.dart';

/// Local Flutter-asset base path for the Silero VAD model. The `vad` package
/// concatenates this with the model filename and, because it is NOT an http(s)
/// URL, loads it via `rootBundle` — so VAD detection runs fully offline (no
/// CDN download). See pubspec `assets:` and `assets/vad/`.
const String kVadAssetBasePath = 'assets/vad/';

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

/// Live-mic engine chunk default: partials appear every ~2 s of speech. The
/// live-mic value now comes from the user-editable streaming settings (see
/// `streaming_settings.dart`), whose default is 2.0 s. A 3 s chunk was trialled
/// on-device (R13-Android stage 1 RTF sweep) but REJECTED: no wall-RTF gain
/// (the sim feed is real-time-paced, so RTF is pinned near 1.0×) while
/// regressing first-partial/first-stable latency AND duplicating a full
/// sentence in the final transcript — so >= 3 s is never offered in the UI.
/// 3 s stays reachable via `--dart-define=SIM_ENGINE_CHUNK_SEC=3.0` for
/// comparison only.

/// Optional override for the simulated-mic engine chunk, in seconds, injected
/// via `--dart-define=SIM_ENGINE_CHUNK_SEC=2.0`. Lets automation exercise the
/// live-mic (short-chunk) latency profile through the sim path when no real
/// microphone input is available. Empty string = use [kSimEngineChunkSec].
const String kSimEngineChunkOverride = String.fromEnvironment(
  'SIM_ENGINE_CHUNK_SEC',
);

/// Optional override for the language mode on the simulated-mic path, injected
/// via `--dart-define=SIM_LANGUAGE=auto|zh|en|ja|ko|yue`. Lets automation
/// (R13-Android Task 5.2 multilingual smoke) drive the multilingual/forced-
/// language mode through the sim path without tapping the settings UI. Empty
/// string = use the persisted settings selection.
const String kSimLanguageOverride = String.fromEnvironment('SIM_LANGUAGE');

/// Optional override for the live mode on the simulated-mic path, injected via
/// `--dart-define=SIM_LIVE_MODE=full|vad`. Lets automation drive the client-side
/// VAD-Live pipeline (Dart segmentation + one-shot offline decode) through the
/// sim path without tapping the settings UI. Empty string = use the persisted
/// settings selection.
const String kSimLiveModeOverride = String.fromEnvironment('SIM_LIVE_MODE');

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

  // ---- VAD Live (client-side, Silero-VAD-driven) state ---------------------
  // Shared one-shot decode helper (too-short guard + transcribePcm + QASR_ logs).
  late final OneShotTranscriber _oneShot = OneShotTranscriber(
    widget.engine,
    metricTag: 'vad',
  );

  /// A VAD-Live mic session is active (distinct from Full-Streaming `_running`).
  bool _vadActive = false;

  /// Mic is currently capturing (false once Stop is pressed but while the queue
  /// still drains).
  bool _capturing = false;

  /// The Silero VAD detector for the active session (owns/consumes audio).
  VadHandler? _vadHandler;

  /// Subscriptions to the detector's event streams (cancelled at session end).
  final List<StreamSubscription<dynamic>> _vadSubs = [];

  /// The queue + sequential one-shot consumer + safety cap.
  VadLivePipeline? _pipeline;

  /// Wall-clock start of the VAD-Live session (for perf reporting).
  int? _vadStartMs;

  /// Total utterance audio (samples) decoded through VAD-Live this session.
  int _vadAudioSamples = 0;

  /// Appended-utterance count captured at the last VAD-Live finalize (for tests
  /// and the perf line, since the pipeline is torn down at session end).
  int _lastVadAppended = 0;

  /// Editable streaming settings (language mode + engine chunk). Loaded from
  /// disk on launch; changes apply at the next Start. Null until loaded.
  StreamingSettings? _settings;

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
    // Load persisted settings, then (for automation) optionally auto-run the sim.
    StreamingSettings.load().then((s) {
      if (!mounted) return;
      setState(() => _settings = s);
      if (widget.autoSimWavPath != null) {
        WidgetsBinding.instance.addPostFrameCallback(
          (_) => _startSimulated(widget.autoSimWavPath!),
        );
      }
    });
  }

  @override
  void dispose() {
    _micSub?.cancel();
    for (final s in _vadSubs) {
      s.cancel();
    }
    _vadHandler?.dispose();
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
    if (_draining) {
      // A non-finalizing drain can be dropped (another is already draining the
      // shared buffer). A finalize must NOT be dropped: when the engine runs
      // slower than real time, an in-flight drain is still consuming the buffer
      // as the last audio arrives, so wait for it to finish, then flush the
      // remaining tail. Otherwise the final transcript is silently truncated.
      if (!finalize) return;
      while (_draining) {
        await Future<void>.delayed(const Duration(milliseconds: 10));
      }
    }
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
    // Apply the current settings to the engine (session-level; before reset).
    final settings = _settings ?? StreamingSettings();
    final chunkSec = settings.engineChunkSec;
    final langLabel = settings.applyLanguage(widget.engine);
    final modeLabel = settings.applyLiveMode(widget.engine);
    widget.engine.setStreamChunkSec(chunkSec);
    await widget.engine.streamReset();
    // Test/automation hook: confirm the settings UI drives the engine.
    debugPrint(
      'QASR_METRIC session_start mode=mic language=$langLabel '
      'liveMode=$modeLabel engineChunk=${chunkSec.toStringAsFixed(1)}s',
    );
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
  // VAD Live (Silero VAD detector + one-shot offline decode)
  // ---------------------------------------------------------------------------
  //
  // The streaming engine is NOT involved here. The `vad` package (Silero VAD via
  // ONNX, running on the bundled offline model) detects utterance boundaries in
  // the mic/sim audio; each completed utterance is transcribed with an
  // independent one-shot offline decode (`transcribePcm`), the project's most
  // WER-validated decode path. This is immune to rolling-window streaming drift
  // by construction; the only cost is no live per-word partial within an
  // utterance (a status indicator is shown instead).

  /// Build the pipeline (queue + sequential one-shot consumer + safety cap)
  /// and the VadHandler, wiring the detector's events into the pipeline. Shared
  /// by the real-mic and simulated-mic paths.
  void _beginVadSession() {
    _vadAudioSamples = 0;
    _vadStartMs = DateTime.now().millisecondsSinceEpoch;
    _pipeline = VadLivePipeline(
      decode: (samples) async {
        final outcome = await _oneShot.transcribe(samples);
        _vadAudioSamples += samples.length;
        return outcome.tooShort ? '' : outcome.transcript;
      },
      onAppend: (text) {
        if (!mounted) return;
        setState(() {
          _transcript = _transcript.isEmpty ? text : '$_transcript $text';
        });
      },
      onStatus: (s) {
        if (!mounted) return;
        setState(() => _status = _capturing ? s : _status);
      },
    );
    final handler = VadHandler.create(isDebug: false);
    _vadSubs
      ..add(handler.onSpeechStart.listen((_) => _pipeline?.onSpeechStart()))
      ..add(handler.onSpeechEnd.listen((_) => _pipeline?.onSpeechEnd()))
      ..add(handler.onVADMisfire.listen((_) => _pipeline?.onMisfire()))
      ..add(handler.onFrameProcessed.listen((f) => _pipeline?.onFrame(f.frame)))
      ..add(
        handler.onError.listen((e) {
          if (mounted) setState(() => _status = 'VAD error: $e');
        }),
      );
    _vadHandler = handler;
    setState(() {
      _vadActive = true;
      _capturing = true;
      _resetSession();
      _status = 'Listening...';
    });
  }

  Future<void> _startVadLiveMic() async {
    if (_running || _simulating || _vadActive) return;
    if (!await _recorder.hasPermission()) {
      setState(() => _status = 'Microphone permission denied');
      return;
    }
    // Language applies once per session (sticky ctx field across the many
    // independent transcribePcm calls). VAD Live does NOT touch the streaming
    // engine or the vad_segment_reset flag.
    final settings = _settings ?? StreamingSettings();
    final langLabel = settings.applyLanguage(widget.engine);
    debugPrint(
      'QASR_METRIC session_start mode=mic liveMode=vad language=$langLabel',
    );
    _beginVadSession();
    // The APP owns the mic (record 7) and feeds the detector via `audioStream`
    // so the `vad` package never uses its own (v6-era) internal recorder. The
    // bundled offline model loads from the asset bundle via the local path.
    final micStream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: kSampleRate,
        numChannels: 1,
      ),
    );
    await _vadHandler?.startListening(
      baseAssetPath: kVadAssetBasePath,
      audioStream: micStream,
    );
  }

  Future<void> _stopMicRecorder() async {
    try {
      await _recorder.stop();
    } catch (_) {
      // Best-effort.
    }
  }

  Future<void> _stopVadLiveMic() async {
    if (!_vadActive) return;
    // Stop capturing new audio immediately.
    _capturing = false;
    if (mounted) setState(() => _status = 'Finalizing...');
    await _stopMicRecorder();
    await _vadHandler?.stopListening();
    // Stop semantics: let the in-flight decode finish; drop unstarted queue and
    // the in-progress buffer.
    await _pipeline?.stop();
    await _finishVadSession(modeTag: 'mic');
  }

  /// Tear down the detector + report perf. Assumes the pipeline has already been
  /// finalized (drained or stopped) by the caller.
  Future<void> _finishVadSession({required String modeTag}) async {
    for (final s in _vadSubs) {
      await s.cancel();
    }
    _vadSubs.clear();
    await _vadHandler?.dispose();
    _vadHandler = null;
    final appended = _pipeline?.appendedCount ?? 0;
    _lastVadAppended = appended;
    _pipeline = null;
    final perf = _computeVadPerf(appended);
    if (!mounted) return;
    setState(() {
      _vadActive = false;
      _simulating = false;
      _status = modeTag == 'sim' ? 'Done (simulated)' : 'Done';
      _perf = perf;
    });
    debugPrint('QASR_METRIC ${modeTag}_vad_perf | $perf');
    debugPrint('QASR_TRANSCRIPT ${modeTag}_vad | $_transcript');
  }

  String _computeVadPerf(int utterances) {
    final now = DateTime.now().millisecondsSinceEpoch;
    final audioSec = _vadAudioSamples / kSampleRate;
    final wallSec = _vadStartMs == null ? 0.0 : (now - _vadStartMs!) / 1000.0;
    final rtf = wallSec > 0 ? audioSec / wallSec : 0.0;
    return 'utterances=$utterances '
        'audio=${audioSec.toStringAsFixed(1)}s '
        'wall=${wallSec.toStringAsFixed(1)}s '
        'rtf=${rtf.toStringAsFixed(2)}x\n'
        '${widget.engine.perfStats()}';
  }

  /// Simulated-mic VAD-Live: feed a WAV through the SAME detector entrypoint the
  /// real mic uses, by handing the VadHandler a synthetic `audioStream` of
  /// PCM16 bytes (at real-time pacing by default), so the whole VAD-Live
  /// pipeline is exercised without a physical mic.
  ///
  /// [realtime] paces the feed at wall-clock speed (as a live mic would). Tests
  /// pass `false` to feed as fast as possible — Silero VAD keys off sample
  /// counts, not wall time, so the utterance boundaries are unchanged.
  Future<void> _runSimulatedVad(
    Float32List samples,
    String langLabel, {
    bool realtime = true,
  }) async {
    _beginVadSession();
    setState(() {
      _simulating = true;
      _status = 'Simulating VAD Live from WAV (language=$langLabel)...';
    });
    // Drive the detector from a synthetic PCM16 stream. Pace at ~0.5 s chunks.
    final controller = StreamController<Uint8List>();
    final feed = () async {
      final chunk = _chunkSamples;
      for (var i = 0; i < samples.length && mounted && _vadActive; i += chunk) {
        final end = (i + chunk).clamp(0, samples.length);
        controller.add(_floatToPcm16(samples.sublist(i, end)));
        if (realtime) {
          await Future<void>.delayed(
            const Duration(milliseconds: (kMicChunkSec * 1000) ~/ 1),
          );
        }
      }
      await controller.close();
    }();
    await _vadHandler?.startListening(
      baseAssetPath: kVadAssetBasePath,
      audioStream: controller.stream,
    );
    await feed;
    // Give the detector a beat to emit any final boundary, then finalize: flush
    // the in-progress utterance and drain the whole queue (keep all results).
    await Future<void>.delayed(const Duration(milliseconds: 200));
    _capturing = false;
    await _vadHandler?.stopListening();
    await _pipeline?.drainAll();
    if (mounted) setState(() => _status = 'Finalizing...');
    await _finishVadSession(modeTag: 'sim');
  }

  /// Test hook: run the full VAD-Live pipeline (segmentation + queue + one-shot
  /// decode + UI append) on an in-memory sample buffer, applying the current
  /// language setting exactly as a session start would. Feeds without real-time
  /// pacing so integration tests stay fast.
  @visibleForTesting
  Future<void> runVadLiveOnSamplesForTest(Float32List samples) async {
    final settings = _settings ?? StreamingSettings();
    final langLabel = settings.applyLanguage(widget.engine);
    await _runSimulatedVad(samples, langLabel, realtime: false);
  }

  /// Test accessor: the running transcript (utterances appended in order).
  @visibleForTesting
  String get transcriptForTest => _transcript;

  /// Test accessor: number of utterances decoded and appended this session
  /// (captured at finalize; the live pipeline's counter after teardown).
  @visibleForTesting
  int get utterancesDoneForTest => _lastVadAppended;

  // ---------------------------------------------------------------------------
  // Simulated microphone (WAV file through the SAME push pipeline)
  // ---------------------------------------------------------------------------

  Future<void> _startSimulated(String wavPath) async {
    if (_running || _simulating || _vadActive) return;
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
    // Language: SIM_LANGUAGE override (automation) else the persisted setting.
    final settings = _settings ?? StreamingSettings();
    if (kSimLanguageOverride.isNotEmpty) {
      settings.languageId = kSimLanguageOverride;
    }
    // Live mode: SIM_LIVE_MODE override (automation) else the persisted setting.
    if (kSimLiveModeOverride.isNotEmpty) {
      settings.liveModeId = kSimLiveModeOverride;
    }
    final langLabel = settings.applyLanguage(widget.engine);

    // VAD Live: feed the WAV through the client-side segmenter (NOT the
    // streaming engine), exactly as a real mic would be ingested.
    if (settings.liveMode.isVadLive) {
      debugPrint(
        'QASR_METRIC session_start mode=sim liveMode=vad language=$langLabel',
      );
      await _runSimulatedVad(samples, langLabel);
      return;
    }

    final modeLabel = settings.applyLiveMode(widget.engine);
    widget.engine.setStreamChunkSec(simEngineChunk);
    await widget.engine.streamReset();
    // Test/automation hook: confirm the settings UI drives the engine.
    debugPrint(
      'QASR_METRIC session_start mode=sim language=$langLabel '
      'liveMode=$modeLabel engineChunk=${simEngineChunk.toStringAsFixed(1)}s',
    );
    setState(() {
      _simulating = true;
      _status =
          'Simulating mic from WAV (engineChunk='
          '${simEngineChunk.toStringAsFixed(1)}s, language=$langLabel)...';
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

  /// Encode float samples (-1..1) to little-endian PCM16 bytes — the format the
  /// VadHandler's `audioStream` expects (mirrors what the mic recorder emits).
  Uint8List _floatToPcm16(List<double> samples) {
    final out = Uint8List(samples.length * 2);
    final view = ByteData.view(out.buffer);
    for (var i = 0; i < samples.length; i++) {
      var v = (samples[i] * 32768.0).round();
      if (v > 32767) v = 32767;
      if (v < -32768) v = -32768;
      view.setInt16(i * 2, v, Endian.little);
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

  /// Start dispatcher: branch on the selected live mode.
  Future<void> _startLive() async {
    final settings = _settings ?? StreamingSettings();
    if (settings.liveMode.isVadLive) {
      await _startVadLiveMic();
    } else {
      await _startMic();
    }
  }

  /// Stop dispatcher: match the mode that is actually running.
  Future<void> _stopLive() async {
    if (_vadActive) {
      await _stopVadLiveMic();
    } else {
      await _stopMic();
    }
  }

  @override
  Widget build(BuildContext context) {
    final micActive = _running || _vadActive;
    final busy = micActive || _simulating;
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
                      : (micActive ? _stopLive : _startLive),
                  icon: Icon(micActive ? Icons.stop : Icons.mic),
                  label: Text(micActive ? 'Stop' : 'Start Mic'),
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
          _buildSettingsCard(context, busy),
          const SizedBox(height: 12),
          if (_vadActive) _buildVadIndicator(context),
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

  /// Settings section: language mode + engine chunk. Editable any time except
  /// while a session is running (disabled when `busy`); selections persist and
  /// apply at the next Start (no model reload).
  Widget _buildSettingsCard(BuildContext context, bool busy) {
    final settings = _settings;
    return Card(
      key: const Key('settings_card'),
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                const Icon(Icons.tune, size: 18),
                const SizedBox(width: 6),
                Text(
                  'Streaming settings',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const Spacer(),
                if (busy)
                  Text(
                    'locked while running',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
              ],
            ),
            const SizedBox(height: 4),
            if (settings == null)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 8),
                child: Text('Loading settings...'),
              )
            else ...[
              DropdownButtonFormField<String>(
                key: const Key('language_dropdown'),
                initialValue: settings.languageId,
                decoration: const InputDecoration(
                  labelText: '语言 / Language',
                  isDense: true,
                  border: OutlineInputBorder(),
                ),
                items: [
                  for (final o in kLanguageOptions)
                    DropdownMenuItem(value: o.id, child: Text(o.label)),
                ],
                onChanged: busy
                    ? null
                    : (v) {
                        if (v == null) return;
                        setState(() => settings.languageId = v);
                        settings.save();
                      },
              ),
              const SizedBox(height: 10),
              DropdownButtonFormField<String>(
                key: const Key('live_mode_dropdown'),
                initialValue: settings.liveModeId,
                decoration: const InputDecoration(
                  labelText: '实时模式 / Live mode',
                  isDense: true,
                  border: OutlineInputBorder(),
                ),
                items: [
                  for (final o in kLiveModeOptions)
                    DropdownMenuItem(value: o.id, child: Text(o.label)),
                ],
                onChanged: busy
                    ? null
                    : (v) {
                        if (v == null) return;
                        setState(() => settings.liveModeId = v);
                        settings.save();
                      },
              ),
              const SizedBox(height: 4),
              Text(
                settings.liveMode.description,
                key: const Key('live_mode_description'),
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// VAD-Live status indicator: a red "listening" dot while capturing, a
  /// spinner while an utterance is being decoded. There is no partial text in
  /// this mode by design, so this stands in for per-word feedback.
  Widget _buildVadIndicator(BuildContext context) {
    final decoding = _pipeline?.isBusy ?? false;
    return Padding(
      key: const Key('vad_indicator'),
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          if (decoding)
            const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          else
            Icon(
              Icons.fiber_manual_record,
              size: 14,
              color: _capturing ? Colors.red : Theme.of(context).disabledColor,
            ),
          const SizedBox(width: 8),
          Text(
            decoding
                ? 'Transcribing utterance...'
                : (_capturing ? 'Listening for speech...' : 'Finishing...'),
            style: Theme.of(context).textTheme.bodySmall,
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
