import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:qwen_asr/qwen_asr.dart';

/// Fixed engine streaming chunk, in seconds.
///
/// 2.0 s is the ONLY validated on-device value: 1 s and 3 s both produce
/// duplicated phrases on-device (R13-Android), so no chunk *selector* is
/// offered — the live-mic path always uses this constant. (The simulated-mic /
/// automation path keeps its own separate override; see `streaming_screen.dart`.)
const double kEngineChunkSec = 2.0;

/// A selectable language option for the streaming settings UI.
class LanguageOption {
  /// Stable persisted key.
  final String id;

  /// Human-readable label (bilingual where useful).
  final String label;

  /// Engine language name passed to `setLanguage` (from the model's
  /// `normalize_language` table), or `null` for auto-multilingual mode.
  final String? engineName;

  const LanguageOption(this.id, this.label, this.engineName);

  bool get isAuto => engineName == null;
}

/// Language choices. `auto` (multilingual) is the default. The forced-language
/// options are all valid `normalize_language` codes the 0.6B model supports.
const List<LanguageOption> kLanguageOptions = [
  LanguageOption('auto', '自动（多语言）/ Auto-multilingual', null),
  LanguageOption('zh', '中文 (Chinese)', 'Chinese'),
  LanguageOption('en', 'English', 'English'),
  LanguageOption('ja', '日本語 (Japanese)', 'Japanese'),
  LanguageOption('ko', '한국어 (Korean)', 'Korean'),
  LanguageOption('yue', '粤语 (Cantonese)', 'Cantonese'),
];

/// A selectable live-streaming mode for the settings UI.
class LiveModeOption {
  /// Stable persisted key.
  final String id;

  /// Human-readable title.
  final String label;

  /// One-line explanation shown under the selector.
  final String description;

  /// Whether this mode drives the streaming engine's `stream_push_audio`
  /// pipeline (Full Streaming). `false` for VAD Live, which is client-side:
  /// it segments the mic stream into utterances in Dart and decodes each one
  /// with an independent one-shot offline `transcribePcm` call — the streaming
  /// engine (and its `vad_segment_reset` flag) is not involved at all.
  final bool usesStreamingEngine;

  const LiveModeOption(
    this.id,
    this.label,
    this.description,
    this.usesStreamingEngine,
  );

  /// True for the client-side VAD-Live mode.
  bool get isVadLive => !usesStreamingEngine;
}

/// Live-mode choices. `full` (continuous rolling stream) is the default; `vad`
/// (VAD Live) is client-side: Dart VAD splits the mic stream at silence gaps and
/// each utterance is decoded independently offline.
const List<LiveModeOption> kLiveModeOptions = [
  LiveModeOption(
    'full',
    'Full Streaming',
    'Continuous rolling stream — live per-word partials, best over long speech.',
    true,
  ),
  LiveModeOption(
    'vad',
    'VAD Live',
    'Detects pauses and transcribes each utterance independently offline — no '
        'live partial text, but immune to streaming drift.',
    false,
  ),
];

/// Persisted streaming settings: chosen language mode + live mode.
///
/// Editable any time except while a session runs; changes apply at the next
/// Start (session-level engine setters, no model reload). Persisted as a small
/// JSON file in the app documents directory (no extra plugin dependency).
class StreamingSettings {
  String languageId;
  String liveModeId;

  StreamingSettings({this.languageId = 'auto', this.liveModeId = 'full'});

  /// Fixed engine chunk (2.0 s) — no longer user-selectable. Kept as a property
  /// so the streaming session wiring (`setStreamChunkSec`) stays unchanged.
  double get engineChunkSec => kEngineChunkSec;

  LanguageOption get language => kLanguageOptions.firstWhere(
    (o) => o.id == languageId,
    orElse: () => kLanguageOptions.first,
  );

  LiveModeOption get liveMode => kLiveModeOptions.firstWhere(
    (o) => o.id == liveModeId,
    orElse: () => kLiveModeOptions.first,
  );

  static Future<File> _file() async {
    final docs = await getApplicationDocumentsDirectory();
    return File('${docs.path}/streaming_settings.json');
  }

  /// Load persisted settings, falling back to defaults on any error.
  static Future<StreamingSettings> load() async {
    final s = StreamingSettings();
    try {
      final f = await _file();
      if (await f.exists()) {
        final m = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
        final id = m['languageId'];
        if (id is String && kLanguageOptions.any((o) => o.id == id)) {
          s.languageId = id;
        }
        final mode = m['liveModeId'];
        if (mode is String && kLiveModeOptions.any((o) => o.id == mode)) {
          s.liveModeId = mode;
        }
      }
    } catch (_) {
      // Ignore and use defaults.
    }
    return s;
  }

  /// Persist the current selections.
  Future<void> save() async {
    try {
      final f = await _file();
      await f.writeAsString(
        jsonEncode({'languageId': languageId, 'liveModeId': liveModeId}),
      );
    } catch (_) {
      // Best-effort; non-fatal for the demo.
    }
  }

  /// Apply the language selection to the engine (session-level; call before
  /// `streamReset`). Auto-multilingual enables per-utterance re-detection;
  /// otherwise force the chosen language. Returns a short label for logging.
  String applyLanguage(QAsrEngine engine) {
    final opt = language;
    if (opt.isAuto) {
      engine.setMultilingual(true);
      return 'auto-multilingual';
    }
    engine.setMultilingual(false);
    engine.setLanguage(opt.engineName!);
    return opt.engineName!;
  }

  /// Apply the live-mode selection to the engine (session-level; call before
  /// `streamReset`). The app no longer uses the in-streaming `vad_segment_reset`
  /// flag for either mode — Full Streaming is a plain continuous stream and VAD
  /// Live is client-side one-shot — so this always ensures the engine flag is
  /// OFF (the library capability stays intact for other integrators). Called on
  /// the Full-Streaming path; the VAD-Live path does not touch the engine.
  /// Returns a short label for logging.
  String applyLiveMode(QAsrEngine engine) {
    engine.setVadSegmentReset(false);
    return liveMode.id;
  }
}
