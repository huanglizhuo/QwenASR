import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:qwen_asr/qwen_asr.dart';

/// Whether the 1.0 s low-latency engine-chunk option is offered in the UI.
///
/// Gated on the R13-Android device chunk test (Task 5.1): the 1 s option ships
/// only if it passes the cool-protocol RTF / word-match / no-duplication gate.
/// Flip to `true` once the device gate passes; keep `false` to omit it.
const bool kAllow1sChunk = false;

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

/// Engine-chunk choices (seconds). 1.0 s is low-latency (gated); 2.0 s default.
/// Values >= 3 s are deliberately NOT offered (on-device sentence-duplication).
class ChunkOption {
  final double sec;
  final String label;
  const ChunkOption(this.sec, this.label);
}

const List<ChunkOption> _allChunkOptions = [
  ChunkOption(1.0, '1.0 s (低延迟 / low-latency)'),
  ChunkOption(2.0, '2.0 s (默认 / default)'),
];

List<ChunkOption> get kChunkOptions => _allChunkOptions
    .where((o) => o.sec != 1.0 || kAllow1sChunk)
    .toList(growable: false);

/// Persisted streaming settings: chosen language mode + engine chunk size.
///
/// Editable any time except while a session runs; changes apply at the next
/// Start (session-level engine setters, no model reload). Persisted as a small
/// JSON file in the app documents directory (no extra plugin dependency).
class StreamingSettings {
  String languageId;
  double engineChunkSec;

  StreamingSettings({this.languageId = 'auto', this.engineChunkSec = 2.0});

  LanguageOption get language => kLanguageOptions.firstWhere(
    (o) => o.id == languageId,
    orElse: () => kLanguageOptions.first,
  );

  static Future<File> _file() async {
    final docs = await getApplicationDocumentsDirectory();
    return File('${docs.path}/streaming_settings.json');
  }

  /// Load persisted settings, falling back to defaults on any error. Coerces an
  /// out-of-range chunk (e.g. a persisted 1.0 s while the option is gated off)
  /// back to the default so the UI always has a valid selection.
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
        final sec = (m['engineChunkSec'] as num?)?.toDouble();
        if (sec != null && kChunkOptions.any((o) => o.sec == sec)) {
          s.engineChunkSec = sec;
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
        jsonEncode({
          'languageId': languageId,
          'engineChunkSec': engineChunkSec,
        }),
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
}
