import 'dart:typed_data';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'rust/api/qwen_asr_bridge.dart';
import 'rust/frb_generated.dart';

/// On-device speech-to-text engine powered by Qwen3-ASR.
///
/// ```dart
/// final engine = await QAsrEngine.load('/path/to/model');
/// final text = await engine.transcribeFile('audio.wav');
/// engine.dispose();
/// ```
class QAsrEngine {
  final QwenAsrEngine _engine;
  QAsrEngine._(this._engine);

  static bool _initialized = false;

  /// Initialize the Rust library with a custom dylib path.
  /// Call this before [load] when running outside a Flutter app
  /// (e.g. in `flutter test`).
  static Future<void> initWith({required String dylibPath}) async {
    if (_initialized) return;
    await RustLib.init(externalLibrary: ExternalLibrary.open(dylibPath));
    _initialized = true;
  }

  /// Load a Qwen3-ASR model from [modelDir].
  ///
  /// The directory must contain `model*.safetensors` and `vocab.json`.
  /// Set [threads] to control parallelism (0 = auto-detect CPU count) and
  /// [verbosity] for logging (0 = silent, 1 = info, 2 = debug).
  ///
  /// Throws an [Exception] if the model cannot be loaded.
  static Future<QAsrEngine> load(
    String modelDir, {
    int threads = 0,
    int verbosity = 0,
  }) async {
    if (!_initialized) {
      await RustLib.init();
      _initialized = true;
    }
    final engine = await QwenAsrEngine.load(
      modelDir: modelDir,
      nThreads: threads,
      verbosity: verbosity,
    );
    if (engine == null) {
      throw Exception('Failed to load model from $modelDir');
    }
    return QAsrEngine._(engine);
  }

  /// Transcribe a WAV file at [wavPath].
  Future<String> transcribeFile(String wavPath) async {
    final result = await _engine.transcribeFile(wavPath: wavPath);
    return result ?? '';
  }

  /// Transcribe raw PCM audio.
  ///
  /// [samples] must be a [Float32List] of 16 kHz mono audio with values
  /// normalized to the range -1.0 to 1.0.
  Future<String> transcribePcm(Float32List samples) async {
    final result = await _engine.transcribePcm(samples: samples.toList());
    return result ?? '';
  }

  /// Transcribe from a WAV file buffer (bytes).
  Future<String> transcribeWavBuffer(Uint8List wavData) async {
    final result = await _engine.transcribeWavBuffer(wavData: wavData.toList());
    return result ?? '';
  }

  /// Set segment duration in seconds for splitting long audio.
  ///
  /// Use 30.0 as a good default for long recordings. Set to 0 to disable
  /// segmentation (transcribe the entire file in one pass).
  void setSegmentSec(double sec) {
    _engine.setSegmentSec(sec: sec);
  }

  /// Force a specific language (e.g. `"English"`, `"Chinese"`, `"Japanese"`).
  ///
  /// Pass an empty string to return to auto-detection. Returns `true` if the
  /// language name was recognized, `false` otherwise.
  bool setLanguage(String language) {
    return _engine.setLanguage(language: language);
  }

  /// Get performance stats from last transcription.
  String perfStats() {
    return _engine.perfStats();
  }

  /// Dispose the engine and free resources.
  void dispose() {
    _engine.dispose();
  }
}
