import 'dart:typed_data';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'rust/api/qasr_bridge.dart';
import 'rust/frb_generated.dart';

class QAsrEngine {
  final QasrEngine _engine;
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

  /// Load model from [modelDir]. Optionally set [threads] (0 = auto)
  /// and [verbosity] (0 = silent, 1 = info, 2 = debug).
  static Future<QAsrEngine> load(
    String modelDir, {
    int threads = 0,
    int verbosity = 0,
  }) async {
    if (!_initialized) {
      await RustLib.init();
      _initialized = true;
    }
    final engine = await QasrEngine.load(
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

  /// Transcribe raw PCM f32 samples (16kHz mono).
  Future<String> transcribePcm(Float32List samples) async {
    final result = await _engine.transcribePcm(samples: samples.toList());
    return result ?? '';
  }

  /// Transcribe from a WAV file buffer (bytes).
  Future<String> transcribeWavBuffer(Uint8List wavData) async {
    final result = await _engine.transcribeWavBuffer(wavData: wavData.toList());
    return result ?? '';
  }

  /// Set segment duration in seconds (0 = no segmentation).
  void setSegmentSec(double sec) {
    _engine.setSegmentSec(sec: sec);
  }

  /// Set forced language. Returns true if valid.
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
