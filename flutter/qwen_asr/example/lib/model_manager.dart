import 'dart:async';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

/// Files that make up the Qwen3-ASR 0.6B model. Mirrors the CLI download list
/// in `crates/qwen-asr-cli/src/download.rs`.
const List<String> kModelFiles = [
  'model.safetensors',
  'vocab.json',
  'merges.txt',
];

/// Default download base URL (HuggingFace). The `resolve/main/<file>` pattern
/// mirrors the CLI. Override in the UI to point at a local test server.
const String kDefaultBaseUrl =
    'https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/';

/// Model directory name inside the app documents directory.
const String kModelDirName = 'qwen3-asr-0.6b';

/// Per-file download progress snapshot.
class FileProgress {
  final String fileName;
  final int received;
  final int total; // 0 if unknown
  final bool done;
  final String? error;

  const FileProgress({
    required this.fileName,
    this.received = 0,
    this.total = 0,
    this.done = false,
    this.error,
  });

  double get fraction => total > 0 ? (received / total).clamp(0.0, 1.0) : 0.0;
}

/// Handles locating, verifying, downloading, and deleting the on-device model.
class ModelManager {
  /// Returns the absolute path to the model directory in app documents.
  static Future<String> modelDir() async {
    final docs = await getApplicationDocumentsDirectory();
    return '${docs.path}/$kModelDirName';
  }

  /// True if every model file exists with a non-zero size.
  static Future<bool> isModelReady() async {
    final dir = await modelDir();
    for (final f in kModelFiles) {
      final file = File('$dir/$f');
      if (!await file.exists()) return false;
      if (await file.length() <= 0) return false;
    }
    return true;
  }

  /// Delete the model directory (used for corruption recovery / retry).
  static Future<void> deleteModel() async {
    final dir = Directory(await modelDir());
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }

  /// Build a per-file URL from a base URL, tolerating a missing trailing slash.
  static String fileUrl(String baseUrl, String fileName) {
    final base = baseUrl.endsWith('/') ? baseUrl : '$baseUrl/';
    return '$base$fileName';
  }

  /// Download all model files, streaming progress to [onProgress].
  ///
  /// Downloads to `<file>.part` then renames on success so a partial file is
  /// never mistaken for a complete one. Verifies each file is > 0 bytes.
  /// Throws on any failure (network, zero-byte, write error).
  static Future<void> download({
    required String baseUrl,
    required void Function(FileProgress) onProgress,
  }) async {
    final dir = await modelDir();
    await Directory(dir).create(recursive: true);
    final client = http.Client();
    try {
      for (final fileName in kModelFiles) {
        final dest = File('$dir/$fileName');
        // Skip files already present and non-empty.
        if (await dest.exists() && await dest.length() > 0) {
          onProgress(FileProgress(fileName: fileName, done: true));
          continue;
        }
        await _downloadOne(
          client: client,
          url: fileUrl(baseUrl, fileName),
          dest: dest,
          fileName: fileName,
          onProgress: onProgress,
        );
      }
    } finally {
      client.close();
    }
  }

  static Future<void> _downloadOne({
    required http.Client client,
    required String url,
    required File dest,
    required String fileName,
    required void Function(FileProgress) onProgress,
  }) async {
    final partFile = File('${dest.path}.part');
    if (await partFile.exists()) {
      await partFile.delete();
    }

    onProgress(FileProgress(fileName: fileName, received: 0, total: 0));

    final request = http.Request('GET', Uri.parse(url));
    final response = await client.send(request);

    if (response.statusCode != 200) {
      throw Exception('HTTP ${response.statusCode} for $fileName');
    }

    final total = response.contentLength ?? 0;
    var received = 0;
    final sink = partFile.openWrite();
    try {
      await for (final chunk in response.stream) {
        sink.add(chunk);
        received += chunk.length;
        onProgress(
          FileProgress(fileName: fileName, received: received, total: total),
        );
      }
      await sink.flush();
    } finally {
      await sink.close();
    }

    if (received <= 0 || await partFile.length() <= 0) {
      await partFile.delete();
      throw Exception('Downloaded 0 bytes for $fileName');
    }

    // Atomic-ish finalize: rename .part -> final.
    if (await dest.exists()) {
      await dest.delete();
    }
    await partFile.rename(dest.path);

    onProgress(
      FileProgress(
        fileName: fileName,
        received: received,
        total: total,
        done: true,
      ),
    );
  }
}
