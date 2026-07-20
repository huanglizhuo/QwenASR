import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qwen_asr/qwen_asr.dart';

/// File transcription demo: a bundled no-setup sample plus a device file picker.
///
/// The picker uses the platform document picker (Android Storage Access
/// Framework / iOS UIDocumentPicker), which needs no runtime storage permission
/// or manifest/Info.plist entry for a single-file, read-only pick — so no
/// permission plumbing is required here. Only WAV (PCM16) files are supported:
/// the engine's `transcribeWavBuffer` parses the RIFF container and handles
/// arbitrary sample-rate / channel-count downmix+resample in Rust.
class FileScreen extends StatefulWidget {
  final QAsrEngine engine;
  const FileScreen({super.key, required this.engine});

  @override
  State<FileScreen> createState() => _FileScreenState();
}

class _FileScreenState extends State<FileScreen> {
  String _transcript = '';
  String _perf = '';
  String _status = 'Idle';
  bool _busy = false;

  Future<void> _transcribeAsset() async {
    setState(() {
      _busy = true;
      _status = 'Transcribing bundled asset...';
      _transcript = '';
      _perf = '';
    });
    try {
      final data = await rootBundle.load('test_fixtures/audio.wav');
      await _runTranscription(data.buffer.asUint8List(), 'file');
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Failed: $e';
        _busy = false;
      });
    }
  }

  Future<void> _pickAndTranscribe() async {
    setState(() {
      _busy = true;
      _status = 'Opening file picker...';
      _transcript = '';
      _perf = '';
    });
    FilePickerResult? result;
    try {
      // FileType.any + in-app WAV validation: SAF/UIDocumentPicker returns
      // content URIs whose extension is not always reliable, so we accept any
      // pick and validate the RIFF/WAVE magic from the bytes below.
      result = await FilePicker.pickFiles(withData: true);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'File picker error: $e';
        _busy = false;
      });
      return;
    }
    if (result == null || result.files.isEmpty) {
      // User cancelled — not an error.
      if (!mounted) return;
      setState(() {
        _status = 'Pick cancelled';
        _busy = false;
      });
      return;
    }

    final picked = result.files.first;
    final bytes = picked.bytes;
    if (bytes == null || bytes.isEmpty) {
      if (!mounted) return;
      setState(() {
        _status = 'Could not read file bytes for "${picked.name}"';
        _busy = false;
      });
      return;
    }
    if (!_looksLikeWav(bytes)) {
      if (!mounted) return;
      setState(() {
        _status =
            'Unsupported format: "${picked.name}". Only WAV (PCM16) files are '
            'supported — please pick a .wav file.';
        _busy = false;
      });
      return;
    }

    setState(() => _status = 'Transcribing "${picked.name}"...');
    try {
      await _runTranscription(bytes, 'picked');
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Failed: $e';
        _busy = false;
      });
    }
  }

  /// Transcribe WAV bytes and update the UI + logcat hooks. [tag] labels the
  /// automation lines (`file` for the bundled sample, `picked` for a picked file).
  Future<void> _runTranscription(Uint8List wavBytes, String tag) async {
    final result = await widget.engine.transcribeWavBuffer(wavBytes);
    if (!mounted) return;
    final perf = widget.engine.perfStats();
    setState(() {
      _transcript = result;
      _perf = perf;
      _status = 'Done';
      _busy = false;
    });
    // Test/automation hook: emit final result to logcat (grep QASR_).
    debugPrint('QASR_METRIC ${tag}_perf | $perf');
    debugPrint('QASR_TRANSCRIPT $tag | $result');
  }

  /// True if [bytes] begins with a `RIFF....WAVE` header.
  bool _looksLikeWav(Uint8List bytes) {
    if (bytes.lengthInBytes < 12) return false;
    final bd = ByteData.view(
      bytes.buffer,
      bytes.offsetInBytes,
      bytes.lengthInBytes,
    );
    return bd.getUint32(0, Endian.big) == 0x52494646 /*RIFF*/ &&
        bd.getUint32(8, Endian.big) == 0x57415645 /*WAVE*/;
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FilledButton.icon(
            key: const Key('transcribe_asset_button'),
            onPressed: _busy ? null : _transcribeAsset,
            icon: const Icon(Icons.audiotrack),
            label: const Text('Transcribe Bundled Sample'),
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            key: const Key('pick_file_button'),
            onPressed: _busy ? null : _pickAndTranscribe,
            icon: const Icon(Icons.folder_open),
            label: const Text('Pick Audio File (WAV)'),
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
            'Transcript:',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Expanded(
            child: SingleChildScrollView(
              child: SelectableText(
                _transcript.isEmpty ? '(none)' : _transcript,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
