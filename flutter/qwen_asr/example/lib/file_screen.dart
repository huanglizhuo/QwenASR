import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qwen_asr/qwen_asr.dart';

/// The original file / bundled-asset transcription demo, kept reachable.
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
    });
    try {
      final data = await rootBundle.load('test_fixtures/audio.wav');
      final result = await widget.engine.transcribeWavBuffer(
        data.buffer.asUint8List(),
      );
      if (!mounted) return;
      final perf = widget.engine.perfStats();
      setState(() {
        _transcript = result;
        _perf = perf;
        _status = 'Done';
        _busy = false;
      });
      // Test/automation hook: emit final result to logcat (grep QASR_).
      debugPrint('QASR_METRIC file_perf | $perf');
      debugPrint('QASR_TRANSCRIPT file | $result');
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Failed: $e';
        _busy = false;
      });
    }
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
