import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qwen_asr/qwen_asr.dart';

void main() {
  runApp(const QAsrDemoApp());
}

class QAsrDemoApp extends StatelessWidget {
  const QAsrDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Qwen ASR Demo',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.blue),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _modelDirController = TextEditingController();
  QAsrEngine? _engine;
  String _status = 'Not loaded';
  String _transcript = '';
  String _perfStats = '';
  bool _loading = false;

  @override
  void dispose() {
    _engine?.dispose();
    _modelDirController.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    final dir = _modelDirController.text.trim();
    if (dir.isEmpty) return;

    setState(() {
      _loading = true;
      _status = 'Loading model...';
    });

    try {
      _engine?.dispose();
      final engine = await QAsrEngine.load(dir, verbosity: 1);
      setState(() {
        _engine = engine;
        _status = 'Model loaded';
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Load failed: $e';
        _loading = false;
      });
    }
  }

  Future<void> _transcribeAsset() async {
    if (_engine == null) return;

    setState(() {
      _loading = true;
      _status = 'Transcribing...';
      _transcript = '';
    });

    try {
      final data = await rootBundle.load('test_fixtures/audio.wav');
      final result =
          await _engine!.transcribeWavBuffer(data.buffer.asUint8List());
      setState(() {
        _transcript = result;
        _perfStats = _engine!.perfStats();
        _status = 'Done';
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Transcribe failed: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Qwen ASR Demo')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _modelDirController,
              decoration: const InputDecoration(
                labelText: 'Model directory path',
                hintText: '/path/to/qwen3-asr-0.6b',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                ElevatedButton(
                  onPressed: _loading ? null : _loadModel,
                  child: const Text('Load Model'),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed: _loading || _engine == null
                      ? null
                      : _transcribeAsset,
                  child: const Text('Transcribe Asset'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text('Status: $_status'),
            if (_perfStats.isNotEmpty) Text('Perf: $_perfStats'),
            const SizedBox(height: 12),
            const Text('Transcript:',
                style: TextStyle(fontWeight: FontWeight.bold)),
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
      ),
    );
  }
}
