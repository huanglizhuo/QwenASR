import 'package:flutter/material.dart';
import 'package:qwen_asr/qwen_asr.dart';

import 'download_screen.dart';
import 'file_screen.dart';
import 'model_manager.dart';
import 'record_screen.dart';
import 'streaming_screen.dart';

// --- Test / automation hooks (via --dart-define) ---
// Override the download base URL and auto-start the download.
const String _kBaseUrl = String.fromEnvironment('DOWNLOAD_BASE_URL');
const bool _kAutoDownload = bool.fromEnvironment('AUTO_DOWNLOAD');
// Path to a WAV to auto-run through the simulated-mic path once loaded.
const String _kAutoSimWav = String.fromEnvironment('AUTO_SIM_WAV');
// Thread count for the engine (0 = auto). Used for tuning experiments.
const int _kThreads = int.fromEnvironment('THREADS', defaultValue: 0);

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
      home: const AppRoot(),
    );
  }
}

enum _Phase { checking, needDownload, loading, ready, error }

class AppRoot extends StatefulWidget {
  const AppRoot({super.key});

  @override
  State<AppRoot> createState() => _AppRootState();
}

class _AppRootState extends State<AppRoot> {
  _Phase _phase = _Phase.checking;
  QAsrEngine? _engine;
  String _error = '';

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _engine?.dispose();
    super.dispose();
  }

  Future<void> _bootstrap() async {
    setState(() => _phase = _Phase.checking);
    if (await ModelManager.isModelReady()) {
      await _loadEngine();
    } else {
      setState(() => _phase = _Phase.needDownload);
    }
  }

  Future<void> _loadEngine() async {
    setState(() => _phase = _Phase.loading);
    try {
      final dir = await ModelManager.modelDir();
      final sw = Stopwatch()..start();
      final engine = await QAsrEngine.load(dir, threads: _kThreads);
      sw.stop();
      // Test/automation hook: emit load time to logcat (grep QASR_METRIC).
      debugPrint(
        'QASR_METRIC load_ms=${sw.elapsedMilliseconds} threads=$_kThreads',
      );
      if (!mounted) return;
      setState(() {
        _engine = engine;
        _phase = _Phase.ready;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _phase = _Phase.error;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    switch (_phase) {
      case _Phase.checking:
      case _Phase.loading:
        return _busyScaffold(
          _phase == _Phase.checking
              ? 'Checking for model...'
              : 'Loading model...',
        );
      case _Phase.needDownload:
        return DownloadScreen(
          initialBaseUrl: _kBaseUrl.isNotEmpty ? _kBaseUrl : null,
          autoStart: _kAutoDownload,
          onComplete: _loadEngine,
        );
      case _Phase.error:
        return Scaffold(
          appBar: AppBar(title: const Text('Error')),
          body: Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Failed to load model:\n$_error',
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 16),
                  FilledButton(
                    onPressed: () async {
                      await ModelManager.deleteModel();
                      _bootstrap();
                    },
                    child: const Text('Delete & Re-download'),
                  ),
                ],
              ),
            ),
          ),
        );
      case _Phase.ready:
        return HomeTabs(engine: _engine!);
    }
  }

  Widget _busyScaffold(String message) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text(message),
          ],
        ),
      ),
    );
  }
}

class HomeTabs extends StatelessWidget {
  final QAsrEngine engine;
  const HomeTabs({super.key, required this.engine});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Qwen ASR Demo'),
          bottom: const TabBar(
            tabs: [
              Tab(icon: Icon(Icons.mic), text: 'Live'),
              Tab(icon: Icon(Icons.radio_button_checked), text: 'Record'),
              Tab(icon: Icon(Icons.audiotrack), text: 'File'),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            StreamingScreen(
              engine: engine,
              autoSimWavPath: _kAutoSimWav.isNotEmpty ? _kAutoSimWav : null,
            ),
            RecordScreen(engine: engine),
            FileScreen(engine: engine),
          ],
        ),
      ),
    );
  }
}
