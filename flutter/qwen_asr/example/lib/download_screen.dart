import 'package:flutter/material.dart';

import 'model_manager.dart';

/// Screen shown when the model is not yet on device. Downloads each model file
/// with a per-file progress bar and an advanced base-URL override (used to test
/// against a local HTTP server).
class DownloadScreen extends StatefulWidget {
  final VoidCallback onComplete;

  /// Optional pre-filled base URL (e.g. injected via --dart-define for tests).
  final String? initialBaseUrl;

  /// If true, kick off the download automatically on first build (test hook).
  final bool autoStart;

  const DownloadScreen({
    super.key,
    required this.onComplete,
    this.initialBaseUrl,
    this.autoStart = false,
  });

  @override
  State<DownloadScreen> createState() => _DownloadScreenState();
}

class _DownloadScreenState extends State<DownloadScreen> {
  late final TextEditingController _urlController;
  final Map<String, FileProgress> _progress = {};
  bool _downloading = false;
  String? _error;
  bool _advancedOpen = false;

  @override
  void initState() {
    super.initState();
    _urlController = TextEditingController(
      text: widget.initialBaseUrl ?? kDefaultBaseUrl,
    );
    if (widget.autoStart) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _startDownload());
    }
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _startDownload() async {
    if (_downloading) return;
    setState(() {
      _downloading = true;
      _error = null;
      _progress.clear();
    });

    try {
      await ModelManager.download(
        baseUrl: _urlController.text.trim(),
        onProgress: (p) {
          if (!mounted) return;
          setState(() => _progress[p.fileName] = p);
        },
      );
      if (!await ModelManager.isModelReady()) {
        throw Exception('Model incomplete after download');
      }
      if (!mounted) return;
      widget.onComplete();
    } catch (e) {
      // Corruption recovery: wipe partial download so retry is clean.
      await ModelManager.deleteModel();
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _downloading = false;
      });
    }
  }

  String _fmtBytes(int b) {
    if (b >= 1 << 30) return '${(b / (1 << 30)).toStringAsFixed(1)} GB';
    if (b >= 1 << 20) return '${(b / (1 << 20)).toStringAsFixed(1)} MB';
    if (b >= 1 << 10) return '${(b / (1 << 10)).toStringAsFixed(1)} KB';
    return '$b B';
  }

  Widget _fileRow(String fileName) {
    final p = _progress[fileName];
    final done = p?.done ?? false;
    final frac = done ? 1.0 : (p?.fraction ?? 0.0);
    final label = p == null
        ? 'waiting'
        : done
        ? 'done (${_fmtBytes(p.received)})'
        : p.total > 0
        ? '${_fmtBytes(p.received)} / ${_fmtBytes(p.total)}'
        : _fmtBytes(p.received);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(child: Text(fileName)),
              Text(label, style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(
            value: (p == null || (!done && p.total == 0 && _downloading))
                ? null
                : frac,
          ),
        ],
      ),
    );
  }

  int get _totalReceived => _progress.values.fold(0, (a, p) => a + p.received);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Download Model')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'The Qwen3-ASR 0.6B model (~1.8 GB) is required for on-device '
              'transcription. It will be saved to the app documents directory.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 16),
            for (final f in kModelFiles) _fileRow(f),
            const SizedBox(height: 8),
            Text('Total downloaded: ${_fmtBytes(_totalReceived)}'),
            const SizedBox(height: 8),
            if (_error != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Error: $_error',
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onErrorContainer,
                  ),
                ),
              ),
            const SizedBox(height: 8),
            ExpansionPanelList(
              expansionCallback: (_, isOpen) =>
                  setState(() => _advancedOpen = !isOpen),
              children: [
                ExpansionPanel(
                  isExpanded: _advancedOpen,
                  headerBuilder: (_, _) =>
                      const ListTile(title: Text('Advanced')),
                  body: Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: TextField(
                      controller: _urlController,
                      enabled: !_downloading,
                      decoration: const InputDecoration(
                        labelText: 'Download base URL',
                        helperText:
                            'Files fetched from <baseUrl>/<file>. Override '
                            'to point at a local server.',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                ),
              ],
            ),
            const Spacer(),
            FilledButton.icon(
              key: const Key('download_button'),
              onPressed: _downloading ? null : _startDownload,
              icon: _downloading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.download),
              label: Text(
                _downloading
                    ? 'Downloading...'
                    : (_error != null ? 'Retry Download' : 'Download'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
