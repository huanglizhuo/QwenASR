# qwen_asr

On-device Qwen3-ASR speech-to-text for Flutter. Runs entirely on the device
using a pure Rust inference engine — no cloud services, no server, no network
required. Supports iOS, Android, and macOS.

The native Rust engine is compiled in release mode automatically by the Flutter
build pipeline. If building the Rust library manually, always use `--release` —
debug builds are 10–50x slower and unusable for real-time inference.

## Setup

Add the dependency:

```yaml
dependencies:
  qwen_asr: ^0.1.0
```

Download the model (e.g. in your app's asset pipeline or at first launch):

```bash
huggingface_hub download Qwen/Qwen3-ASR-0.6B --local-dir <your-model-dir>
```

## Usage

```dart
import 'package:qwen_asr/qwen_asr.dart';

// Load the model (once, typically at app start)
final engine = await QAsrEngine.load('/path/to/qwen3-asr-0.6b');

// Transcribe a WAV file
final text = await engine.transcribeFile('/path/to/audio.wav');
print(text);

// Transcribe raw PCM (Float32List, 16 kHz, mono, values in -1..1)
final pcmText = await engine.transcribePcm(samples);

// Transcribe from an in-memory WAV buffer
final bufText = await engine.transcribeWavBuffer(wavBytes);

// Clean up when done
engine.dispose();
```

## Configuration

```dart
// Force a language (e.g. "English", "Chinese"); empty string = auto-detect
engine.setLanguage('English');

// Enable segmented mode for long audio (seconds per segment, 0 = off)
engine.setSegmentSec(30.0);

// Get timing stats from the last transcription
print(engine.perfStats());
```

## Forced Alignment

Forced alignment (word-level timestamps for a known transcript) is available
through the Rust library (`qwen_asr::align::forced_align`) for native
integration. See the [crate documentation](../crates/qwen-asr/README.md) for
details.

## Platform Support

| Platform | Status |
|----------|--------|
| iOS      | Supported |
| Android  | Supported |
| macOS    | Supported |
| Linux    | Planned |
| Windows  | Planned |

## License

MIT
