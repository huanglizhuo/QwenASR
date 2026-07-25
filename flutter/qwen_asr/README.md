# qwen_asr

[![pub package](https://img.shields.io/pub/v/qwen_asr.svg)](https://pub.dev/packages/qwen_asr)
[![ci](https://github.com/huanglizhuo/QwenASR/actions/workflows/ci.yml/badge.svg)](https://github.com/huanglizhuo/QwenASR/actions/workflows/ci.yml)

On-device Qwen3-ASR speech-to-text for Flutter, from the
[QwenASR](https://github.com/huanglizhuo/QwenASR) project. Runs entirely on the
device using a pure Rust inference engine — no cloud services, no server, no
network required (beyond the one-time model download you provide). Supports
Android, iOS, and macOS.

The pub package ships **precompiled native binaries** for all supported
platforms (Android, iOS, macOS), so consumers do **not** need a Rust toolchain.
If a precompiled binary is unavailable for your target (or you set
`use_precompiled_binaries: false` in `cargokit_options.yaml`), the Rust engine
is compiled from source in release mode automatically by the Flutter build
pipeline — that fallback requires a Rust toolchain. If you build the Rust
library manually, always use `--release` — debug builds are 10–50× slower and
unusable for real-time inference.

### Building from source (contributors)

Clone the [QwenASR](https://github.com/huanglizhuo/QwenASR) repository and use
the example app under `example/`; the Rust bridge crate (`rust/`) depends on
the engine crate by path, so source builds require the full repo checkout.

For a complete integration reference — model download UI, live streaming, VAD
segmentation, file transcription, and push-to-talk — see the
[example app](example/) and its [README](example/README.md).

## Setup

Add the dependency:

```yaml
dependencies:
  qwen_asr: ^0.4.0
```

The model is not bundled; ship or download it at runtime. The engine needs a
directory containing `model.safetensors`, `vocab.json`, and `merges.txt`
(the 0.6B model is ~1.8 GB). One way to fetch it:

```bash
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir <your-model-dir>
```

The example app downloads these files at first launch with a progress UI; see
`example/lib/model_manager.dart` for a self-contained downloader you can adapt.

## One-shot transcription

```dart
import 'package:qwen_asr/qwen_asr.dart';

// Load the model once, typically at app start (heavy: ~2–3 s on device).
final engine = await QAsrEngine.load('/path/to/qwen3-asr-0.6b');

// Transcribe a WAV file on disk.
final text = await engine.transcribeFile('/path/to/audio.wav');

// Transcribe raw PCM (Float32List, 16 kHz, mono, values in -1.0..1.0).
final pcmText = await engine.transcribePcm(samples);

// Transcribe from an in-memory WAV buffer (any sample rate / channel count —
// the engine downmixes and resamples to 16 kHz mono internally).
final bufText = await engine.transcribeWavBuffer(wavBytes);

// Free resources when done.
engine.dispose();
```

`transcribePcm` is the most WER-validated decode path in the project; the example
app builds both its push-to-talk Record tab and its VAD Live mode on top of it.

For long audio, enable segmentation so the engine splits the input into
fixed-length windows:

```dart
engine.setSegmentSec(30.0); // 30 s windows; 0 disables segmentation.
```

## Live streaming

The streaming API decodes a rolling microphone stream incrementally, emitting
partial transcripts as audio arrives.

```dart
// Optional: tune the engine before the session (see "Streaming tuning" below).
engine.setStreamChunkSec(2.0);      // decode cadence — the primary latency knob.
engine.setStreamUnfixedChunks(2);   // start committing stable text after 2 chunks.
engine.setPastTextConditioning(true);

await engine.streamReset();         // once, before capture starts.

// For each ~0.5 s chunk of 16 kHz mono PCM captured from the mic:
final StreamPartial p = await engine.streamPush(chunk);
render(stable: p.text, provisional: p.provisional);

// On the last chunk (Stop), flush the tail:
final StreamPartial last = await engine.streamPush(tail, finalize: true);
```

### Stable vs provisional text

`streamPush` returns a `StreamPartial` with two fields:

- **`text`** — the committed **stable** transcript accumulated so far. Once text
  lands here it does not change.
- **`provisional`** — the newest **unfixed tail**: a lower-confidence hypothesis
  for the most recent audio. It exists to cut perceived latency (you see words
  the moment they are decoded, before they are confirmed). A later `streamPush`
  may revise this tail or promote it into `text`. It is always empty after a
  `finalize: true` push.

Render the two distinctly — the example app draws `text` in normal weight and
`provisional` in grey italic — so users understand the tail may still change.

## Language and multilingual behavior

Two independent session-level controls, both applied **before** `streamReset`
(or before a one-shot call); neither reloads the model:

```dart
// Force a language for every utterance (recognized model language name).
engine.setLanguage('English');   // 'Chinese', 'Japanese', 'Korean', 'Cantonese', …
engine.setLanguage('');          // empty string = auto-detect

// Opt in to per-utterance language re-detection within a streaming session.
engine.setMultilingual(true);
```

`setLanguage` pins the decode to one language. `setMultilingual(true)` instead
clears any forced language and makes the **streaming engine** re-detect the
language from fresh audio at every utterance boundary (silence gap), so a Chinese
utterance followed by an English one is transcribed in each language rather than
the second being rendered as a translation of the first. This is a property of
the streaming decoder itself — distinct from any app-level VAD segmentation you
might build on top of the one-shot API.

## Streaming tuning

| Method | Default | Effect |
|--------|---------|--------|
| `setStreamChunkSec(sec)` | 8.0 | Engine decode cadence. Smaller = more frequent partials / lower latency. The engine only decodes once a full chunk has accumulated. |
| `setStreamUnfixedChunks(n)` | 99 | Leading chunks held "unfixed" before tokens commit to `text`. The default holds everything until finalize; use a small value (e.g. 2) for progressive live output. |
| `setStreamRollback(tokens)` | 5 | Token rollback window for revising the unfixed tail. |
| `setStreamMaxNewTokens(tokens)` | 32 | Max tokens decoded per streaming chunk. |
| `setPastTextConditioning(bool)` | — | Reuse previously decoded text as decoder context across chunks. Recommended `true` for streaming quality. |
| `setVadSegmentReset(bool)` | off | When on, each detected silence boundary hard-resets the decode segment (drops carried text + encoder/KV state) for discrete per-utterance segmentation. Composes with `setMultilingual`. |

`perfStats()` returns a timing summary string for the last transcription (encode
/ decode milliseconds, etc.).

## Performance

On desktop (macOS / Linux) the engine uses a BLAS backend (Accelerate /
OpenBLAS). The fused INT8 single-token decode path runs on hand-written SIMD
kernels — NEON on aarch64, AVX2 on x86_64 (bit-matched to the NEON
implementations, with a scalar fallback on other architectures).

Android ships no BLAS, so the Android build uses a **NEON pool-parallel no-BLAS
GEMM fallback** — hand-written NEON kernels sliced across the persistent thread
pool. This is the base path that makes real-time on-device ASR viable on a phone.

On top of that, two **optional, Android-only INT8 Cargo features** trade a small
amount of accuracy for speed. Both are:

- **Compile-time gated** behind `all(feature, not(feature = "blas"),
  target_arch = "aarch64")`, so desktop/iOS/CLI builds never compile a line of
  them — off-Android behavior is byte-for-byte identical by construction.
- **Wired ON by default for Android** in
  [`rust/Cargo.toml`](rust/Cargo.toml) (`cfg(target_os = "android")` →
  `features = ["int8-prefill", "int8-encoder"]`), because they are what make the
  mobile experience usable. Each has a runtime kill switch.

| Feature | Wins | Accuracy cost | Status |
|---------|------|---------------|--------|
| `int8-prefill` | **~4.3× faster** decoder prefill on device (decode 29.6 s → 6.9 s on the 28 s bench clip) | **+3.28% relative WER** on full LibriSpeech dev-clean (0.0269 → 0.0278) | **Validated.** Passes a user-set **+15% mobile-only** WER budget (looser than the desktop gate, which this does not touch). Kill switch: `QWEN_ASR_INT8_PREFILL=0`. |
| `int8-encoder` | Modest encode speedup (~5%: 7.0 s → 6.7 s cool on the bench clip) — **not** a halving | Cumulative WER **not fully measured** | **Provisional.** Kernel is bit-exact and word-match is preserved on the bench clip, but the full-set `+prefill+encoder` WER sweep was aborted, so the cumulative-vs-budget verdict is unverified. Costs ~+86 MB resident. Kill switch: `QWEN_ASR_INT8_ENCODER=0`. |

To roll either back, drop it from the Android feature list in `rust/Cargo.toml`;
the two are independent. The **+15% budget is mobile-only** and never governs the
desktop numerics gate (the features are not compiled there).

## Forced alignment

Word-level timestamp alignment for a known transcript is available through the
Rust library (`qwen_asr::align::forced_align`) for native integration; it is not
yet exposed over the Dart bridge. See the
[crate documentation](../../crates/qwen-asr/README.md).

## Platform support

| Platform | Status |
|----------|--------|
| Android  | Supported — validated on a physical Snapdragon 8 Elite device (see [example README](example/README.md)) |
| iOS      | Supported — **Simulator-validated only**; no physical-device validation yet |
| macOS    | Supported |
| Linux    | Planned |
| Windows  | Planned |

## License

MIT
