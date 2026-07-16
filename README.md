# qwen-asr

[![OctoCounts](https://api.octocounts.com/badge/huanglizhuo/QwenASR/branch/main)](https://octocounts.com/?q=https%3A%2F%2Fgithub.com%2Fhuanglizhuo%2FQwenASR&ref=main)

A fast, pure-Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text. It has no heavyweight runtime dependency—only `libc`—and is optimized for low-latency inference on Apple Silicon.

Supports the 0.6B and 1.7B models, offline and streaming transcription, live capture with VAD, subtitles, structured JSON, and forced alignment.

## Performance

On an Apple M5 Pro, qwen-asr transcribes the 28.2-second benchmark clip in **613 ms**—**46× faster than realtime**. In the same 10-run comparison, this CPU-only Rust implementation has the lowest median inference latency of all five implementations tested:

- **1.12× faster** than `mlx-audio` on the GPU
- **2.31× faster** than second-state MLX on the GPU
- **2.71× faster** than the upstream pure-C implementation
- **2.77× faster** than the first qwen-asr Rust port

| Implementation | Median inference | Realtime factor |
|---|---:|---:|
| **qwen-asr (CPU, latest)** | **613 ms** | **46.00×** |
| mlx-audio Python MLX (GPU) | 688 ms | 40.94× |
| second-state MLX (GPU) | 1,414 ms | 19.91× |
| pure C upstream (CPU) | 1,660 ms | 16.96× |
| qwen-asr first port (CPU) | 1,698 ms | 16.61× |

<p float="left">
  <img src="docs/benchmarks/charts/benchmark-unified-latency.png" width="48%" alt="Latency comparison" />
  <img src="docs/benchmarks/charts/benchmark-unified-rtf.png" width="48%" alt="Realtime factor comparison" />
</p>

> Apple M5 Pro, 15 cores, 48 GB RAM; same 0.6B model and 28.2 s audio; 10 standalone runs per implementation; median inference latency; each implementation uses its shipped default configuration. Results generated at `d141bca4` in `bench/compare-results/20260716T070644Z`. See the [methodology and full results](docs/benchmarks/comparison.md).

## Quick Start

```bash
# Install
cargo install qwen-asr-cli

# Download model
qwen-asr download qwen3-asr-0.6b

# Transcribe
qwen-asr -d qwen3-asr-0.6b -i audio.wav
```

Or download a pre-built binary from [GitHub Releases](https://github.com/huanglizhuo/QwenASR/releases).

## Why qwen-asr is fast

- Hand-tuned NEON, Accelerate, and AMX-aware kernels for Apple Silicon
- Quantized decode weights and batched decoding paths that reduce memory traffic
- Dynamic scheduling across performance and efficiency cores
- No tensor framework, Python runtime, GPU dispatch, or server process

See the [optimization catalog](docs/optimizations/overview.md) for implementation details and the [research log](docs/research/) for measured experiments.

## Usage

```bash
qwen-asr -d qwen3-asr-0.6b -i audio.wav              # basic
qwen-asr -d qwen3-asr-0.6b -i audio.wav --silent      # transcript only
cat audio.wav | qwen-asr -d qwen3-asr-0.6b --stdin     # pipe from stdin
qwen-asr -d qwen3-asr-0.6b -i long.wav -S 30           # segmented
qwen-asr -d qwen3-asr-0.6b -i audio.wav --stream       # streaming
qwen-asr -d qwen3-asr-0.6b -i audio.wav --srt           # SRT subtitles
qwen-asr -d qwen3-asr-0.6b -i audio.wav --json out.json # structured JSON
qwen-asr -d qwen3-asr-0.6b -i audio.wav --aligner-dir qwen3-aligner-0.6b --srt out.srt --vtt out.vtt
qwen-asr -d qwen3-asr-0.6b --live --device "BlackHole 2ch"         # live capture (macOS)
qwen-asr -d qwen3-asr-0.6b --live --vad --device "BlackHole 2ch"   # VAD live
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world" --align-language English  # alignment
```

<details>
<summary>All options</summary>

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory (required) | — |
| `-i <file>` | Input WAV file | — |
| `--stdin` | Read audio from stdin (WAV or raw s16le 16kHz) | off |
| `--live` | Live capture from audio device (macOS) | off |
| `--device <name>` | Input device for live capture | system default |
| `--list-devices` | List audio input devices | — |
| `--vad` | VAD live mode | off |
| `-t <n>` | Thread count | performance cores |
| `-S <secs>` | Segment target seconds | 0 (full) |
| `--stream` | Streaming mode | off |
| `--stream-chunk-sec <s>` | Chunk size for streaming | 2.0 |
| `--language <lang>` | Force output language (`en`, `zh`, `ja`, ...) | auto |
| `--srt [path]` | Write SRT subtitles | `<input>.srt` |
| `--vtt [path]` | Write WebVTT subtitles | `<input>.vtt` |
| `--json [path]` | Write structured JSON; stdout when path is omitted | off |
| `--aligner-dir <dir>` | ForcedAligner model for word timestamps and sentence-level subtitles | off |
| `--silent` | Transcript only, no status output | off |
| `--profile` | Print timing breakdown | off |

</details>

## Output Formats

By default, `qwen-asr` prints plain text to stdout. Output flags are opt-in and can be combined; the audio is transcribed once and shared across requested outputs.

```bash
qwen-asr -d qwen3-asr-0.6b -i audio.wav --srt
qwen-asr -d qwen3-asr-0.6b -i audio.wav --vtt captions.vtt
qwen-asr -d qwen3-asr-0.6b -i audio.wav --json transcript.json
qwen-asr -d qwen3-asr-0.6b -i audio.wav --json
```

SRT and VTT use segment-level timestamps unless an aligner model is supplied. For sentence-level subtitle cues and JSON word timestamps, pass a Qwen3-ForcedAligner model:

```bash
qwen-asr -d qwen3-asr-0.6b -i audio.wav \
  --aligner-dir qwen3-aligner-0.6b \
  --srt captions.srt --vtt captions.vtt --json transcript.json
```

The JSON output has this shape:

```json
{
  "transcription_info": {
    "language": "en",
    "duration": 123.456
  },
  "text": "Full transcript text",
  "word_count": 42,
  "segments": [
    {
      "start": 0.000,
      "end": 8.750,
      "text": "Segment transcript text",
      "words": [
        { "word": "Shenyang,", "start": 1.120, "end": 1.440 }
      ],
      "word_count": 3
    }
  ],
  "vtt": "WEBVTT\n\n1\n00:00:01.120 --> 00:00:08.750\nSegment transcript text\n\n"
}
```

## Build

**Always use release mode.** Debug builds are 10–50× slower.

```bash
# macOS
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Linux
sudo apt install libopenblas-dev   # Debian/Ubuntu
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Without BLAS
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features

# iOS (static library + C-FFI)
cargo build --release --target aarch64-apple-ios --features ios

# Android (shared library + JNI)
cargo ndk -t arm64-v8a build --release --features android
```

| Feature | Description |
|---------|-------------|
| `blas` (default) | BLAS linking (Accelerate on macOS, OpenBLAS on Linux) |
| `vdsp` | Accelerate vDSP/vForce for AMX (macOS) |
| `ios` | C-FFI API |
| `android` | JNI API |

## Reproducing Benchmarks

```bash
# Speed benchmark
./bench/run.sh --label current --runs 10

# WER benchmark (100-file LibriSpeech offline)
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 --mode offline

# Cross-implementation comparison (30–60 min)
./bench/benchmark-all.sh --runs 10
```

See [`docs/benchmarks/`](docs/benchmarks/) for full details.

## Documentation

- [Benchmark methodology and results](docs/benchmarks/)
- [Optimization catalog](docs/optimizations/overview.md)
- [Research and experiment history](docs/research/)
- [Automated release process](RELEASE_PROCESS.md)

## OpenClaw Skill

One-command install for [OpenClaw](https://github.com/anthropics/openclaw) users:

```bash
bash skills/qwen-asr/scripts/install.sh
bash skills/qwen-asr/scripts/transcribe.sh audio.wav
```

## Acknowledgments

Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr), a pure C implementation of Qwen3-ASR inference by antirez.

## License

Same license as [antirez/qwen-asr](https://github.com/antirez/qwen-asr).
