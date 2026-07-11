# qwen-asr

[![OctoCounts](https://api.octocounts.com/badge/huanglizhuo/QwenASR/branch/main)](https://octocounts.com/?q=https%3A%2F%2Fgithub.com%2Fhuanglizhuo%2FQwenASR&ref=main)

A **blazing fast**, pure Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text. Zero heavy runtime dependencies (only `libc`). Ported from [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

Supports 0.6B and 1.7B models with offline, segmented, streaming, live capture, VAD live, and forced alignment modes.

## Performance

On an Apple M5 Pro, qwen-asr transcribes the standard 28.2-second benchmark clip in **563 ms** in the dedicated local benchmark — about **50× faster than realtime** — while producing the full transcript. In the latest full cross-implementation comparison, the pure-CPU qwen-asr is the **fastest implementation overall**, ahead of both GPU baselines (`mlx-audio` MLX and second-state MLX) and 2.5× faster than the upstream C implementation.

| Implementation | Median inference | Realtime factor |
|---|---:|---:|
| **qwen-asr (latest)** | **658 ms** | **42.86×** |
| mlx-audio Python MLX | 687 ms | 40.97× |
| second-state/qwen3_asr_rs MLX GPU | 1,388 ms | 20.29× |
| qwen-asr (first Rust port) | 1,649 ms | 17.10× |
| pure C upstream | 1,662 ms | 16.94× |

<p float="left">
  <img src="docs/benchmarks/charts/benchmark-unified-latency.png" width="48%" alt="Latency comparison" />
  <img src="docs/benchmarks/charts/benchmark-unified-rtf.png" width="48%" alt="Realtime factor comparison" />
</p>

> Benchmarked on the same 28.2 s sample with 10 runs each. The table shows the latest full cross-implementation run (`bench/compare-results/20260711T145612Z`) at qwen-asr `d241af9b` with `--threads 15` for every implementation; the dedicated speed benchmark (binary default threads) reports 563 ms / 50.09×. All rows produce the full transcript. See [`docs/benchmarks/comparison.md`](docs/benchmarks/comparison.md) for full details and reproduction steps.

## Documentation

- [`docs/benchmarks/`](docs/benchmarks/) — benchmark methodology, latest results, and reproduction instructions
- [`docs/optimizations/overview.md`](docs/optimizations/overview.md) — catalog of implemented performance optimizations
- [`docs/research/`](docs/research/) — historical autoresearch experiment logs and protocols

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
