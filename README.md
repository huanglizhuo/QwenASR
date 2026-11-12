# qwen-asr

[![OctoCounts](https://api.octocounts.com/badge/huanglizhuo/QwenASR/branch/main)](https://octocounts.com/?q=https%3A%2F%2Fgithub.com%2Fhuanglizhuo%2FQwenASR&ref=main)

A fast, pure-Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text. It has no heavyweight runtime dependency—only `libc`—and is optimized for low-latency inference on Apple Silicon.

Supports the 0.6B and 1.7B models, offline and streaming transcription, live capture with VAD, subtitles, structured JSON, and forced alignment.

## Performance

On an Apple M5 Pro, the current 0.6B CPU build transcribes the **28.16-second**
benchmark clip in **529 ms inference / 634 ms end to end**
(medians of 10 standalone runs): **53.23× realtime** by inference time.

| Implementation | Median inference | Median process wall | Inference throughput |
|---|---:|---:|---:|
| qwen-asr (current CPU) | 529.0 ms | 633.9 ms | 53.23× |
| mlx-audio Python (GPU) | 659.5 ms | 1,462.6 ms | 42.70× |
| second-state MLX (GPU) | 1,295.6 ms | 1,464.1 ms | 21.74× |
| qwen-asr first port (CPU) | 1,591.5 ms | 1,908.8 ms | 17.69× |
| pure C upstream (CPU) | 1,603.5 ms | 1,858.6 ms | 17.56× |

<p float="left">
  <img src="docs/benchmarks/charts/benchmark-unified-latency.png" width="48%" alt="Measured inference latency comparison" />
  <img src="docs/benchmarks/charts/benchmark-unified-rtf.png" width="48%" alt="Measured inference throughput comparison" />
</p>

> Run started 2026-11-12; Apple M5 Pro, 15 cores, 48 GiB; 0.6B model family,
> each implementation's default configuration. Current build is commit
> `b6ed4ea4` (Round 16), binary `c86d2194333b…`. Desktop processes kept
> running, but the load was lighter than on November 6 and every implementation
> measured faster. Native inference boundaries and quantization differ between
> implementations; wall time includes process startup/load/exit.
> See [versions, methodology and full comparison](docs/benchmarks/comparison.md).

Long-audio results (three measured runs per side, alternating against main `6a88ff62`):

| Audio / mode | Median inference | Median process wall |
|---|---:|---:|
| 120.705 s offline | 4.04 s | 4.14 s |
| 120.705 s segmented (-S30) | 2.45 s | 2.56 s |
| 602.465 s offline | 48.16 s | 48.32 s |
| 602.465 s segmented (-S30) | 9.74 s | 9.91 s |

Against main, the 602 s segmented run was 10.8% faster. The other three cases
stayed within run-to-run spread. The 602 s single pass is especially
load-sensitive: 46.8–54.5 s wall here, versus 51.9–57.4 s for main.

Round 16 changes no output. On edge-length clips, encoder outputs and prefill
KV tensors hash identically to main, and transcripts and forced-alignment JSON
match byte for byte. All 2,703 LibriSpeech dev-clean transcripts match the
November 6 full run byte for byte, so offline WER stays **2.7591%**.
[Round 16 run record](docs/benchmarks/2026-11-12/README.md).

qwen-asr 0.11.2 fixed the final file `--stream` decode, which previously
stopped after 32 tokens. November 7 results, measured before Round 16 (three
measured runs per side on a loaded M5 Pro, 0.6B model, 12 threads):

| File `--stream` | WER before → after | Fixed median inference | Fixed median process wall |
|---|---:|---:|---:|
| 120.705 s | 91.7857% → **2.8571%** | 4.29 s | 4.41 s |
| 602.465 s | 98.3780% → **2.8914%** | 54.21 s | 54.39 s |

The old timings measured partial transcripts. A subsequent fix removes
destructive intermediate file-stream resets and reuses unchanged decode input.
File mode retains its deferred output defaults and complete context through EOF;
memory and decoding work grow with recording length.
[EOF fix](docs/benchmarks/2026-11-07-stream/README.md) and
[small-chunk fix, coverage checks and timings](docs/benchmarks/2026-11-07-stream-reset/README.md).

The small-chunk follow-up uses **0.5 s chunks**, with three measured runs per
side on the same loaded desktop. All 22 / 102 source utterances retain matching
text; the longest unmatched reference spans are only 2 / 3 words.
Here, “before” includes the EOF-budget fix but retains intermediate resets.

| File `--stream`, 0.5 s chunk | WER before → after | Fixed median inference | Fixed median process wall |
|---|---:|---:|---:|
| 120.705 s | 58.2143% → **2.8571%** | 7.55 s | 7.68 s |
| 602.465 s | 91.7489% → **2.6093%** | 220.90 s | 221.13 s |

The longer complete context makes 0.5-second file decoding more expensive
than the default configuration; these timings should be compared within their
respective chunk settings and output coverage.
[All measurements and per-utterance comparison](docs/benchmarks/results.md).

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

- Hand-tuned NEON, SME, Accelerate, and AMX-aware kernels for Apple Silicon
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
| `--stream-chunk-sec <s>` | Chunk size for streaming | 8.0 (2.0 for stdin) |
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
./bench/run.sh --samples-dir bench/samples/audio.wav --threads 12 --label current --runs 10

# WER benchmark (full 2,703-item LibriSpeech dev-clean offline)
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-full \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results \
  --label current-offline-full \
  --mode offline

# Cross-implementation comparison (30–60 min)
./bench/benchmark-all.sh --runs 10 --current-binary target/release/qwen-asr
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
