# q-asr

Pure Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text models. Zero runtime Rust crate dependencies (only `libc`). Ported from [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

Supports the 0.6B and 1.7B model variants with three runtime modes: offline, segmented, and streaming.

## Prerequisites

- **Rust** 1.70+ (stable)
- **BLAS library** (linked automatically):
  - macOS: Apple Accelerate (included with Xcode Command Line Tools)
  - Linux: OpenBLAS (`sudo apt install libopenblas-dev` or equivalent)

## Download Model

Download the Qwen3-ASR-0.6B model files from HuggingFace:

```bash
mkdir -p qwen3-asr-0.6b && cd qwen3-asr-0.6b
for f in model.safetensors vocab.json merges.txt; do
  curl -LO "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/$f"
done
cd ..
```

For the 1.7B variant:

```bash
mkdir -p qwen3-asr-1.7b && cd qwen3-asr-1.7b
for f in model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors vocab.json merges.txt; do
  curl -LO "https://huggingface.co/Qwen/Qwen3-ASR-1.7B/resolve/main/$f"
done
cd ..
```

## Build

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The `target-cpu=native` flag enables NEON (Apple Silicon) or AVX2+FMA (x86_64) SIMD acceleration. The binary is at `target/release/q-asr`.

To build without BLAS (pure Rust fallback for all matmul):

```bash
cargo build --release --no-default-features
```

## Usage

```
q-asr -d <model_dir> (-i <input.wav> | --stdin) [options]
```

### Basic Examples

```bash
# Transcribe a WAV file
./target/release/q-asr -d qwen3-asr-0.6b -i audio.wav

# Pipe audio from stdin
cat audio.wav | ./target/release/q-asr -d qwen3-asr-0.6b --stdin

# Raw s16le 16kHz mono from stdin
ffmpeg -i video.mp4 -f s16le -ar 16000 -ac 1 - | ./target/release/q-asr -d qwen3-asr-0.6b --stdin

# Silent mode (only transcript on stdout, no status on stderr)
./target/release/q-asr -d qwen3-asr-0.6b -i audio.wav --silent
```

### Segmented Mode

Split long audio at silence boundaries for better accuracy and lower memory:

```bash
./target/release/q-asr -d qwen3-asr-0.6b -i long_audio.wav -S 30
```

### Streaming Mode

Process audio in 2-second chunks with incremental output:

```bash
./target/release/q-asr -d qwen3-asr-0.6b -i audio.wav --stream
```

### All Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory (required) | — |
| `-i <file>` | Input WAV file (16-bit PCM, any sample rate) | — |
| `--stdin` | Read audio from stdin (WAV or raw s16le 16kHz mono) | off |
| `-t <n>` | Number of threads | all CPUs |
| `-S <secs>` | Segment target seconds (0 = full-audio decode) | 0 |
| `-W <secs>` | Silence search window for segment splits | 3.0 |
| `--stream` | Streaming mode with chunked rollback | off |
| `--stream-max-new-tokens <n>` | Max tokens per stream step | 32 |
| `--enc-window-sec <secs>` | Encoder attention window (1–8) | 8 |
| `--past-text <yes\|no\|auto>` | Reuse decoded text as context for next segment | auto |
| `--skip-silence` | Drop long silent spans before inference | off |
| `--prompt <text>` | System prompt for biasing | — |
| `--language <lang>` | Force output language (e.g., `en`, `zh`, `ja`) | auto |
| `--debug` | Verbose per-layer output | off |
| `--silent` | No status output, only transcription on stdout | off |

## Testing

Run the full test suite (requires model downloaded to `qwen3-asr-0.6b/` and sample audio in `/tmp/qwen-asr-ref/samples/`):

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release
```

Tests include:
- **Kernel tests** — SIMD vs generic numerical correctness, norms, activations, vector ops
- **Audio tests** — WAV loading, mel spectrogram shape and value ranges, silence compaction
- **Tokenizer tests** — BPE encode/decode round-trip, vocabulary loading
- **Regression tests** — End-to-end transcription accuracy (offline, segmented, streaming)

To run only unit-level tests without a model:

```bash
cargo test --release --test kernels --test audio
```

## Project Structure

```
src/
  main.rs          CLI entry point
  lib.rs           Library re-exports
  config.rs        Model config, variant detection (0.6B vs 1.7B)
  safetensors.rs   Mmap-based weight loader (multi-shard)
  audio.rs         WAV decode, resample, mel spectrogram
  tokenizer.rs     GPT-2 byte-level BPE
  encoder.rs       Conv2D stem + windowed transformer
  decoder.rs       28-layer GQA decoder + KV cache
  context.rs       Top-level state (QwenCtx)
  transcribe.rs    Offline / segmented / streaming orchestration
  kernels/
    mod.rs         BLAS bindings, thread pool, dispatch
    generic.rs     Portable f32 fallbacks
    neon.rs        ARM NEON SIMD (aarch64)
    avx.rs         x86 AVX2+FMA SIMD
tests/
  kernels.rs       Kernel numerical correctness
  audio.rs         Audio pipeline verification
  tokenizer.rs     BPE round-trip tests
  regression.rs    End-to-end transcription tests
```

## Performance

Benchmarks on Apple M-series (10 cores), Qwen3-ASR-0.6B:

| Mode | Audio Length | Inference Time | Realtime Factor |
|------|-------------|----------------|-----------------|
| Offline | 11s | 1.8s | 6.2x |
| Segmented (`-S 30`) | 45s | 4.6s | 9.8x |
| Segmented (`-S 30`) | 89s | 17.4s | 5.1x |
| Streaming | 45s | 18.0s | 2.5x |

## License

Same license as the upstream [antirez/qwen-asr](https://github.com/antirez/qwen-asr) project.
