# qwen-asr

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

### macOS (CLI)

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The `target-cpu=native` flag enables NEON (Apple Silicon) or AVX2+FMA (x86_64) SIMD acceleration. The binary is at `target/release/qwen-asr`.

For additional performance on Apple Silicon, enable vDSP (uses the AMX coprocessor via Accelerate):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features vdsp
```

### Linux (CLI)

Install OpenBLAS first:

```bash
# Debian/Ubuntu
sudo apt install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel
```

Then build:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Without BLAS

To build with pure Rust fallback for all matmul (no BLAS dependency):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features
```

### iOS (Static Library)

Produces a static library (`libqwen_asr.a`) and C header for integration via the C-FFI API.

```bash
# Device (arm64)
cargo build --release --target aarch64-apple-ios --features ios

# Simulator (arm64 Apple Silicon host)
cargo build --release --target aarch64-apple-ios-sim --features ios
```

The static library is at `target/aarch64-apple-ios/release/libqwen_asr.a`. Link it into your Xcode project along with the Accelerate framework. The C-FFI API is defined in `src/c_api.rs`:

```c
// Load model, returns opaque handle
void *qwen_asr_load_model(const char *model_dir, int n_threads, int verbosity);

// Transcribe a WAV file on disk
const char *qwen_asr_transcribe_file(void *engine, const char *wav_path);

// Transcribe raw PCM float samples (16kHz mono)
const char *qwen_asr_transcribe_pcm(void *engine, const float *samples, int n_samples);

// Transcribe in-memory WAV data
const char *qwen_asr_transcribe_wav_buffer(void *engine, const uint8_t *data, int len);

// Configuration
void qwen_asr_set_segment_sec(void *engine, float sec);
void qwen_asr_set_language(void *engine, const char *lang);

// Free returned strings and engine
void qwen_asr_free_string(const char *s);
void qwen_asr_free(void *engine);
```

### Android (Shared Library)

Requires the Android NDK. Install [cargo-ndk](https://github.com/nicohman/cargo-ndk) for convenience:

```bash
cargo install cargo-ndk
rustup target add aarch64-linux-android
```

Build the shared library:

```bash
cargo ndk -t arm64-v8a build --release --features android
```

The `.so` is at `target/aarch64-linux-android/release/libqwen_asr.so`. The JNI API maps to the Java class `com.qwenasr.QAsrEngine`:

```java
public class QAsrEngine {
    static { System.loadLibrary("qwen_asr"); }
    public native boolean loadModel(String modelDir, int nThreads);
    public native String transcribePcm(float[] samples);
    public native String transcribeWav(byte[] wavData);
    public native void setSegmentSec(float sec);
    public native void setLanguage(String language);
    public native void free();
}
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `blas` (default) | Link BLAS (Accelerate on macOS/iOS, OpenBLAS on Linux) |
| `vdsp` | Use Accelerate vDSP/vForce for AMX acceleration (macOS/iOS) |
| `ios` | Enable C-FFI API for iOS integration |
| `android` | Enable JNI API for Android integration |

## Usage

```
qwen-asr -d <model_dir> (-i <input.wav> | --stdin) [options]
```

### Basic Examples

```bash
# Transcribe a WAV file
./target/release/qwen-asr -d qwen3-asr-0.6b -i audio.wav

# Pipe audio from stdin
cat audio.wav | ./target/release/qwen-asr -d qwen3-asr-0.6b --stdin

# Raw s16le 16kHz mono from stdin
ffmpeg -i video.mp4 -f s16le -ar 16000 -ac 1 - | ./target/release/qwen-asr -d qwen3-asr-0.6b --stdin

# Silent mode (only transcript on stdout, no status on stderr)
./target/release/qwen-asr -d qwen3-asr-0.6b -i audio.wav --silent
```

### Segmented Mode

Split long audio at silence boundaries for better accuracy and lower memory:

```bash
./target/release/qwen-asr -d qwen3-asr-0.6b -i long_audio.wav -S 30
```

### Streaming Mode

Process audio in 2-second chunks with incremental output:

```bash
./target/release/qwen-asr -d qwen3-asr-0.6b -i audio.wav --stream
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
| `--profile` | Print per-operation timing breakdown | off |
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
  c_api.rs         C-FFI API for iOS integration (feature: ios)
  jni_api.rs       JNI API for Android integration (feature: android)
  kernels/
    mod.rs         BLAS/vDSP bindings, thread pool, profiling, dispatch
    generic.rs     Portable f32 fallbacks
    neon.rs        ARM NEON SIMD (aarch64)
    avx.rs         x86 AVX2+FMA SIMD
.cargo/
  config.toml      Cross-compilation targets (iOS, Android)
tests/
  kernels.rs       Kernel numerical correctness
  audio.rs         Audio pipeline verification
  tokenizer.rs     BPE round-trip tests
  regression.rs    End-to-end transcription tests
```

## Benchmarking

A shell-based benchmark suite lives in `bench/` for comparing performance and accuracy across code changes or against other binaries (e.g. the C reference).

```bash
# Run all modes (offline, segmented, streaming) against bench/samples/audio.wav
bench/run.sh --label before-optimization

# Rebuild after changes, run again
RUSTFLAGS="-C target-cpu=native" cargo build --release
bench/run.sh --label after-optimization

# Compare the two runs
bench/compare.sh before-optimization after-optimization
```

Results are saved as JSON in `bench/results/<label>/`. Each result file contains timing breakdowns, per-op profile data, and WER/CER accuracy against the reference transcript.

Options for `bench/run.sh`:

| Option | Description | Default |
|--------|-------------|---------|
| `--binary PATH` | Path to ASR binary | `./target/release/qwen-asr` |
| `--model-dir DIR` | Model directory | `qwen3-asr-0.6b` |
| `--samples-dir DIR` | Audio samples directory | `bench/samples` |
| `--label NAME` | Label for this run | git short rev or timestamp |
| `--modes LIST` | Comma-separated modes | `offline,segmented,streaming` |
| `--threads N` | Thread count | all CPUs |
| `--runs N` | Repeat each test N times, keep best | 1 |

To compare against the C reference:

```bash
bench/run.sh --binary /path/to/qwen_asr --label c-reference
bench/compare.sh c-reference after-optimization
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
