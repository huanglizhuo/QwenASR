# qwen-asr

Pure Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text. Zero runtime Rust crate deps (only `libc`). Ported from [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

Supports 0.6B and 1.7B models with five modes: offline, segmented, streaming, live capture, and VAD live.

## Quick Start (Pre-built Binary)

Download a pre-built binary from [GitHub Releases](https://github.com/huanglizhuo/QwenASR/releases) — no Rust toolchain needed.

```bash
# Install via cargo (alternative)
cargo install qwen-asr-cli

# Download model
qwen-asr download qwen3-asr-0.6b

# Transcribe
qwen-asr -d qwen3-asr-0.6b -i audio.wav
```

## OpenClaw Skill

One-command install for [OpenClaw](https://github.com/anthropics/openclaw) users:

```bash
bash skills/qwen-asr/scripts/install.sh
```

This downloads the binary + model to `~/.openclaw/tools/qwen-asr/`. Then transcribe:

```bash
bash skills/qwen-asr/scripts/transcribe.sh audio.wav
```

## Usage

```
qwen-asr -d <model_dir> (-i <file> | --stdin | --live) [options]
```

```bash
# Basic transcription
qwen-asr -d qwen3-asr-0.6b -i audio.wav

# Silent mode (transcript only, no progress)
qwen-asr -d qwen3-asr-0.6b -i audio.wav --silent

# Pipe from stdin
cat audio.wav | qwen-asr -d qwen3-asr-0.6b --stdin

# Raw PCM from ffmpeg
ffmpeg -i video.mp4 -f s16le -ar 16000 -ac 1 - | qwen-asr -d qwen3-asr-0.6b --stdin

# Segmented mode (long audio)
qwen-asr -d qwen3-asr-0.6b -i long.wav -S 30

# Streaming mode
qwen-asr -d qwen3-asr-0.6b -i audio.wav --stream

# Live capture (macOS)
qwen-asr -d qwen3-asr-0.6b --live --device "BlackHole 2ch"

# VAD live mode (macOS)
qwen-asr -d qwen3-asr-0.6b --live --vad --device "BlackHole 2ch"

# Forced alignment
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world" --align-language English
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory (required) | — |
| `-i <file>` | Input WAV file | — |
| `--stdin` | Read audio from stdin (WAV or raw s16le 16kHz) | off |
| `--live` | Live capture from audio device (macOS) | off |
| `--device <name>` | Input device for live capture | system default |
| `--list-devices` | List audio input devices | — |
| `--vad` | VAD live mode | off |
| `-t <n>` | Thread count | all CPUs |
| `-S <secs>` | Segment target seconds | 0 (full) |
| `--stream` | Streaming mode | off |
| `--stream-chunk-sec <s>` | Chunk size for streaming | 2.0 |
| `--language <lang>` | Force output language (`en`, `zh`, `ja`, ...) | auto |
| `--silent` | Transcript only, no status output | off |
| `--profile` | Print timing breakdown | off |

## Build from Source

**Always build in release mode.** Debug builds are 10-50x slower.

### macOS

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Linux

```bash
sudo apt install libopenblas-dev   # Debian/Ubuntu
sudo dnf install openblas-devel    # Fedora/RHEL

RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Without BLAS

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features
```

### iOS / Android

```bash
# iOS (static library + C-FFI)
cargo build --release --target aarch64-apple-ios --features ios

# Android (shared library + JNI)
cargo ndk -t arm64-v8a build --release --features android
```

See `src/c_api.rs` for C-FFI and `src/jni_api.rs` for JNI interfaces.

### Feature Flags

| Feature | Description |
|---------|-------------|
| `blas` (default) | BLAS linking (Accelerate on macOS, OpenBLAS on Linux) |
| `vdsp` | Accelerate vDSP/vForce for AMX (macOS) |
| `ios` | C-FFI API |
| `android` | JNI API |

## Performance

**Hardware:** Apple M1 Pro (10 cores), 32 GB RAM
**Model:** Qwen3-ASR-0.6B, **Audio:** 28.2s sample, 3 runs (best)

| Mode | Inference | Realtime Factor | Encode | Decode | Tokens/s |
|------|-----------|-----------------|--------|--------|----------|
| Offline | 2.87s | 9.8x | 744ms | 2124ms | 15.7 |
| Segmented (`-S 30`) | 2.86s | 9.9x | 712ms | 2145ms | 15.8 |
| Streaming | 9.29s | 3.0x | 2195ms | 7088ms | 4.9 |

## Testing

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release

# Unit tests only (no model needed)
cargo test --release --test kernels --test audio
```

## Acknowledgments

This project is a Rust port inspired by [antirez/qwen-asr](https://github.com/antirez/qwen-asr), a pure C implementation of Qwen3-ASR inference by Salvatore Sanfilippo (antirez). The architecture, algorithms, and approach are derived from that work.

## License

Same license as [antirez/qwen-asr](https://github.com/antirez/qwen-asr).
