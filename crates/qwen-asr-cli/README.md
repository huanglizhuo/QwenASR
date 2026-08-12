# qwen-asr-cli

CLI for [qwen-asr](https://crates.io/crates/qwen-asr): CPU-only Qwen3-ASR speech-to-text in pure Rust.

Current benchmark results are published in the repository README and compare the
CLI's Rust CPU backend with upstream C and MLX-based GPU baselines.

## Install

```bash
cargo install qwen-asr-cli

# Recommended: enable native CPU SIMD tuning
RUSTFLAGS="-C target-cpu=native" cargo install qwen-asr-cli
```

vDSP/Accelerate is auto-enabled on macOS via default features.

## Download Model

```bash
qwen-asr download qwen3-asr-0.6b
```

## Usage

```bash
# Transcribe a file
qwen-asr -d qwen3-asr-0.6b -i audio.wav

# Streaming mode
qwen-asr -d qwen3-asr-0.6b -i audio.wav --stream

# Live capture (macOS)
qwen-asr -d qwen3-asr-0.6b --live --stream --device "BlackHole 2ch"

# VAD live mode (macOS)
qwen-asr -d qwen3-asr-0.6b --live --vad --device "BlackHole 2ch"

# Forced alignment
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world"

# All options
qwen-asr -h
```

See the [project README](https://github.com/huanglizhuo/QwenASR) for full documentation.

## BLAS scheduling (tuning knobs)

Four kernels can either issue **one** `cblas_sgemm` and let the BLAS library
thread it internally, or slice the work across our own thread pool so several
threads call `cblas_sgemm` at once. Slicing is faster on Apple Accelerate and
slower — sometimes to the point of appearing to hang — on OpenBLAS, so the
default follows the backend:

| Variable | Kernel | Default on macOS | Default elsewhere |
|---|---|---|---|
| `QASR_CONV_POOLED` | encoder conv stem | on | off |
| `QASR_LINEAR_POOLED` | `linear` / attention projections | on | off |
| `QASR_ATTN_POOLED` | multi-token (prefill) attention | on | off |
| `QASR_SEG_POOLED` | concurrent segment workers in `-S` mode | on | off |

Set any of them to `1`/`0` (also `true`/`false`, `on`/`off`, `yes`/`no`) to
override. They exist so a single binary can be measured both ways on hardware
the maintainers do not have — if transcription is unexpectedly slow or pegs
every core on your machine, try flipping them and please report the numbers.

```bash
# A/B the conv stem against the non-default policy
QASR_CONV_POOLED=1 qwen-asr -d qwen3-asr-0.6b -i audio.wav --profile
```

## License

MIT
