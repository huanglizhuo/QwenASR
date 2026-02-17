# Changelog

## [0.2.0] - 2026-02-15

### Performance

- **Reusable BF16→F32 scratch buffer** — Pre-allocated scratch in `DecoderBuffers` eliminates ~140 heap allocations per prefill pass
- **SIMD BF16→F32 bulk conversion** — NEON (`vshll_n_u16`) and AVX2 (`_mm256_cvtepu16_epi32`) paths for 4-8x faster weight conversion
- **Threaded encoder attention** — `bidirectional_attention` parallelized across heads via thread pool for near-linear scaling
- **Cached mel filter bank** — `OnceLock`-based lazy initialization avoids redundant computation in streaming mode
- **SIMD activation functions** — Vectorized `rms_norm`, `gelu`, and `swiglu` with fast polynomial exp approximation (NEON + AVX2)
- **Encoder buffer reuse** — New `EncoderBuffers` struct with persistent scratch avoids per-call allocations in encoder forward pass
- **vDSP integration** (macOS, `--features vdsp`) — `vDSP_dotpr`, `vDSP_vsmul`, `vDSP_vsma`, `vvexpf` leverage Apple AMX coprocessor

### Features

- **Built-in profiling** — `--profile` flag prints per-operation timing breakdown (call count, total/avg time)
- **iOS support** — Static library target with C-FFI API (`src/c_api.rs`): `qwen_asr_load_model`, `qwen_asr_transcribe_file`, `qwen_asr_transcribe_pcm`, `qwen_asr_free`
- **Android support** — Shared library target with JNI API (`src/jni_api.rs`) for `com.qwenasr.QAsrEngine` Java class
- **Feature flags** — `blas` (default), `vdsp`, `ios`, `android` for platform-specific builds
- **Cross-compilation config** — `.cargo/config.toml` with tuned CPU targets for iOS (`apple-a14`) and Android (`cortex-a76`)

### Changed

- Library crate renamed to `qwen_asr` (was `q-asr`) for valid Rust identifier in imports
- Library target now produces `lib`, `staticlib`, and `cdylib` outputs
- Thread pool workers recover from poisoned mutex instead of panicking
- Regression tests serialized via `Mutex` to prevent thread pool race conditions
- README updated with per-platform build instructions (macOS, Linux, iOS, Android)

## [0.1.0] - 2026-02-15

Initial release — pure Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- CPU-only Qwen3-ASR inference (0.6B and 1.7B models)
- Three runtime modes: offline, segmented (`-S`), streaming (`--stream`)
- NEON SIMD (aarch64) and AVX2+FMA (x86_64) acceleration
- BLAS via Accelerate (macOS) / OpenBLAS (Linux)
- Zero runtime Rust crate dependencies (only `libc`)
- 22 tests (kernels, audio, tokenizer, regression)
