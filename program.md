# QwenASR Autoresearch Program

> Autonomous optimization of the QwenASR Rust inference engine using the autoresearch pattern.
> Human writes this file. Agent executes the loop.

## Goal

Maximize inference performance (lower latency, higher realtime factor) and/or improve transcription accuracy (lower WER/CER) of the QwenASR Rust inference engine on the current hardware, without breaking correctness.

## Setup Phase (one-time, confirm with human)

1. Create the branch: `git checkout -b autoresearch/<tag>` from current `main`.
2. Read the in-scope files. The repo is a Rust workspace. Read these for full context:
   - `README.md` — repository context, build instructions, feature flags
   - `Cargo.toml` — workspace config, dependencies, features
   - `crates/` — the core library crate(s)
   - `src/` — CLI entry point and all core modules:
     - `kernels/` — SIMD kernels (generic, NEON, AVX), BLAS bindings, thread pool ← **primary optimization target**
     - `encoder.rs` — Conv2D stem + windowed transformer
     - `decoder.rs` — GQA decoder + KV cache
     - `audio.rs` — WAV decode, resample, mel spectrogram
     - `transcribe.rs` — offline / segmented / streaming orchestration
     - `config.rs`, `context.rs`, `safetensors.rs`, `tokenizer.rs`
   - `bench/` — benchmark scripts and comparison tools
   - `tests/` — kernel, audio, tokenizer, regression tests
3. Verify model exists: Check that `qwen3-asr-0.6b/` (or `qwen3-asr-1.7b/`) contains model files. If not, tell the human to download them per README.
4. Verify benchmark audio exists: Check `bench/samples/` for WAV files and `audio.wav` in root. If missing, tell human.
5. Establish baseline:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 | tail -5
   bench/run.sh --label baseline --runs 3
   ```
6. Initialize `results.tsv` with header:
   ```
   experiment	description	build_ok	test_ok	offline_time_ms	offline_rtf	segmented_time_ms	segmented_rtf	streaming_time_ms	streaming_rtf	status
   ```
   Record baseline results as experiment 0.
7. Confirm setup looks good. Once you get human confirmation, kick off experimentation.

## Experiment Loop

Repeat indefinitely:

### 1. Pick an idea

Choose ONE focused change per experiment. Ideas to explore (not exhaustive):

**Kernel / SIMD optimizations:**
- Vectorize hot loops that are still using generic.rs fallbacks
- Improve NEON/AVX kernel implementations (fused ops, reduce branching)
- Optimize matmul tiling, cache blocking, prefetch hints
- Try different BLAS call patterns or batch sizes
- Reduce unnecessary memory allocations in hot paths

**Decoder / Encoder optimizations:**
- KV cache memory layout (contiguous vs strided, pre-allocation)
- Attention computation optimizations (fused softmax, flash-attention-style chunking)
- Layer fusion opportunities (combine adjacent operations)
- Reduce redundant computation in streaming mode rollback

**Audio pipeline:**
- Mel spectrogram computation optimization
- FFT/windowing optimizations
- Silence detection efficiency

**Memory & allocation:**
- Reduce heap allocations in the inference hot path
- Use arena allocators or pre-allocated buffers
- Optimize tensor layout for cache locality

**Architecture-level:**
- Parallelism tuning (thread count, work distribution)
- Batch processing of encoder windows
- Async I/O for audio loading

### 2. Implement the change

Edit the relevant Rust source file(s). Keep changes minimal and focused. Write a clear git commit message describing the hypothesis.

```bash
git add -A && git commit -m "experiment: <brief description of change>"
```

### 3. Build

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 > build.log
```

If build fails, read `tail -30 build.log`, attempt a fix. If you can't fix after 2 attempts, revert and move on.

### 4. Run tests (quick sanity check)

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release --test kernels --test audio 2>&1 > test.log
tail -5 test.log
```

If tests fail, the change broke correctness. Revert immediately.

### 5. Benchmark

```bash
bench/run.sh --label "exp-$(date +%s)" --runs 3 > bench.log 2>&1
```

If bench/run.sh is not available or doesn't work, fall back to direct timing:

```bash
# Offline mode benchmark
time ./target/release/qwen-asr -d qwen3-asr-0.6b -i audio.wav --silent > /dev/null 2>&1

# If segmented mode test audio exists:
time ./target/release/qwen-asr -d qwen3-asr-0.6b -i bench/samples/audio.wav -S 30 --silent > /dev/null 2>&1
```

Run each 3 times, take the best (lowest) time.

### 6. Evaluate

Extract timing results. Compare against the current best baseline.

**Keep criteria (ALL must hold):**
- Build succeeds
- Tests pass
- Inference time improved (even by a small margin) OR accuracy improved without significant speed regression
- No correctness regression (spot-check transcript output hasn't degraded)

### 7. Keep or revert

**If improved:**
```bash
# Record in results.tsv
echo "<exp>\t<description>\tyes\tyes\t<offline_ms>\t<offline_rtf>\t<seg_ms>\t<seg_rtf>\t<stream_ms>\t<stream_rtf>\tkept" >> results.tsv
# This commit stays. It becomes the new baseline.
```

**If not improved or regressed:**
```bash
echo "<exp>\t<description>\tyes\tyes\t<offline_ms>\t<offline_rtf>\t<seg_ms>\t<seg_rtf>\t<stream_ms>\t<stream_rtf>\treverted" >> results.tsv
git reset --hard HEAD~1
```

### 8. Repeat

Go back to step 1. Try a different idea. Learn from what worked and what didn't.

## Rules

- **Do NOT modify test files** to make tests pass. If tests fail, the code change is wrong.
- **Do NOT modify bench/ scripts** unless they are genuinely broken.
- **One idea per experiment.** Compound changes make it impossible to attribute results.
- **Always build in release mode** with `--release` and `target-cpu=native`.
- **Redirect output.** Never let build/test/bench output flood your context. Use `> file.log 2>&1`.
- **Be bold but reversible.** Try architectural changes, not just parameter tweaks. Git makes everything reversible.
- **Track everything** in results.tsv. Do not commit results.tsv (keep it untracked).
- **If stuck after 3 failed experiments in a row**, step back and re-read the source to find a new angle.
- **Unsafe Rust:** You may use `unsafe` blocks for performance-critical SIMD code, but be extra careful. Run tests after every unsafe change.

## Context for the Agent

This is a pure Rust, CPU-only ASR inference engine. There is no GPU, no CUDA, no Python in the hot path. Performance gains come from:
1. Better utilization of CPU SIMD (NEON on ARM, AVX2+FMA on x86)
2. Better memory access patterns (cache-friendly layouts)
3. Reducing allocations and copies
4. Algorithmic improvements in the attention/matmul/FFT paths
5. Better thread utilization

The model weights are fixed (loaded from safetensors). You are optimizing the inference code, not training a model. The "loss function" here is wall-clock inference time for a given audio input.
