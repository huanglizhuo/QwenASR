# Cross-Implementation Comparison

Apples-to-apples comparison of qwen-asr against the upstream pure C implementation and MLX-based baselines on the same audio and model.

## Methodology

- Offline benchmark on `bench/samples/audio.wav` (28.2 s, mono 16 kHz)
- Model: `qwen3-asr-0.6b`
- Implementations benchmarked sequentially, not in parallel
- Primary metric: median inference time across standalone rounds
- qwen-asr and pure C use internal inference timers; MLX-based implementations are timed after model load with explicit GPU synchronization
- Wall-clock time is retained as a secondary metric
- Default runs: 10
- **Thread policy: out-of-the-box defaults.** No thread flag is passed to any implementation; each one uses its own default configuration (qwen-asr picks `P + min(E, P + (E−P)/2)` = 12 threads on M5 Pro, upstream C picks its own default, the MLX implementations run on the GPU and are insensitive to CPU thread flags). `--threads N` can still be passed to force a uniform count for every CPU implementation.

### Note on the thread-policy change (2026-07-12)

Runs up to and including `20260711T145612Z` forced `--threads 15` (`min(system CPUs, 16)`) onto every CPU implementation for uniformity. That policy predates qwen-asr's tuned thread default: after Round 11's dynamic work-stealing scheduler, qwen-asr's optimum on M5 Pro is 12 threads, and 15 threads sits in a measured oversubscription regime (~660-690 ms offline vs ~563 ms at the default — see `experiments.md` R11-G for the t5..t15 sweep). The forced-15 policy therefore reported qwen-asr at 658 ms while the dedicated speed benchmark (binary default) reported 563 ms on the same commit. Comparing each implementation at its own default configuration is the fairer "as-shipped" comparison and is now the default; the GPU baselines are unaffected by this change.

## Reproduce

```bash
./bench/benchmark-all.sh --runs 10
```

This script:
1. Builds qwen-asr first (`bf52daf`) and latest (current HEAD)
2. Clones/builds upstream C (`antirez/qwen-asr`)
3. Clones/builds `second-state/qwen3_asr_rs` (MLX backend)
4. Runs `mlx-audio` (MLX Python)
5. Normalizes results and renders `report.md` plus charts

Output: `bench/compare-results/<timestamp>/` with `report.md`, `summary.json`, charts, and raw logs.

> **Note:** the full comparison takes 30–60 minutes because it clones and builds three external implementations.

## Latest Cross-Implementation Results

> Generated on: 2026-07-16 from `bench/compare-results/20260716T070644Z/`
> Runs per target: 10
> Hardware: Apple M5 Pro, 15 cores, 48 GB RAM, macOS 26.4
> Versions: upstream C `main` (`b00b789`), second-state `v0.2.0` (`0226270`), mlx-audio `v0.4.5`
> Thread policy: out-of-the-box defaults (no thread flag passed; qwen-asr auto-picks 12 threads on this machine)
> Results are sorted by median inference latency (fastest first).

| Implementation | Commit / Version | Median inference ms | Mean ms | Best ms | RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest, CPU) | `d141bca4` | **613** | 617 | 595 | **46.00×** |
| mlx-audio Python MLX (GPU) | `0.4.5` | 688 | 835 | 682 | 40.94× |
| second-state MLX (GPU) | `0226270` (v0.2.0) | 1,414 | 1,482 | 1,386 | 19.91× |
| pure C upstream (CPU) | `b00b789` | 1,660 | 1,665 | 1,646 | 16.96× |
| qwen-asr first port (CPU) | `bf52dafe` | 1,698 | 1,823 | 1,645 | 16.61× |

### Wall-clock timing

| Implementation | Commit / Version | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest, CPU) | `d141bca4` | **865** | 930 | 844 | **32.60×** |
| second-state MLX (GPU) | `0226270` (v0.2.0) | 1,616 | 1,745 | 1,587 | 17.42× |
| mlx-audio Python MLX (GPU) | `0.4.5` | 1,712 | 1,939 | 1,699 | 16.45× |
| pure C upstream (CPU) | `b00b789` | 1,943 | 1,950 | 1,932 | 14.49× |
| qwen-asr first port (CPU) | `bf52dafe` | 2,049 | 2,364 | 2,000 | 13.76× |

<p float="left">
  <img src="charts/benchmark-unified-latency.png" width="48%" alt="Unified latency" />
  <img src="charts/benchmark-unified-rtf.png" width="48%" alt="Unified realtime factor" />
</p>

### Findings

- With every implementation at its own default configuration, qwen-asr `d141bca4` is the **fastest implementation overall**: the pure-CPU Rust engine has lower median inference latency than both GPU baselines.
- qwen-asr is **2.77×** faster than the initial Rust port `bf52dafe`.
- qwen-asr is **2.71×** faster than the upstream pure-C implementation.
- qwen-asr is **2.31×** faster than second-state MLX GPU by inference latency.
- qwen-asr is **1.12×** faster than mlx-audio Python MLX by inference latency (613 vs 688 ms), and **1.98×** faster on wall clock (865 vs 1,712 ms).

## Why is pure CPU Rust competitive with GPU baselines?

1. **Hand-optimized NEON kernels** — custom `vDSP`/`Accelerate`, hand-written `neon_dotprod` matmul, and fused fast-attention tuned for the 0.6B model and Apple Silicon cache hierarchy.
2. **Zero framework overhead** — no tensor dispatch, memory pools, or FFI bridging; 100% Rust end-to-end.
3. **Small-model overheads matter** — a 0.6B model does not always saturate the GPU enough to dominate CPU launch, synchronization, and framework overheads.
4. **Result depends on implementation details** — as of `d141bca4`, qwen-asr CPU beats every baseline in this comparison, including `mlx-audio` on the GPU.

## Perf-round2 vs. previous implementation

A separate apples-to-apples comparison of the `perf-round2` optimization branch against the previous implementation (`main` @ `9e8205f`) is available in [`docs/research/experiments.md`](../research/experiments.md). Summary:

| Metric | Previous (`9e8205f`) | Latest (`perf-round2`) | Δ |
|---|---:|---:|---:|
| offline wall / infer | 1106 / 495 ms | 860 / 470 ms | −22.2% / −5.1% |
| segmented wall / infer | 987 / 378 ms | 740 / 356 ms | −25.0% / −5.8% |
| streaming wall / infer | 1003 / 390 ms | 753 / 365 ms | −24.9% / −6.4% |
| load floor (0.5 s clip) | 0.39 s | 0.17 s | −56% |
| 100-file LibriSpeech WER | 0.0387 | 0.0379 | better |

Accepted wins: parallel model-load conversions, batched-GEMM prefill causal attention, and default threads = performance cores.
