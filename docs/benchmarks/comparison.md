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

## Current qwen-asr HEAD

> Generated on: 2026-07-11
> Commit: `d241af9b`
> Runs: 10

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor |
|---|---:|---:|---:|---:|
| offline | 563 | 568.8 | 557 | 50.09× |
| segmented | 572 | 570.4 | 559 | 49.30× |
| streaming | 544 | 544.3 | 531 | 51.89× |

Previous dedicated benchmark (`a7470a2`, full-transcript fix not yet applied in the saved run):

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor |
|---|---:|---:|---:|---:|
| offline | 576 | 579.4 | 560 | 48.92× |
| segmented | 448 | 448.0 | 434 | 62.88× |
| streaming | 496 | 496.0 | 480 | 56.91× |

The saved `a7470a2` result stopped after 27 tokens (`WER=0.4324` on the speed sample) and is kept only as historical context. As of Round 11 (`d241af9b`) the full 46-token transcript is produced *and* the offline median (563 ms) is faster than that truncated run — the current offline/segmented transcripts match the sample reference exactly (`WER=0.0000`).

> Note: the `1d677db` fix removed the remaining punctuation/token-count early stop, so post-fix speed numbers are not directly comparable to earlier truncated rows.

See [`results.md`](./results.md) for the full speed-benchmark page.

## Latest Cross-Implementation Results

> Generated on: 2026-07-12 from `bench/compare-results/20260711T235106Z/`
> Runs per target: 10
> Hardware: Apple M5 Pro, 15 cores, 48 GB RAM, macOS 26.4
> Versions: upstream C `main` (`b00b789`), second-state `v0.2.0` (`0226270`), mlx-audio `v0.4.5`
> Thread policy: out-of-the-box defaults (no thread flag passed; qwen-asr auto-picks 12 threads on this machine)
> Results are sorted by median inference latency (fastest first).

| Implementation | Commit / Version | Median inference ms | Mean ms | Best ms | RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `50c84d45` | 595 | 597 | 582 | 47.44× |
| mlx-audio Python MLX | `0.4.5` | 730 | 790 | 721 | 38.56× |
| second-state MLX GPU | `0226270` (v0.2.0) | 1,446 | 1,513 | 1,437 | 19.47× |
| qwen-asr (first) | `bf52daf` | 1,722 | 1,834 | 1,710 | 16.38× |
| pure C upstream | `b00b789` | 1,732 | 1,734 | 1,703 | 16.26× |

> **Note:** with the out-of-the-box thread policy the comparison and the dedicated speed benchmark now use the same qwen-asr configuration; the residual difference (595 ms here vs 563 ms in the dedicated run) is ordinary run-to-run machine variance. The previous run (`20260711T145612Z`, forced `--threads 15`) reported qwen-asr at 658 ms — see the thread-policy note above.

### Wall-clock timing

| Implementation | Commit / Version | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `50c84d45` | 877 | 931 | 864 | 32.17× |
| second-state MLX GPU | `0226270` (v0.2.0) | 1,666 | 1,800 | 1,652 | 16.90× |
| mlx-audio Python MLX | `0.4.5` | 1,828 | 1,930 | 1,801 | 15.41× |
| pure C upstream | `b00b789` | 2,033 | 2,035 | 2,008 | 13.85× |
| qwen-asr (first) | `bf52daf` | 2,093 | 2,423 | 2,081 | 13.47× |

<p float="left">
  <img src="charts/benchmark-unified-latency.png" width="48%" alt="Unified latency" />
  <img src="charts/benchmark-unified-rtf.png" width="48%" alt="Unified realtime factor" />
</p>

### Findings

- With every implementation at its own default configuration, qwen-asr `50c84d45` is the **fastest implementation overall** — the pure-CPU Rust engine beats every GPU baseline on median inference latency.
- qwen-asr is **2.90×** faster than the initial Rust port `bf52daf`.
- qwen-asr is **2.91×** faster than the upstream pure C implementation.
- qwen-asr is **2.43×** faster than second-state MLX GPU (v0.2.0) by inference latency.
- qwen-asr is **1.23×** faster than mlx-audio Python MLX (v0.4.5) by median inference latency (595 vs 730 ms), and **2.08×** faster on wall clock (877 vs 1,828 ms).

## Why is pure CPU Rust competitive with GPU baselines?

1. **Hand-optimized NEON kernels** — custom `vDSP`/`Accelerate`, hand-written `neon_dotprod` matmul, and fused fast-attention tuned for the 0.6B model and Apple Silicon cache hierarchy.
2. **Zero framework overhead** — no tensor dispatch, memory pools, or FFI bridging; 100% Rust end-to-end.
3. **Small-model overheads matter** — a 0.6B model does not always saturate the GPU enough to dominate CPU launch, synchronization, and framework overheads.
4. **Result depends on implementation details** — as of `d241af9b` (Round 11: dynamic work-stealing scheduling, 12-thread default, INT4 decode FFN weights), qwen-asr CPU beats every baseline in the comparison, including `mlx-audio` on the GPU.

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
