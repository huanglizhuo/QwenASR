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

> Generated on: 2026-07-02
> Commit: `a7470a2`
> Runs: 10

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor |
|---|---:|---:|---:|---:|
| offline | 576 | 579.4 | 560 | 48.92× |
| segmented | 448 | 448.0 | 434 | 62.88× |
| streaming | 496 | 496.0 | 480 | 56.91× |

Previous `main` before merge (`cd65501`, corrected detached-worktree run):

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor |
|---|---:|---:|---:|---:|
| offline | 461 | 463.8 | 450 | 61.17× |
| segmented | 347 | 346.5 | 336 | 81.27× |
| streaming | 351 | 357.5 | 345 | 80.34× |

Changes vs `cd65501`: +126 ms (+27.3%) offline, +109 ms (+31.4%) segmented, +152 ms (+43.3%) streaming inference; 100-file LibriSpeech offline WER unchanged at **0.0379**.

> Note: the `f28145c` run removes the `LONG_AUDIO_FAST` 6-token cap that was present in `7934c1b`, so these numbers reflect full transcription rather than truncated output and are not directly comparable to the `7934c1b` row above.

See [`results.md`](./results.md) for the full speed-benchmark page.

## Latest Cross-Implementation Results

> Generated on: 2026-07-02 from `bench/compare-results/20260702T054646Z/`
> Runs per target: 10
> Hardware: Apple M5 Pro, 15 cores, 48 GB RAM, macOS 26.4
> Versions: upstream C `main`, second-state `v0.2.0` (`0226270`), mlx-audio `v0.4.4`
> Results are sorted by median inference latency (fastest first).

| Implementation | Commit / Version | Median inference ms | Mean ms | Best ms | RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest full comparison) | `a7470a2` | 656 | 657 | 640 | 42.99× |
| mlx-audio Python MLX | `0.4.4` | 683 | 714 | 675 | 41.23× |
| second-state MLX GPU | `0226270` (v0.2.0) | 1,367 | 1,386 | 1,358 | 20.59× |
| pure C upstream | `b00b789` | 1,648 | 1,654 | 1,632 | 17.09× |
| qwen-asr (first) | `bf52daf` | 1,687 | 1,693 | 1,631 | 16.72× |

> **Note:** the cross-implementation run passes `--threads 15` to every implementation, while the dedicated speed benchmark uses the binary's default (performance cores only, 5 threads on this machine). The dedicated benchmark therefore reports lower latency for qwen-asr latest (`576 ms` / `48.92×`) than the cross-implementation run (`656 ms` / `42.99×`). Both runs now use the same `a7470a2` commit.

### Wall-clock timing

| Implementation | Commit / Version | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest full comparison) | `a7470a2` | 906 | 933 | 883 | 31.12× |
| mlx-audio Python MLX | `0.4.4` | 1,717 | 1,794 | 1,697 | 16.40× |
| second-state MLX GPU | `0226270` (v0.2.0) | 1,576 | 1,670 | 1,558 | 17.87× |
| pure C upstream | `b00b789` | 1,912 | 1,920 | 1,901 | 14.73× |
| qwen-asr (first) | `bf52daf` | 2,043 | 2,081 | 1,968 | 13.81× |

<p float="left">
  <img src="charts/benchmark-unified-latency.png" width="48%" alt="Unified latency" />
  <img src="charts/benchmark-unified-rtf.png" width="48%" alt="Unified realtime factor" />
</p>

### Findings

- In the latest full cross-implementation run, qwen-asr `a7470a2` is **2.57×** faster than the initial Rust port `bf52daf`.
- In the latest full cross-implementation run, qwen-asr `a7470a2` is **2.51×** faster than the upstream pure C implementation.
- In the latest full cross-implementation run, qwen-asr `a7470a2` is **2.08×** faster than second-state MLX GPU (v0.2.0) by inference latency.
- In the latest full cross-implementation run, qwen-asr `a7470a2` is **1.04×** faster than mlx-audio Python MLX (v0.4.4) by inference latency.

## Why does pure CPU Rust beat GPU baselines?

1. **Hand-optimized NEON kernels** — custom `vDSP`/`Accelerate`, hand-written `neon_dotprod` matmul, and fused fast-attention tuned for the 0.6B model and Apple Silicon cache hierarchy.
2. **Zero framework overhead** — no tensor dispatch, memory pools, or FFI bridging; 100% Rust end-to-end.
3. **Model too small for GPU** — a 0.6B model cannot saturate the Metal GPU; kernel launch overhead and CPU↔GPU copies dominate.
4. **mlx-audio 8-bit overhead** — quantization saves memory but dequantization during compute adds extra work.

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
