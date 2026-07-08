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

> Generated on: 2026-07-08
> Commit: `aea0cc2`
> Runs: 10

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor |
|---|---:|---:|---:|---:|
| offline | 800 | 810.1 | 784 | 35.25× |
| segmented | 794 | 799.1 | 783 | 35.52× |
| streaming | 773 | 772.9 | 759 | 36.48× |

Previous dedicated benchmark (`a7470a2`, full-transcript fix not yet applied in the saved run):

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor |
|---|---:|---:|---:|---:|
| offline | 576 | 579.4 | 560 | 48.92× |
| segmented | 448 | 448.0 | 434 | 62.88× |
| streaming | 496 | 496.0 | 480 | 56.91× |

The current numbers are slower because the decoder now emits the full transcript. The saved `a7470a2` cross-implementation result stopped after 27 tokens (`WER=0.4324` on the speed sample); the current offline and segmented runs emit 46 tokens and score `WER=0.0270` on the same sample.

> Note: the `1d677db` fix removed the remaining punctuation/token-count early stop, so post-fix speed numbers are not directly comparable to earlier truncated rows.

See [`results.md`](./results.md) for the full speed-benchmark page.

## Latest Cross-Implementation Results

> Generated on: 2026-07-08 from `bench/compare-results/20260708T055239Z/`
> Runs per target: 10
> Hardware: Apple M5 Pro, 15 cores, 48 GB RAM, macOS 26.4
> Versions: upstream C `main`, second-state `v0.2.0` (`0226270`), mlx-audio `v0.4.4`
> Results are sorted by median inference latency (fastest first).

| Implementation | Commit / Version | Median inference ms | Mean ms | Best ms | RTF |
|---|---:|---:|---:|---:|---:|
| mlx-audio Python MLX | `0.4.4` | 693 | 753 | 679 | 40.66× |
| qwen-asr (latest full comparison) | `aea0cc2` | 878 | 1,475 | 846 | 32.10× |
| second-state MLX GPU | `0226270` (v0.2.0) | 1,397 | 1,515 | 1,377 | 20.16× |
| pure C upstream | `b00b789` | 1,660 | 1,668 | 1,645 | 16.97× |
| qwen-asr (first) | `bf52daf` | 1,670 | 1,783 | 1,656 | 16.88× |

> **Note:** the cross-implementation run passes `--threads 15` to every implementation, while the dedicated speed benchmark uses the binary default. The dedicated benchmark reports `800 ms` / `35.25×` for qwen-asr latest offline; the full comparison reports `878 ms` / `32.10×`. Both current runs use `aea0cc2` and produce the full transcript.

### Wall-clock timing

| Implementation | Commit / Version | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| mlx-audio Python MLX | `0.4.4` | 1,745 | 1,859 | 1,698 | 16.14× |
| qwen-asr (latest full comparison) | `aea0cc2` | 1,175 | 1,774 | 1,106 | 24.01× |
| second-state MLX GPU | `0226270` (v0.2.0) | 1,599 | 1,779 | 1,576 | 17.61× |
| pure C upstream | `b00b789` | 1,938 | 1,950 | 1,922 | 14.53× |
| qwen-asr (first) | `bf52daf` | 2,020 | 2,354 | 2,004 | 13.96× |

<p float="left">
  <img src="charts/benchmark-unified-latency.png" width="48%" alt="Unified latency" />
  <img src="charts/benchmark-unified-rtf.png" width="48%" alt="Unified realtime factor" />
</p>

### Findings

- In the latest full cross-implementation run, qwen-asr `aea0cc2` is **1.90×** faster than the initial Rust port `bf52daf`.
- In the latest full cross-implementation run, qwen-asr `aea0cc2` is **1.89×** faster than the upstream pure C implementation.
- In the latest full cross-implementation run, qwen-asr `aea0cc2` is **1.59×** faster than second-state MLX GPU (v0.2.0) by inference latency.
- In the latest full cross-implementation run, qwen-asr `aea0cc2` is **0.79×** as fast as mlx-audio Python MLX (v0.4.4) by inference latency.

## Why is pure CPU Rust competitive with GPU baselines?

1. **Hand-optimized NEON kernels** — custom `vDSP`/`Accelerate`, hand-written `neon_dotprod` matmul, and fused fast-attention tuned for the 0.6B model and Apple Silicon cache hierarchy.
2. **Zero framework overhead** — no tensor dispatch, memory pools, or FFI bridging; 100% Rust end-to-end.
3. **Small-model overheads matter** — a 0.6B model does not always saturate the GPU enough to dominate CPU launch, synchronization, and framework overheads.
4. **Result depends on implementation details** — in the latest run, qwen-asr CPU beats upstream C and second-state MLX, while `mlx-audio` is faster on median inference time.

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
