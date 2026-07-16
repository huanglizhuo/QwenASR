# Benchmark Report

## Methodology

- Offline benchmark on the same input WAV and model across five implementations.
- qwen-asr first: `bf52dafe`.
- qwen-asr latest: `d141bca4`.
- Upstream C: `antirez/qwen-asr`.
- GPU baselines: `second-state/qwen3_asr_rs` MLX and `mlx-audio` Python MLX.
- Implementations are benchmarked sequentially, not in parallel; each round is a standalone process invocation.
- Primary metric is median inference time across standalone rounds for every implementation.
- qwen-asr and pure C use their internal inference timers. MLX-based implementations are timed after model load with explicit GPU synchronization.
- macOS Accelerate enabled for qwen-asr and pure C where applicable.
- Wall-clock time is retained as a secondary metric.
- Standalone rounds per target: `10`.
- Modes requested: `offline`.
- Results in the table and charts are sorted by median inference latency (fastest leftmost).

## Environment

- CPU: `Apple M5 Pro`
- Cores: `15 physical / 15 logical`
- Memory: `48.0 GB`
- Machine arch: `arm64`
- macOS: `26.4`
- Rustc: `rustc 1.90.0 (1159e78c4 2025-09-14)`
- Model dir: `qwen3-asr-0.6b`
- Input file: `bench/samples/audio.wav`

## Results

| Implementation | Commit | Median inference ms | Mean ms | Best ms | RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `d141bca4` | `613` | `617` | `595` | `46.00x` |
| mlx-audio Python MLX | `0.4.5` | `688` | `835` | `682` | `40.94x` |
| second-state MLX GPU | `0226270` | `1,414` | `1,482` | `1,386` | `19.91x` |
| pure C upstream | `b00b789` | `1,660` | `1,665` | `1,646` | `16.96x` |
| qwen-asr (first) | `bf52dafe` | `1,698` | `1,823` | `1,645` | `16.61x` |

<details>
<summary>Wall-clock timing</summary>

| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `d141bca4` | `865` | `930` | `844` | `32.60x` |
| mlx-audio Python MLX | `0.4.5` | `1,712` | `1,939` | `1,699` | `16.45x` |
| second-state MLX GPU | `0226270` | `1,616` | `1,745` | `1,587` | `17.42x` |
| pure C upstream | `b00b789` | `1,943` | `1,950` | `1,932` | `14.49x` |
| qwen-asr (first) | `bf52dafe` | `2,049` | `2,364` | `2,000` | `13.76x` |

</details>

![Unified latency](bench/charts/benchmark-unified-latency.png)

![Unified realtime factor](bench/charts/benchmark-unified-rtf.png)

## Findings

- qwen-asr latest `d141bca4` is `2.77x` the speed of qwen-asr first `bf52dafe`.
- qwen-asr latest `d141bca4` is `2.71x` faster than the upstream pure C implementation.
- qwen-asr latest `d141bca4` is `2.31x` faster than second-state MLX GPU by inference latency.
- qwen-asr latest `d141bca4` is `1.12x` faster than mlx-audio Python MLX by inference latency.

