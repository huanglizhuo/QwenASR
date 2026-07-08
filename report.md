# Benchmark Report

## Methodology

- Offline benchmark on the same input WAV and model across five implementations.
- qwen-asr first: `bf52daf`.
- qwen-asr latest: `aea0cc2`.
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
- Model dir: `/Users/lizhuo/owork/q-asr/qwen3-asr-0.6b`
- Input file: `/Users/lizhuo/owork/q-asr/bench/samples/audio.wav`

## Results

| Implementation | Commit | Median inference ms | Mean ms | Best ms | RTF |
|---|---:|---:|---:|---:|---:|
| mlx-audio Python MLX | `0.4.4` | `693` | `753` | `679` | `40.66x` |
| qwen-asr (latest) | `aea0cc2` | `878` | `1,475` | `846` | `32.10x` |
| second-state MLX GPU | `0226270` | `1,397` | `1,515` | `1,377` | `20.16x` |
| pure C upstream | `b00b789` | `1,660` | `1,668` | `1,645` | `16.97x` |
| qwen-asr (first) | `bf52daf` | `1,670` | `1,783` | `1,656` | `16.88x` |

<details>
<summary>Wall-clock timing</summary>

| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| mlx-audio Python MLX | `0.4.4` | `1,745` | `1,859` | `1,698` | `16.14x` |
| qwen-asr (latest) | `aea0cc2` | `1,175` | `1,774` | `1,106` | `24.01x` |
| second-state MLX GPU | `0226270` | `1,599` | `1,779` | `1,576` | `17.61x` |
| pure C upstream | `b00b789` | `1,938` | `1,950` | `1,922` | `14.53x` |
| qwen-asr (first) | `bf52daf` | `2,020` | `2,354` | `2,004` | `13.96x` |

</details>

![Unified latency](bench/charts/benchmark-unified-latency.png)

![Unified realtime factor](bench/charts/benchmark-unified-rtf.png)

## Findings

- qwen-asr latest `aea0cc2` is `1.90x` the speed of qwen-asr first `bf52daf`.
- qwen-asr latest `aea0cc2` is `1.89x` faster than the upstream pure C implementation.
- qwen-asr latest `aea0cc2` is `1.59x` faster than second-state MLX GPU by inference latency.
- qwen-asr latest `aea0cc2` is `0.79x` faster than mlx-audio Python MLX by inference latency.

