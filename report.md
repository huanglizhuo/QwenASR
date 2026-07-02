# Benchmark Report

## Methodology

- Offline benchmark on the same input WAV and model across five implementations.
- qwen-asr first: `bf52daf`.
- qwen-asr latest: `a7470a2`.
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
| qwen-asr (latest) | `a7470a2` | `656` | `657` | `640` | `42.99x` |
| mlx-audio Python MLX | `0.4.4` | `683` | `714` | `675` | `41.23x` |
| second-state MLX GPU | `0226270` | `1,367` | `1,386` | `1,358` | `20.59x` |
| pure C upstream | `b00b789` | `1,648` | `1,654` | `1,632` | `17.09x` |
| qwen-asr (first) | `bf52daf` | `1,687` | `1,693` | `1,631` | `16.72x` |

<details>
<summary>Wall-clock timing</summary>

| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `a7470a2` | `906` | `933` | `883` | `31.12x` |
| mlx-audio Python MLX | `0.4.4` | `1,717` | `1,794` | `1,697` | `16.40x` |
| second-state MLX GPU | `0226270` | `1,576` | `1,670` | `1,558` | `17.87x` |
| pure C upstream | `b00b789` | `1,912` | `1,920` | `1,901` | `14.73x` |
| qwen-asr (first) | `bf52daf` | `2,043` | `2,081` | `1,968` | `13.81x` |

</details>

![Unified latency](bench/charts/benchmark-unified-latency.png)

![Unified realtime factor](bench/charts/benchmark-unified-rtf.png)

## Findings

- qwen-asr latest `a7470a2` is `2.57x` the speed of qwen-asr first `bf52daf`.
- qwen-asr latest `a7470a2` is `2.51x` faster than the upstream pure C implementation.
- qwen-asr latest `a7470a2` is `2.08x` faster than second-state MLX GPU by inference latency.
- qwen-asr latest `a7470a2` is `1.04x` faster than mlx-audio Python MLX by inference latency.

