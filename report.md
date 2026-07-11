# Benchmark Report

## Methodology

- Offline benchmark on the same input WAV and model across five implementations.
- qwen-asr first: `bf52dafe`.
- qwen-asr latest: `d241af9b`.
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
| qwen-asr (latest) | `d241af9b` | `658` | `668` | `629` | `42.86x` |
| mlx-audio Python MLX | `0.4.5` | `687` | `750` | `682` | `40.97x` |
| second-state MLX GPU | `0226270` | `1,388` | `1,466` | `1,378` | `20.29x` |
| qwen-asr (first) | `bf52dafe` | `1,649` | `1,651` | `1,630` | `17.10x` |
| pure C upstream | `b00b789` | `1,662` | `1,661` | `1,637` | `16.94x` |

<details>
<summary>Wall-clock timing</summary>

| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `d241af9b` | `932` | `976` | `901` | `30.25x` |
| mlx-audio Python MLX | `0.4.5` | `1,718` | `1,829` | `1,701` | `16.39x` |
| second-state MLX GPU | `0226270` | `1,593` | `1,736` | `1,579` | `17.67x` |
| qwen-asr (first) | `bf52dafe` | `2,005` | `2,040` | `1,985` | `14.06x` |
| pure C upstream | `b00b789` | `1,941` | `1,942` | `1,915` | `14.51x` |

</details>

![Unified latency](bench/charts/benchmark-unified-latency.png)

![Unified realtime factor](bench/charts/benchmark-unified-rtf.png)

## Findings

- qwen-asr latest `d241af9b` is `2.51x` the speed of qwen-asr first `bf52dafe`.
- qwen-asr latest `d241af9b` is `2.53x` faster than the upstream pure C implementation.
- qwen-asr latest `d241af9b` is `2.11x` faster than second-state MLX GPU by inference latency.
- qwen-asr latest `d241af9b` is `1.04x` faster than mlx-audio Python MLX by inference latency.

