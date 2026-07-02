# Benchmark Report

## Methodology

- Offline benchmark on the same input WAV and model across five implementations.
- qwen-asr first: `bf52daf`.
- qwen-asr latest: `f28145c`.
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
| qwen-asr (latest) | `f28145c` | `486` | `486` | `477` | `58.02x` |
| mlx-audio Python MLX | `0.4.4` | `688` | `892` | `680` | `40.92x` |
| second-state MLX GPU | `0226270` | `1,401` | `1,530` | `1,388` | `20.10x` |
| pure C upstream | `b00b789` | `1,650` | `1,656` | `1,639` | `17.06x` |
| qwen-asr (first) | `bf52daf` | `1,669` | `1,667` | `1,644` | `16.90x` |

<details>
<summary>Wall-clock timing</summary>

| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `f28145c` | `734` | `779` | `726` | `38.39x` |
| mlx-audio Python MLX | `0.4.4` | `1,732` | `2,006` | `1,707` | `16.26x` |
| second-state MLX GPU | `0226270` | `1,607` | `1,792` | `1,584` | `17.53x` |
| pure C upstream | `b00b789` | `1,922` | `1,926` | `1,907` | `14.65x` |
| qwen-asr (first) | `bf52daf` | `2,010` | `2,042` | `1,986` | `14.03x` |

</details>

![Unified latency](bench/charts/benchmark-unified-latency.png)

![Unified realtime factor](bench/charts/benchmark-unified-rtf.png)

## Findings

- qwen-asr latest `f28145c` is `3.43x` the speed of qwen-asr first `bf52daf`.
- qwen-asr latest `f28145c` is `3.40x` faster than the upstream pure C implementation.
- qwen-asr latest `f28145c` is `2.88x` faster than second-state MLX GPU by inference latency.
- qwen-asr latest `f28145c` is `1.42x` faster than mlx-audio Python MLX by inference latency.

