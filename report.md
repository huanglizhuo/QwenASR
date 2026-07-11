# Benchmark Report

## Methodology

- Offline benchmark on the same input WAV and model across five implementations.
- qwen-asr first: `bf52dafe`.
- qwen-asr latest: `50c84d45`.
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
| qwen-asr (latest) | `50c84d45` | `594` | `596` | `582` | `47.44x` |
| mlx-audio Python MLX | `0.4.5` | `730` | `790` | `721` | `38.56x` |
| second-state MLX GPU | `0226270` | `1,446` | `1,512` | `1,437` | `19.47x` |
| qwen-asr (first) | `bf52dafe` | `1,722` | `1,834` | `1,710` | `16.38x` |
| pure C upstream | `b00b789` | `1,732` | `1,734` | `1,703` | `16.26x` |

<details>
<summary>Wall-clock timing</summary>

| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `50c84d45` | `877` | `931` | `864` | `32.17x` |
| mlx-audio Python MLX | `0.4.5` | `1,828` | `1,930` | `1,801` | `15.41x` |
| second-state MLX GPU | `0226270` | `1,666` | `1,800` | `1,652` | `16.90x` |
| qwen-asr (first) | `bf52dafe` | `2,093` | `2,423` | `2,081` | `13.47x` |
| pure C upstream | `b00b789` | `2,033` | `2,035` | `2,008` | `13.85x` |

</details>

![Unified latency](bench/charts/benchmark-unified-latency.png)

![Unified realtime factor](bench/charts/benchmark-unified-rtf.png)

## Findings

- qwen-asr latest `50c84d45` is `2.90x` the speed of qwen-asr first `bf52dafe`.
- qwen-asr latest `50c84d45` is `2.91x` faster than the upstream pure C implementation.
- qwen-asr latest `50c84d45` is `2.43x` faster than second-state MLX GPU by inference latency.
- qwen-asr latest `50c84d45` is `1.23x` faster than mlx-audio Python MLX by inference latency.

