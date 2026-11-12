# Benchmark Report

## Methodology

- Offline benchmark on the same input WAV and Qwen3-ASR-0.6B model family across five implementations; quantization/runtime defaults differ.
- qwen-asr first: `bf52dafe`.
- qwen-asr latest: `binary-c86d2194333b`.
- Upstream C: `antirez/qwen-asr`.
- GPU baselines: `second-state/qwen3_asr_rs` MLX and `mlx-audio` Python MLX.
- Implementations are benchmarked sequentially, not in parallel; each round is a standalone process invocation.
- Primary metric is median inference time across standalone rounds for every implementation.
- qwen-asr and pure C use their internal inference timers. MLX-based implementations are timed after model load with explicit GPU synchronization; their file-path APIs include input preprocessing, so inference boundaries differ.
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
- macOS: `26.7`
- Rustc: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Model dir: `qwen3-asr-0.6b`
- Input file: `bench/samples/audio.wav`

## Results

| Implementation | Commit | Median inference ms | Mean ms | Best ms | RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `binary-c86d2194333b` | `529` | `529` | `515` | `53.23x` |
| mlx-audio Python MLX | `0.4.5` | `659` | `740` | `656` | `42.70x` |
| second-state MLX GPU | `0226270` | `1,296` | `1,297` | `1,290` | `21.74x` |
| qwen-asr (first) | `bf52dafe` | `1,592` | `1,592` | `1,571` | `17.69x` |
| pure C upstream | `9246942` | `1,604` | `1,609` | `1,562` | `17.56x` |

<details>
<summary>Wall-clock timing</summary>

| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (latest) | `binary-c86d2194333b` | `634` | `681` | `621` | `44.43x` |
| mlx-audio Python MLX | `0.4.5` | `1,463` | `1,614` | `1,458` | `19.25x` |
| second-state MLX GPU | `0226270` | `1,464` | `1,543` | `1,457` | `19.23x` |
| qwen-asr (first) | `bf52dafe` | `1,909` | `1,953` | `1,888` | `14.75x` |
| pure C upstream | `9246942` | `1,859` | `1,865` | `1,817` | `15.15x` |

</details>

![Unified latency](bench/charts/benchmark-unified-latency.png)

![Unified realtime factor](bench/charts/benchmark-unified-rtf.png)

## Findings

- qwen-asr latest `binary-c86d2194333b` is `3.01x` the speed of qwen-asr first `bf52dafe`.
- qwen-asr latest `binary-c86d2194333b` is `3.03x` the inference throughput of the upstream pure C implementation.
- qwen-asr latest `binary-c86d2194333b` is `2.45x` the inference throughput of second-state MLX GPU.
- qwen-asr latest `binary-c86d2194333b` is `1.25x` the inference throughput of mlx-audio Python MLX.

