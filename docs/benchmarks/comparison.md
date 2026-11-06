# Cross-Implementation Comparison

Fresh 2026-11-06 run on Apple M5 Pro (15 cores, 48 GiB). Each implementation
completed **10 successful standalone offline runs** on the same 28.16-second,
44.1 kHz mono WAV. No discarded warmup, no forced thread count; each engine
uses its shipped defaults. qwen-asr current selects 12 global threads and 10
single-token decode participants. Background desktop load and a booted iOS
simulator were present; this is not a controlled idle-machine ranking.

| Implementation | Median inference | Median process wall | Inference throughput |
|---|---:|---:|---:|
| qwen-asr (current CPU) | 581.0 ms | 693.4 ms | 48.47× |
| mlx-audio Python (GPU) | 746.3 ms | 1,630.4 ms | 37.73× |
| second-state MLX (GPU) | 1,439.7 ms | 1,679.7 ms | 19.56× |
| qwen-asr first port (CPU) | 1,647.0 ms | 1,984.2 ms | 17.10× |
| pure C upstream (CPU) | 1,653.5 ms | 1,922.5 ms | 17.03× |


## Interpretation

- These are Qwen3-ASR-0.6B **model-family** comparisons, not identical runtime
  or quantization configurations. mlx-audio uses the cached MLX 8-bit model;
  the CPU engines and second-state load the local model using their own paths.
- Native inference timers differ: the current Rust timer excludes resampling,
  while MLX file-path APIs include input preprocessing after model load. GPU
  work is synchronized for its inference timer. Process wall includes startup,
  model/input loading, inference and exit and is provided alongside inference.
- The first mlx-audio invocation took 19.2154 s inference / 22.8444 s wall
  and remains included in the ten-run statistics. Medians describe typical
  runs here, not guaranteed first-use latency; OS/runtime caches are uncontrolled.
- Throughput is 28.16 s divided by inference duration. CPU/C and Python MLX
  were measured in sequential cohorts; second-state was retried later after
  installing a missing Metal Toolchain and rebuilding its temporary clone.
  Do not interpret small differences as a statistically established ranking.
- Transcripts are not identical across all implementations; this speed comparison
  does not establish accuracy equivalence. Current-project WER and streaming
  limitations are documented in [results.md](results.md).

## Versions and evidence

- qwen-asr (current CPU): `binary-de15033ee6e7`.
- mlx-audio Python (GPU): `0.4.5`.
- second-state MLX (GPU): `0226270`.
- qwen-asr first port (CPU): `bf52dafe`.
- pure C upstream (CPU): `9246942`.

The current binary SHA256 is
`de15033ee6e7034f46264ac6fea7cec16789d7845e3950c6fdefa9e3298845e6`.
It includes local changes beyond `ef62a79d`; it is not attributed to clean HEAD.
mlx-audio is pinned to 0.4.5; its cached model revision is
`89e96d92ba34aca20b3e29fb10cc284097d1219f`.

[Normalized results](2026-11-06/cross/summary.json) ·
[System information](2026-11-06/cross/system_info.json) ·
[Source, binaries and model inputs](2026-11-06/provenance.json) ·
[Complete run record](2026-11-06/README.md).

<p float="left">
  <img src="charts/benchmark-unified-latency.png" width="48%" alt="Measured inference latency by implementation" />
  <img src="charts/benchmark-unified-rtf.png" width="48%" alt="Measured inference throughput by implementation" />
</p>

## Reproduce

```sh
cargo build --release -p qwen-asr-cli
bash bench/benchmark-all.sh --runs 10 --current-binary target/release/qwen-asr
```

The explicit supplied binary avoids silently benchmarking clean HEAD instead
of local optimizations. Use the pinned versions/model path in the run record
for this exact comparison. External Metal builds require the Metal Toolchain
component; an installed macOS SDK alone is insufficient.
