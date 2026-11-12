# Benchmarks

This folder collects the benchmark methodology, scripts, and latest results for QwenASR.

## Scripts

| Script | Path | Purpose |
|---|---|---|
| `run.sh` | `bench/run.sh` | Core micro-benchmark driver for the local `qwen-asr` binary. Runs offline/segmented/streaming modes, repeats each config `--runs` times, picks the median inference run, computes WER/CER, and writes per-mode JSON files plus a `summary.json`. |
| `benchmark-all.sh` | `bench/benchmark-all.sh` | Full cross-implementation benchmark. Builds/runs qwen-asr first (`bf52daf`), qwen-asr latest (HEAD or `--current-binary`), upstream C, `second-state/qwen3_asr_rs` MLX, and `mlx-audio`. Produces a timestamped `bench/compare-results/<timestamp>/` directory and updates root `report.md`, `bench/charts/`, and the chart copies under `docs/benchmarks/charts/`. |
| `benchmark-second-state.sh` | `bench/benchmark-second-state.sh` | Compares current qwen-asr against `second-state/qwen3_asr_rs` (libtorch CPU and MLX Metal GPU) and `mlx-audio`. |
| `r15/ab.py` | `bench/r15/ab.py` | Alternating main/candidate comparisons with discarded warmups, exact output checks, per-run phase/wall timing and process/power/VM snapshots; covers long, batch, stdin and alignment paths. |
| `compare.sh` | `bench/compare.sh` | Compares two existing result directories under `bench/results/<baseline>` vs `bench/results/<current>`. Emits a markdown table and/or JSON. |
| `render_benchmark_report.py` | `bench/render_benchmark_report.py` | Reads `summary.json` and renders the official `report.md` plus bar charts into `bench/charts/`. |
| `librispeech_wer.py` | `librispeech-wer-bench/librispeech_wer.py` | Run a single WER/CER evaluation pass against a LibriSpeech-style dataset. |
| `run_wer_range.py` | `librispeech-wer-bench/run_wer_range.py` | Evaluate WER across a git commit range by checking out each commit into a temporary worktree. |

## What we measure

| Benchmark | Script | Purpose | Primary metric |
|---|---|---|---|
| Speed | `bench/run.sh` | Inference and separate process wall latency on a fixed 28.16 s sample across offline, segmented, and streaming modes | Median inference time / realtime factor |
| WER | `librispeech-wer-bench/librispeech_wer.py` | Accuracy on LibriSpeech `dev-clean` | Corpus WER / CER |
| Comparison | `bench/benchmark-all.sh` | qwen-asr vs. upstream C, second-state MLX, and `mlx-audio` | Median inference time / realtime factor |

## Prerequisites

- Release build of `qwen-asr`:
  ```bash
  RUSTFLAGS="-C target-cpu=native" cargo build --release
  ```
- Model directory `qwen3-asr-0.6b/` present in the project root.
- Sample `bench/samples/audio.wav` present.
- For WER: `librispeech-wer-bench/dev-clean-2/` dataset (or use `--download-dataset`).
- For comparison: network access to clone upstream C, second-state, and mlx-audio repos.

## Run the benchmarks

### Speed benchmark

```bash
./bench/run.sh --label current --runs 10
```

Results land in `bench/results/current/`:
- `audio_offline.json`
- `audio_segmented.json`
- `audio_streaming.json`
- `summary.json`

### WER benchmark (100-file offline)

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 --mode offline
```

Results land in `librispeech-wer-bench/results-100/current-offline-100/`:
- `results.jsonl`
- `summary.json`

The local `dev-clean-2` directory is a 1,089-utterance subset. For the full
2,703-utterance dev-clean evaluation, use `--dataset librispeech-wer-bench/dev-clean-full`
and omit `--limit 100`. Always check `items` in the generated summary.

To download the dataset automatically:

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --download-dataset \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results \
  --label current-offline-full \
  --mode offline
```

### Full current-project matrix

See the [2026-11-06 run manifest and commands](2026-11-06/README.md) for the
short/long, streaming, batch, alignment and full-set accuracy matrix.
The [2026-11-12 Round 16 refresh](2026-11-12/README.md) reran the
cross-implementation comparison, four long-audio cases and the full-set check.
Do not run builds, tests or other benchmarks concurrently with measurements.

### Cross-implementation comparison

```bash
./bench/benchmark-all.sh --runs 10

# Include local, uncommitted optimizations after a release build:
./bench/benchmark-all.sh --runs 10 --current-binary target/release/qwen-asr
```

This produces a timestamped directory `bench/compare-results/<timestamp>/` containing:
- `report.md` — human-readable report
- `summary.json` — machine-readable normalized results
- `normalized/` — per-implementation JSON files
- `raw/` — build logs and run outputs
- `system_info.json` — environment snapshot

It also updates:
- `report.md` in the project root
- `bench/charts/benchmark-unified-latency.png`
- `bench/charts/benchmark-unified-rtf.png`
- matching chart copies in `docs/benchmarks/charts/`

> **Note:** this script clones/builds three external implementations and can take 30–60 minutes.

## Results layout

```
bench/results/<label>/
├── audio_offline.json
├── audio_segmented.json
├── audio_streaming.json
└── summary.json

bench/compare-results/<timestamp>/
├── normalized/       # unified-schema JSON per (impl, mode)
├── raw/              # build logs and run outputs
├── report.md         # rendered markdown report
├── summary.json      # array of all normalized results
└── system_info.json  # environment snapshot

librispeech-wer-bench/results-100/<label>/
├── results.jsonl     # per-utterance records
└── summary.json      # aggregate WER/CER and failure list
```

## Interpreting results

- **Inference time** excludes process startup and model load; it is the engine's internal timer.
- **Wall-clock time** is the full command runtime from shell invocation to exit.
- **Realtime factor (RTF)** = `audio_duration_s / inference_time_s`. Higher is faster.
- Speed benchmarks use medians across standalone processes; medians do not eliminate changing background load. Record load, power and thermal conditions and report the range as well.
- Standard `audio.wav` is 28.16 seconds at 44.1 kHz. Resampling and process setup are included in wall time, not the inference timer.
- A supplied binary is identified by its SHA256, not attributed to a clean HEAD commit.

## Latest results

- [Current project results (speed + WER)](./results.md)
- [Cross-implementation comparison](./comparison.md)

## Reproducibility helper

Run `docs/benchmarks/regenerate.sh` to re-run the current speed + WER benchmarks.
