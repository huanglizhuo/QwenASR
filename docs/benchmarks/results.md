# Benchmark Results

Latest results for the standard speed sample and LibriSpeech WER.

## Contents

- [Speed Benchmark](#speed-benchmark)
- [WER Benchmark](#wer-benchmark)

## Speed Benchmark

Speed benchmark for the standard 28.2 s mono WAV sample (`bench/samples/audio.wav`).

### Methodology

- Machine: Apple M5 Pro (15 cores), 48 GB RAM
- Model: `qwen3-asr-0.6b`
- Audio: `bench/samples/audio.wav` (28.2 s)
- Binary: `target/release/qwen-asr` built with `RUSTFLAGS="-C target-cpu=native"`
- Modes:
  - `offline` — full-file transcription
  - `segmented` — `-S 30`
  - `streaming` — `--stream`
- Metric: median inference time across 10 standalone runs
- Reference transcript: `bench/samples/audio.txt`

### Reproduce

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
./bench/run.sh --label current --runs 10
```

Results are written to `bench/results/current/`.

### Latest Results

> Generated on: 2026-07-08
> Commit: `aea0cc2`
> Hardware: Apple M5 Pro, 48 GB RAM
> Threads: default CLI thread policy

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor | WER (sample) |
|---|---:|---:|---:|---:|---:|
| offline | 800 | 810.1 | 784 | 35.25× | 0.0270 |
| segmented | 794 | 799.1 | 783 | 35.52× | 0.0270 |
| streaming | 773 | 772.9 | 759 | 36.48× | 0.2973 |

#### Wall-clock timing

| Mode | Median wall ms | Mean ms | Best ms | Wall realtime factor |
|---|---:|---:|---:|---:|
| offline | 1052.4 | 1108.4 | 1038.9 | 26.80× |
| segmented | 1045.6 | 1052.9 | 1033.7 | 26.97× |
| streaming | 1025.6 | 1025.2 | 1012.4 | 27.50× |

#### Note on sample WER

Offline and segmented now emit the full 46-token transcript for the bundled 28.2 s clip (`WER=0.0270`). Earlier speed rows around `437-656 ms` used truncating paths, including a punctuation/token-count early stop or long-audio cap, and are not comparable as full-transcript results. Streaming still uses a lower-latency partial path on this sample (`WER=0.2973`).

#### Kernel profile (offline)

When run with `--profile`, the offline run reports per-kernel timings. The latest profile will be inserted here after regeneration.

### Historical context

- Initial Rust port (`bf52daf`): 1,670 ms offline / 16.88× RTF (latest cross-implementation run, `--threads 15`)
- Current implementation (`aea0cc2`): 800 ms offline / 35.25× RTF (dedicated speed benchmark), 878 ms offline / 32.10× RTF (latest cross-implementation run, `--threads 15`)

See [`comparison.md`](./comparison.md) for the latest cross-implementation numbers and [`experiments.md`](../research/experiments.md) for the full optimization diaries.

---

## WER Benchmark

Word-error-rate benchmark on LibriSpeech `dev-clean`.

### Methodology

- Dataset: LibriSpeech `dev-clean` (cached locally as `librispeech-wer-bench/dev-clean-2/`)
- Model: `qwen3-asr-0.6b`
- Binary: `target/release/qwen-asr`
- Mode: `offline` (default for the 100-file gate)
- Metric: corpus WER/CER and macro WER/CER
- Preprocessing: lowercasing and punctuation stripping before Levenshtein distance

### Reproduce

#### 100-file offline gate

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 --mode offline
```

#### Full 1,089-utterance dataset

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results \
  --label current-offline-full \
  --mode offline
```

#### Auto-download dataset

If `dev-clean-2/` is not present:

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --download-dataset \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 --mode offline
```

### Latest Results

> Generated on: 2026-07-07
> Dataset: LibriSpeech `dev-clean-2`
> Items evaluated: 100
> Mode: offline
> Artifact: `bench/wer-results/20260707T144417Z/summary.json`

The later `aea0cc2` long-audio parallel decode work recorded this 100-file gate as unchanged in `docs/research/experiments.md`.

| Metric | Value |
|---|---:|
| Corpus WER | 0.0357 |
| Macro WER | 0.0397 |
| Corpus CER | 0.0122 |
| Macro CER | 0.0133 |
| Failed utterances | 0 / 100 |

### Historical context

- Early baseline (`step0-current`, `12663c5`): corpus WER 0.1101
- After WER recovery tuning: corpus WER 0.0387
- Latest target: keep corpus WER ≤ 0.04 while improving speed

See [`experiments.md`](../research/experiments.md) for the full tuning diary.

---
