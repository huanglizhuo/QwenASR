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

> Generated on: 2026-07-11
> Commit: `d241af9b`
> Hardware: Apple M5 Pro, 48 GB RAM
> Threads: default CLI thread policy (12 on this machine since R11-G)

| Mode | Median inference ms | Mean ms | Best ms | Realtime factor | WER (sample) |
|---|---:|---:|---:|---:|---:|
| offline | 563 | 568.8 | 557 | 50.09× | 0.0270 |
| segmented | 572 | 570.4 | 559 | 49.30× | 0.0270 |
| streaming | 544 | 544.3 | 531 | 51.89× | 0.2973 |

#### Wall-clock timing

| Mode | Median wall ms | Mean ms | Best ms | Wall realtime factor |
|---|---:|---:|---:|---:|
| offline | 828.5 | 900.0 | 822.0 | 34.04× |
| segmented | 839.0 | 838.0 | 822.6 | 33.61× |
| streaming | 810.5 | 810.8 | 799.0 | 34.79× |

#### Note on sample WER

Offline and segmented emit the full 46-token transcript for the bundled 28.2 s clip. R11-I's INT4 decode-FFN quantization (`6ed39526`) happened to drop one spurious comma on this clip (`WER=0.0000`); reverting to INT8 in R12-B5 restores the pre-R11-I output with that one comma (`WER=0.0270`) — a single-clip artifact, not a quality signal (the full dev-clean gate moves the other way: INT8 corpus WER 0.0271 vs INT4 0.0299). The WER column here is the R12-B5 output; the speed columns remain the `d241af9b` snapshot. Earlier speed rows around `437-656 ms` used truncating paths, including a punctuation/token-count early stop or long-audio cap, and are not comparable as full-transcript results. Streaming still uses a lower-latency partial path on this sample (`WER=0.2973`).

#### Kernel profile (offline)

Per-kernel timings from a single `--profile` offline run at `d241af9b` (inference buckets only; load/audio buckets excluded):

| Kernel | Total ms | Calls |
|---|---:|---:|
| sgemm | 231.8 | 335 |
| bf16_matvec (incl. INT8 decode matvecs) | 170.0 | 196 |
| conv2d_op | 63.4 | 87 |
| attention_causal | 26.7 | 28 |
| attention_bidir | 15.3 | 18 |
| gelu | 6.7 | 106 |
| rms_norm | 5.1 | 2678 |

### Historical context

- Initial Rust port (`bf52daf`): 1,670 ms offline / 16.88× RTF (cross-implementation run, `--threads 15`)
- Round 10 (`aea0cc2`): 800 ms offline / 35.25× RTF (dedicated speed benchmark)
- Round 11 (`d241af9b`): 563 ms offline / 50.09× RTF — dynamic work-stealing scheduling, 12-thread default, INT4 decode FFN weights (see `experiments.md` Round 11)

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

> Generated on: 2026-07-11
> Commit: `d241af9b`
> Dataset: LibriSpeech `dev-clean-2`
> Items evaluated: 100
> Mode: offline
> Artifact: `librispeech-wer-bench/results-100/current-offline-100/summary.json`

| Metric | Value |
|---|---:|
| Corpus WER | 0.0350 |
| Macro WER | 0.0387 |
| Corpus CER | 0.0127 |
| Macro CER | 0.0135 |
| Failed utterances | 0 / 100 |

> Note: values updated at `R12-B5`, which reverted the R11-I INT4 decode-FFN
> quantization back to INT8 (restoring the pre-R11-I WER); the full dev-clean
> gate (2703 utts) reads corpus WER 0.0271. The speed tables above remain the
> `d241af9b` snapshot.

### Historical context

- Early baseline (`step0-current`, `12663c5`): corpus WER 0.1101
- After WER recovery tuning: corpus WER 0.0387
- Rounds 10 and 11 pre-INT4 (`963a4041`): corpus WER 0.0350
- Round 11 INT4 decode FFN (`6ed39526`): corpus WER 0.0357 — accepted at the time under the ≤ 0.0358 100-file gate, but the full dev-clean gate (R12-B3) later measured it as a +10.2% relative regression (0.0271 → 0.0299) and it was reverted to INT8 in R12-B5, restoring corpus WER 0.0350 (100-file) / 0.0271 (full set)
- Standing target: keep corpus WER ≤ 0.04 while improving speed

See [`experiments.md`](../research/experiments.md) for the full tuning diary.

---
