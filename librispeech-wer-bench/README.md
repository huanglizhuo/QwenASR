# LibriSpeech WER Benchmark

This directory contains Git-tracked WER benchmark scripts only. Downloaded datasets, result files, reports, archives, and temporary worktrees are ignored by Git.

For the full methodology, latest results, and interpretation, see [`docs/benchmarks/results.md`](../docs/benchmarks/results.md).

## Scripts

| Script | Purpose |
|---|---|
| [`librispeech_wer.py`](librispeech_wer.py) | Run a single WER/CER evaluation pass against a LibriSpeech-style dataset. |
| [`run_wer_range.py`](run_wer_range.py) | Evaluate WER across a git commit range by checking out each commit into a temporary worktree. |

## Quick usage

### 100-file offline WER

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 \
  --mode offline
```

### Full dev-clean evaluation (2,703 utterances)

The local `dev-clean-2` cache contains only 1,089 utterances. Use the complete
split and omit `--limit` for a full-set result:

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-full \
  --binary target/release/qwen-asr --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results \
  --label current-offline-full --mode offline
```

### Download `dev-clean` automatically

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --download-dataset \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results-100 \
  --label current-offline-100 \
  --limit 100 \
  --mode offline
```

Default download URL:

```text
https://www.openslr.org/resources/12/dev-clean.tar.gz
```

## Results layout

```
librispeech-wer-bench/results-100/<label>/
├── results.jsonl   # per-utterance records
└── summary.json    # aggregate WER/CER and failure list
```
