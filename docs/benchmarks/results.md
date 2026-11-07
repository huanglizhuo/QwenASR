# Benchmark Results

Measured on 2026-11-06 (run start), Apple M5 Pro / 15 cores / 48 GiB.
Current build: `r15-speedups` working tree with the wait-backoff fix, binary
`de15033ee6e7…`; this is **not clean HEAD**. Default release features, native
CPU, 12 global threads and the unchanged 10-participant single-token policy.
A simulator and desktop background tasks remained active. Results describe
this recorded loaded environment, not idle-machine peak performance.

[Commands, source/model/sample hashes and scope](2026-11-06/README.md) ·
[Cross-implementation comparison](comparison.md).

## Standard short audio

28.16-second mono 44.1 kHz WAV, `bench/samples/audio.wav`, 0.6B model.
10 successful standalone processes per mode; no discarded warmup. OS page
caches are uncontrolled. These dedicated runs are separate from the cross-
implementation cohort, so their medians need not match exactly.

| Mode | Median inference | Range | Median process wall | Inference throughput | Peak RSS median | Raw sample WER |
|---|---:|---:|---:|---:|---:|---:|
| offline | 569.5 ms | 553–587 ms | 676.8 ms | 49.45× | 3.28 GiB | 2.70% |
| segmented | 570.5 ms | 554–587 ms | 678.5 ms | 49.36× | 3.28 GiB | 2.70% |
| streaming | 534.0 ms | 527–558 ms | 642.0 ms | 52.73× | 3.26 GiB | 29.73% |

Inference excludes model loading, resampling and exit; process wall includes
them. Throughput uses exact WAV duration, not the CLI's rounded 28.2 s display.
RSS is the whole process's measured peak and includes mapped model pages;
it is not an incremental model memory requirement.

Raw single-sample WER is case/punctuation sensitive. Offline/segmented differ
by one punctuation edit; streaming emits a partial transcript on this clip
and has substantially worse WER. Its lower latency is **not an accuracy-
equivalent speedup**. These raw sample scores are not comparable with the
normalized LibriSpeech scores below. Per-mode data: [short/](2026-11-06/short/).

## Main-versus-current path coverage

One discarded warmup per side and three alternating AB/BA pairs for each
case, same release toolchain/profile, main `37c1e35b` versus the current binary.
All 136 invocations succeeded; all 17 cases preserve transcript bytes, and
alignment JSON also matches. Phase work can overlap in batch/segmented paths;
never sum those phase totals to infer elapsed latency. Multi-file rows use
whole-batch wall time. Negative paired changes indicate lower elapsed time.

| Case | Current median inference | Current median wall | Wall range | Paired median wall change vs main |
|---|---:|---:|---:|---:|
| 28.16 s offline (44.1 kHz) | 0.559 s | 0.663 s | 0.655–0.670 s | -20.27% |
| 28.16 s segmented | 0.557 s | 0.660 s | 0.655–0.679 s | -19.52% |
| 28.16 s streaming | 0.530 s | 0.632 s | 0.629–0.645 s | -21.10% |
| 37.19 s mixed fixture offline | 1.033 s | 1.130 s | 1.114–1.152 s | -7.61% |
| 4 s offline | 0.147 s | 0.244 s | 0.243–0.247 s | -16.66% |
| 28 s offline (16 kHz) | 0.709 s | 0.810 s | 0.788–0.856 s | -5.92% |
| 120.705 s offline | 4.180 s | 4.286 s | 4.042–5.965 s | -19.97% |
| 602.465 s offline | 57.675 s | 57.869 s | 49.650–57.994 s | -20.45% |
| 120.705 s segmented (-S30) | 2.520 s | 2.690 s | 2.655–2.712 s | -3.97% |
| 602.465 s segmented (-S30) | 11.134 s | 11.314 s | 11.226–11.502 s | -3.13% |
| 120.705 s file streaming (partial output) | 2.056 s | 2.161 s | 2.152–2.174 s | -11.99% |
| 602.465 s file streaming (partial output) | 14.796 s | 14.954 s | 14.584–15.007 s | -23.10% |
| 28 s stdin streaming | 1.859 s | 1.969 s | 1.940–1.998 s | -5.64% |
| 120.705 s stdin streaming | 12.705 s | 12.823 s | 12.341–13.042 s | -1.78% |
| 12-file batch | — | 2.634 s | 2.570–2.661 s | -20.03% |
| 60-file batch | — | 12.032 s | 12.005–12.405 s | -14.44% |
| 120.705 s segmented + aligner/JSON | 2.635 s | 5.699 s | 5.600–5.817 s | -3.15% |

The aligner row reports ASR inference separately. Its process wall time includes
forced alignment and JSON output.

The 10-minute samples are 602.465 seconds, not exactly 600. The original
historical +21% decode regression is not proven to originate from a specific
R15 commit. These measurements support the retained changes under recorded
load; they do not complete the separate idle-machine review.

[Per-run snapshots and metrics](2026-11-06/matrix/runs.jsonl) ·
[Summary with individual values and both delta definitions](2026-11-06/matrix/summary.json) ·
[Representative output files](2026-11-06/matrix/outputs/).

## Mixed fixture

37.19-second 16 kHz fixture, three standalone runs per mode:

| Mode | Median inference | Median process wall |
|---|---:|---:|
| offline | 1019 ms | 1123.3 ms |
| segmented | 1226 ms | 1334.0 ms |
| streaming | 670 ms | 779.5 ms |

No reference transcript is available. The offline output includes repetitions;
these are latency/behavior observations, not validated bilingual accuracy.
[Results and representative transcripts](2026-11-06/mixed/).

## Accuracy

| Evaluation | Items | Corpus WER | Corpus CER | Failed |
|---|---:|---:|---:|---:|
| Full dev-clean, offline | 2,703 | 2.7591% | 0.7542% | 0 |
| Same first 100 IDs, offline (from full pass) | 100 | 4.8960% | 1.6348% | 0 |
| First 100 dev-clean items, streaming | 100 | 4.8960% | 1.6348% | 0 |

Offline totals: 1,501 word edits / 54,402 reference
words; macro WER 3.1926%. Lowercase/punctuation normalization
uses the existing LibriSpeech scorer. The streaming row is a **100-item subset**,
not full-set streaming accuracy. The same 100 IDs have identical offline and
streaming transcripts in this pass; this does not extend to long samples.
No utterance failed or produced empty output.
We reran main `37c1e35b` with the same dataset, model, feature configuration
and 12 global threads. Both builds completed all 2,703 IDs and produced the
same corpus WER and CER. All normalized hypotheses matched. The recorded
transcriptions matched exactly on 2,702 items. On `1272-141231-0009`, the
candidate writes “therefore, nights” where main writes “therefore nights”.
Three targeted repetitions per binary reproduced the difference. The pre-fix
R15 binary also includes the comma, and overriding the candidate's decode
participants to 12 preserves it. [Main summary](2026-11-06/wer-main-full.json) ·
[Per-ID comparison](2026-11-06/wer-main-comparison.json) ·
[Targeted repeats](2026-11-06/wer-punctuation-recheck.json).

Three runs each of fresh builds from `70c56f25` and `76bf6add` (R15-8) emitted
main’s punctuation. All three runs at `3c0b6f9a` (R15-10, tiled multi-token
attention) emitted the additional comma. This localizes the observed change
on this recording to the R15-10 commit boundary. Full-set WER passes and the
strict transcript-identity check has one reproducible mismatch.
[Revision results and binary hashes](2026-11-06/wer-punctuation-revisions.json).

[Full-set summary](2026-11-06/wer-full.json) ·
[Per-utterance counts and output hashes](2026-11-06/wer-full-utterances.jsonl) ·
[Streaming subset summary](2026-11-06/wer-stream100.json).

### Generated long-sample accuracy

| Input / mode | Normalized WER |
|---|---:|
| 120.705 s offline | 2.8571% |
| 120.705 s segmented | 3.9286% |
| 120.705 s file `--stream` | **91.7857% (partial output)** |
| 120.705 s stdin streaming | 5.0000% |
| 602.465 s offline | 3.3145% |
| 602.465 s segmented | 3.1735% |
| 602.465 s file `--stream` | **98.3780% (partial output)** |

The November 6 file `--stream` runs produced only a small portion of these long
recordings in both main and the candidate. Their timing rows above measure partial-output execution,
**not complete long-audio transcription**. Matching main's bytes does not make
these outputs accurate. The stdin streaming path behaves differently and was
scored separately. Do not promote these file-streaming numbers as an offline-
quality speedup. Full counts/CER: [long-accuracy.json](2026-11-06/long-accuracy.json).
Concatenated samples are a regression workload, not an independent accuracy corpus.

The default finite-file EOF truncation is fixed in the November 7 follow-up.
See the [diagnosis, corrected WER and timing measurements](2026-11-07-stream/README.md).
Historical rows above retain the original partial-output measurements.
The subsequent [small-chunk follow-up](2026-11-07-stream-reset/README.md)
fixes intermediate resets, verifies coverage of every source utterance, and
measures both long recordings with 0.5-second chunks.

## Reproduce

```sh
cargo build --release -p qwen-asr-cli
bash bench/run.sh --samples-dir bench/samples/audio.wav --threads 12 --runs 10 --label current
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-full \
  --binary target/release/qwen-asr --model-dir qwen3-asr-0.6b \
  --threads 12 --mode offline --label current-full
```

Full path/cross-implementation commands and exclusions are in the
[run record](2026-11-06/README.md). Previous dated results remain in Git history
and the [experiment log](../research/experiments.md); no old timing is presented
as a fresh measurement.
