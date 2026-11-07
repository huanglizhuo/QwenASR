# Finite-file streaming EOF fix — 2026-11-07

This follow-up measures default file `--stream` on the same generated LibriSpeech
recordings used in the [November 6 benchmark](../2026-11-06/README.md).
It corrects truncated output; the old partial-output timings do not measure the
same amount of transcription work as the fixed implementation.
This is an unreleased source fix based on `qwen-asr 0.11.1`.

## Results

All values below are medians of three measured runs, excluding warmup. A is
the unchanged release source; B includes the EOF fix.

| File `--stream` | A WER | B WER | A inference | B inference | A process wall | B process wall |
|---|---:|---:|---:|---:|---:|---:|
| 28.16 s | 29.7297% | **0.0000%** | 0.555 s | 0.603 s | 0.673 s | 0.720 s |
| 120.705 s | 91.7857% | **2.8571%** | 2.040 s | 4.290 s | 2.163 s | 4.411 s |
| 602.465 s | 98.3780% | **2.8914%** | 16.616 s | 54.209 s | 16.804 s | 54.390 s |

The 2-minute result has 8 word edits out of 280 reference words, and the
10-minute result has 41 out of 1418. Fixed outputs contain 370 and 1764 text
tokens respectively, below the 2048-token budget; A stops at 32 in every case.
All 24 invocations succeeded, and each binary's output hashes agree across its
warmup and three measured runs for each case. Thus every measured run has the
same WER within its case and binary.

Fixed inference ranges are 4.076–4.738 s for the 2-minute sample and
49.045–59.449 s for the 10-minute sample. Fixed process-wall ranges are
4.193–4.858 s and 49.228–59.653 s. These ranges reflect the loaded desktop;
the larger absolute time after the fix includes decoding the rest of the audio.
The A/B runner's percentage deltas must not be interpreted as a throughput
regression for equivalent output.

[WER/counts and timing summary](results.json),
[raw run records with environment snapshots](ab/runs.jsonl),
[A/B metadata](ab/metadata.json), [timing summary](ab/summary.json).

## Cause and scope

The finite-file defaults are 8-second chunks and 99 unfixed chunks. Both the
120.705-second and 602.465-second recordings defer decode until EOF. The final
decode nevertheless used the per-step budget of 32 tokens, so it returned only
the start of either recording. Raising that budget to 4096 on the unchanged
release binary restored the 2-minute transcript, reducing WER from 91.7857% to
2.8571%.

The fix gives the final file chunk at least the existing offline budget of
2048 tokens. A larger explicit `--stream-max-new-tokens` remains supported;
intermediate steps retain the configured budget. Neither degeneracy nor
periodic context reset runs at EOF, where a reset could discard the final
hypothesis before emitting it. The offline and incremental push paths retain
their existing decoding behavior.

The model-gated regression test uses the existing 28.16-second fixture. It
checks the final phrase, generation beyond 32 tokens, callback/return-value
agreement, and EOF at chunk index 45 (the periodic reset boundary). It failed
on the original code with the sentence cut off after “worth bringing”.

## Measurement

Both binaries were built locally from the same release commit with Rust 1.97.1,
default features, and `target-cpu=native`. Hardware is an Apple M5 Pro with
48 GiB RAM. Runs use 12 threads and the 0.6B model, with one warmup per side
followed by three measured alternating pairs per case. No agent build or test
runs overlap the timing pass. This is a loaded-desktop measurement, with
BTLEServer and simulator processes observed, not an idle-machine acceptance.
The 10-minute warmup transcript was scored while the subsequent A run began;
these are descriptive loaded-desktop timings, not an isolated performance gate.

Process wall time includes startup, model loading and exit. Inference is the
CLI's existing internal timer. WER uses the shared LibriSpeech normalization
and scorer against the unchanged sidecar references. The concatenated long
samples are regression fixtures, not a separate accuracy corpus.

See [provenance.json](provenance.json) for compiler, base commit, binary and
sample hashes, and [source.patch](source.patch) for the measured source delta.

```sh
cargo test --release -p qwen-asr --test file_stream
python3 bench/r15/ab.py tmp/stream-fix/before tmp/stream-fix/after \
  --cases wav_stream,2min_stream,10min_stream --runs 3 --threads 12 \
  --out tmp/stream-fix/recheck
python3 bench/long/score_long.py bench/long/samples/long-2min.ref.txt \
  < tmp/stream-fix/recheck/2min_stream-1-B.stdout
python3 bench/long/score_long.py bench/long/samples/long-10min.ref.txt \
  < tmp/stream-fix/recheck/10min_stream-1-B.stdout
```

The A/B runner's exact-output check is expected to exit with status 1 because
this fix intentionally completes previously truncated transcripts. Evaluate
WER, successful exits, and repeatability within each binary instead.

Final validation passed 96 workspace tests with 3 pre-existing ignored tests,
the new regression under `--no-default-features`, Clippy with warnings denied,
and formatting. Details are in [validation.json](validation.json).

## Remaining limits

- The final decode still has a bounded token budget. Dense or longer recordings
  may need segmentation or a larger explicit budget.
- File streaming still defers output under the existing unfixed-chunk defaults;
  this change does not make it equivalent to live stdin streaming.
- This EOF-only phase left non-final reanchor text loss unresolved. It is fixed
  in the [small-chunk follow-up](../2026-11-07-stream-reset/README.md), which
  retains complete file context and checks source-utterance coverage. Results
  and source artifacts on this page predate that subsequent fix.
- This pass does not re-run the full 2703-item streaming corpus or validate
  the 1.7B model.
