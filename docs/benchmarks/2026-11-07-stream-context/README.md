# File-stream allocation and repeated-work optimization — 2026-11-07

## Scope

This experiment compares the complete file-stream implementation released as
`qwen-asr 0.11.2` (commit `6a88ff620f2f1cabbdbe7ce1925fe702f0a03a5b`)
with a candidate that preserves all acoustic and text context while avoiding
redundant input preparation and excess allocations. Both process the 120.705 s and 602.465 s
LibriSpeech-derived recordings with **0.5 s chunks**, the 0.6B model, and
12 threads. This baseline already preserves the full transcript; it is not
the earlier reset implementation that dropped most of the 10-minute audio.

The retained candidate skips unchanged decoder inputs before assembling them,
borrows the input audio samples instead of copying the entire recording, and limits unused
KV allocation headroom. It does not discard old audio, text or KV positions.
Context still grows with recording length; this is not a fixed context cap or
a claim of constant memory for arbitrarily long files. The input WAV and
returned transcript also remain resident. Input preparation changes only the
file-stream path; KV allocation is shared by offline, batched and live decoders.
Live/push stream state and reset logic are unchanged.

## Results

Medians of three measured runs per binary, excluding warmup. M5 Pro, 48 GiB,
macOS 26.7, Rust 1.97.1, native CPU release build, default features. This was a
loaded desktop on AC power, with no recorded thermal/performance warnings;
it is not an idle-machine acceptance run.

| File / 0.5 s chunk | WER A = B | Inference A → B | Process wall A → B | Peak RSS A → B |
|---|---:|---:|---:|---:|
| 120.705 s | **2.8571%** (8 / 280 words) | 7.463 → 7.406 s | 7.569 → 7.514 s | 3.440 → 3.430 GiB |
| 602.465 s | **2.6093%** (37 / 1418 words) | 186.650 → 182.250 s | 186.827 → 182.432 s | **8.943 → 7.265 GiB** |

The 10-minute peak-RSS median falls **18.77%**, or **1.679 GiB**. Each measured
pair saves 17.15%–20.65%. Inference medians improve 0.76% / 2.36%, but the
10-minute paired changes range from **−3.66% to +3.03%**; three loaded-machine
pairs do not establish a stable throughput improvement. Its A inference range
is 179.470–187.887 s and B is 179.823–184.905 s. The consistent result is lower
peak memory while retaining complete output.

All 16 invocations exit successfully. Each case's transcripts are byte-identical
across both binaries, warmup and repetitions, and match the previous complete
file-stream fix. WER/CER are unchanged. All 22 / 102 source utterances retain
matching text, the longest unmatched reference spans remain 2 / 3 words, and
both recordings have zero beginning/end gaps.

Final KV usage/capacity is **1956 / 2048** positions for 2 minutes and
**9613 / 10368** for 10 minutes. These are retained logical positions, not a
sliding window or a transcript cutoff. See [results and coverage](results.json),
[raw measurements](ab/runs.jsonl), [timing summaries](ab/summary.json),
[source/build provenance](provenance.json), and [validation](validation.json).

## Context-bounding experiments

Independent segments at targets of 30, 60, 90 and 120 seconds failed the strict
WER non-regression gate. Retaining the preceding finalized audio segment and
its frozen transcript also failed that gate. Retaining text alone was not a
safe substitute for the missing audio context. These bounded-context variants
are excluded from the candidate; [diagnostic results and transcripts](trials/results.json)
are retained separately. They are exploratory runs, not controlled timing comparisons.
The retained optimization prioritizes the complete baseline's exact transcript.

## Implementation and checks

Completed file encoder windows are immutable. An unchanged encoder length and
an exactly equal rollback prefix therefore identify the same decoder input
before allocating its embedding matrix. New audio and changed prefixes still
decode normally; EOF always drains the pending transcript.

KV capacity previously doubled, and three decode call sites added another
1024 positions before requesting growth. The shared allocator now doubles only
small capacities and grows larger capacities by about 1.5×; callers
request the actual required length. It releases old K storage before allocating
new V storage, reducing the overlap of live allocations. No cached values are discarded, reordered,
quantized or recomputed. The cache regression checks bitwise preservation across
multiple layers/heads and growth boundaries, including signed zero, new zeroed
slots and no-op allocation stability. It fails on the old 2048 → 4096 growth.

Before the timed pass, 97 workspace tests passed (3 existing ignored tests),
the model-backed file-stream regression passed without default features,
Clippy passed with warnings denied in both feature configurations, and all
9 A/B-harness tests passed. A separate 10-minute offline before/after check
also produced byte-identical transcripts. No full 2703-item or 1.7B rerun is claimed.

An earlier 1024-position growth policy also preserved every transcript but
increased median 10-minute inference by 1.32% while lowering peak RSS by 12.14%.
Its [complete alternating pass](linear-growth/results.json), raw measurements
and source patch are retained under `linear-growth/`. The final candidate uses
geometric growth to reduce repeated cache copying, plus sequential K/V replacement.

## Timing protocol

Use the same compiler, release profile, features and target CPU for both
binaries. Each case receives one discarded warmup per binary, followed by
three alternating measured pairs (AB, BA, AB). All 16 invocations use fresh
processes. Do not run agent builds, model tests or WER scoring during this pass.
Machine load, power and thermal snapshots accompany every run; identify whether
the measurements came from an idle or loaded desktop in the final results.

Inference is the existing CLI timer. Process wall includes startup, loading,
inference, exit and the small `/usr/bin/time` wrapper overhead. Peak RSS comes
from each individual invocation, with BSD bytes/GNU KiB normalized to bytes;
it is not a delta of the cumulative `RUSAGE_CHILDREN` maximum. Retain all raw
stdout/stderr, command lines, timings, binary hashes and environment snapshots
under `ab/`.

An optional `[stream context] kv_len=N kv_capacity=N` line records final logical
KV length and allocated position capacity. When present, the analyzer checks
that the candidate's unused headroom is less than half its logical length, except for
the initial 2048-position allocation. These are final allocation observations,
not context truncation, peak RSS or a maximum over every decode step. Missing
logs are represented as `null`, not inferred from process memory.

## Accuracy and completeness gates

[`analyze.py`](analyze.py) reuses the previous report's exact-word coverage helper
and the official LibriSpeech WER/CER scorer. Original manifest references must
concatenate exactly to each scored reference. The gates require:

- Every invocation exits successfully, and transcripts are byte-identical
  across both binaries, warmup and repetitions.
- Candidate WER and CER are unchanged from the complete baseline on both recordings.
- Neither binary has an original utterance with zero exact-word matches, an
  unmatched reference span longer than six words, or a beginning/end gap larger
  than three words.
- Final KV allocation logs, when present, are repeatable and satisfy the
  allocation-headroom check.

Coverage uses `SequenceMatcher(autojunk=False)` as a supplemental missing-region
diagnostic, not as the Levenshtein alignment used by WER. Passing these checks
does not mean error-free recognition. Both the A/B runner's exact-output gate
and the analyzer's accuracy/completeness gates must pass.

## Reproduce

```sh
python3 bench/r15/ab.py /path/to/before /path/to/after \
  --cases 2min_stream,10min_stream --stream-chunk-sec 0.5 \
  --runs 3 --threads 12 --timeout 900 --out /path/to/new-results

# Once the raw artifacts are saved in this report's ab/ directory:
python3 docs/benchmarks/2026-11-07-stream-context/analyze.py
```

These two regression recordings do not establish full-corpus, multilingual or
1.7B-model accuracy.
