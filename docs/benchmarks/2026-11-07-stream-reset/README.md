# Small-chunk finite-file streaming fix — 2026-11-07

This follow-up fixes the intermediate text loss left open by the
[EOF-budget fix](../2026-11-07-stream/README.md). It measures both the 120.705 s
and 602.465 s LibriSpeech-derived recordings with **0.5 s chunks**, the 0.6B
model, and 12 threads. This chunk size exercises the old periodic and stale
hypothesis resets within both recordings.

## Results

Medians of three measured runs per side, excluding warmup. A is the EOF-only
fix; B also retains context and reuses unchanged decode input.

| File / 0.5 s chunk | A WER | B WER | A inference | B inference | A process wall | B process wall |
|---|---:|---:|---:|---:|---:|---:|
| 120.705 s | 58.2143% | **2.8571%** | 11.576 s | **7.548 s** | 11.698 s | **7.683 s** |
| 602.465 s | 91.7489% | **2.6093%** | 81.926 s | **220.899 s** | 82.082 s | **221.133 s** |

All 16 invocations succeeded. Each binary's output is byte-identical across
its warmup and three measured runs for each case. B also matches the complete
context-only implementation's saved output byte for byte, so skipping
unchanged decode input does not alter these transcripts.

| Coverage diagnostic | 2 min A → B | 10 min A → B |
|---|---:|---:|
| Reference word edits | 163 → 8 / 280 words | 1301 → 37 / 1418 words |
| Original utterances without any exact-word matches | 16 → **0 of 22** | 96 → **0 of 102** |
| Longest contiguous unmatched reference span | 162 → **2 words** | 1300 → **3 words** |
| Fixed output's beginning / ending reference gap | **0 / 0 words** | **0 / 0 words** |
| Emitted text tokens | 144 → **369** | 144 → **1763** |
| Reused unchanged steps in B | **116** | **936** |

B inference ranges are 7.544–9.003 s and 213.957–221.551 s. Process-wall ranges
are 7.672–9.154 s and 214.167–221.790 s. The 10-minute before/after timings do
not represent equivalent output: A emits only 144 text tokens and loses most
of the recording. B generates the full transcript while maintaining a growing
context. Small-chunk progressive file decoding still costs substantially more
than the default 8-second, final-only configuration.

[WER, coverage and per-run times](results.json),
[raw run records and environment snapshots](ab/runs.jsonl),
[binary/configuration metadata](ab/metadata.json), and
[timing summaries](ab/summary.json).

## Cause and fix

The finite-file decoder used two resets without a mapping between text tokens
and audio positions. A periodic reset discarded cached audio while keeping a
short text tail; a stale/repeated-hypothesis reset also shortened token history.
Neither kept the accumulated stable-text cursor aligned with the shortened
history. Uncommitted rollback text could disappear, and subsequent words could
be suppressed even though more audio was processed.

Small chunks made a second issue visible: the file encoder normally adds
completed 8-second windows. Several consecutive small steps can therefore have
identical encoder input and identical hypotheses. Treating four such steps as
degeneration is incorrect. In the 2-minute 0.5-second smoke test, the old code
triggered 15 resets and scored 58.2143% WER; removing the resets restored
2.8571% WER.

The fix removes both destructive resets from `transcribe_stream`. Audio cache,
raw text, and stable text now retain the same origin until EOF. Prefix rollback
still runs against the corresponding audio, and the existing final decode
flushes pending text. The live/incremental `stream_push_audio` implementation
is unchanged. No new dependency or public configuration field is added.

The initial context-only fix completed the 10-minute recording with 2.6093%
WER, but took 301.394 s of inference in its first warmup. It repeatedly decoded
the same rollback tail between encoder updates. The final implementation also
reuses the previous hypothesis when the **entire decode input** is unchanged:
all prefill row keys and the last input row must match. New audio or a changed
text prefix still decodes normally, and EOF always decodes and flushes.
The final outputs must match the saved context-only outputs byte for byte.
The [context-only evidence](no-dedup/provenance.json) and
[coverage checks](no-dedup/coverage.json) are retained; its incomplete timing
pass was stopped before adding this reuse and is not treated as a final A/B.

The regression test covers default EOF, the old EOF reanchor boundary,
progressive decoding across that boundary, stale hypotheses with 1-second
chunks, and a per-step budget smaller than the rollback length. It checks
middle phrases, the ending, and callback/return agreement.
The progressive-boundary case fails before this fix in about two seconds.

## Measurement and completeness checks

A contains the previous EOF-only fix. B additionally retains finite-file
context. Both use the same Rust compiler, native CPU build flags, model and
samples. See [provenance.json](provenance.json) and [source.patch](source.patch).

Each case has one warmup per binary followed by three alternating measured
pairs. All invocations are fresh processes. Inference is the existing CLI
internal timer; process wall also includes model loading, startup and exit.
This was a loaded M5 Pro desktop with active simulator/browser processes, not
an idle-machine performance gate. No agent builds, model tests or WER scoring overlap
the timed pass. Per-run snapshots record load, power, thermal and VM state.

Official WER/CER uses the existing LibriSpeech `score` and normalization.
Supplemental coverage uses `SequenceMatcher(autojunk=False)` over normalized
words to locate the longest unmatched reference span, beginning/end gaps,
and exact matches within each original manifest utterance. This supplemental
diagnostic is not a Levenshtein edit alignment and does not redefine WER.
The manifest references must concatenate exactly to the scored sidecar text.

The acceptance checks require repeatable outputs, successful exits, WER below
6%, no original utterance without matching words, no unmatched span longer
than six reference words, and beginning/end gaps no larger than three words.
These checks detect the large missing regions caused by reset; they do not
claim error-free ASR. All warmup and measured transcripts are retained.

## Reproduce

```sh
cargo test --release -p qwen-asr --test file_stream
python3 bench/r15/ab.py tmp/stream-reset/before tmp/stream-reset/after \
  --cases 2min_stream,10min_stream --stream-chunk-sec 0.5 \
  --runs 3 --threads 12 --timeout 900 --out tmp/stream-reset-recheck

# Re-score the checked-in A/B artifacts (requires the LibriSpeech fixtures):
python3 docs/benchmarks/2026-11-07-stream-reset/analyze.py
```

The A/B runner's exact-output gate is expected to return status 1 because the
fix restores missing text. Successful child exits and repeatability within
each binary are checked separately by `analyze.py`.

Final checks passed 96 workspace tests (3 pre-existing ignored tests), the
model-backed file-stream regression with no default features, Clippy with
warnings denied in both feature configurations, formatting, and all 7 A/B
harness tests. `analyze.py` passed its self-checks and all completeness gates.
See [validation.json](validation.json).

## Limits

Retaining complete file context increases memory and decoding work as audio
gets longer. This is already how default unsegmented offline and final-only
file decoding work; small-chunk file decoding now retains that context too.
Use segmented offline decoding when bounded contexts are needed. These
measurements establish coverage on the two generated regression recordings,
not unlimited-duration or full-corpus accuracy. This pass does not re-run the
2703-item corpus or test the 1.7B model, and does not alter live-stream resets.
