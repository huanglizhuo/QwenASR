# Long-audio decode investigation — 2026-11-06

## Finding and retained change

The historical +21% decode regression was not reproduced consistently: a
fresh main-versus-current three-pair run had a +2.26% median paired decode
delta, -24.51% prefill and -3.35% wall time. All transcripts matched. That
result does not establish which R15 commit caused the historical observation.

Resource measurements exposed a separate, actionable scheduler hotspot.
`Backoff::wait` spun for 4,096 polls once, then called `sched_yield()` on every
subsequent poll. Both pool joins and decode region barriers use this helper.
This policy predates R15 (introduced in `b71f3386` for Linux oversubscription).

The retained change resets the spin counter after yielding, **only on
macOS/aarch64**. Long waits alternate the existing bounded spin budget with
a yield. There is no new threshold, sleep, thread heuristic or arithmetic
change. Other platforms retain their previous policy. R15-3 remains intact.
The tradeoff is more bounded userspace spinning in exchange for fewer
scheduler calls; this is not a barrier that tolerates an indefinitely
preempted participant.

## Controlled comparisons

Each case discards one warmup per side, then measures three alternating
AB/BA pairs. Comparisons run serially, with no concurrent project builds or
tests. This M5 Pro machine had varying external background load; none of
these results is an idle-machine peak claim. Binary hashes, overrides and
before/after process, power, thermal and VM snapshots are archived alongside
the results. Models and audio are the harness defaults.

### Current branch before fix versus re-armed backoff

A = `batch-static-safe`, B = `backoff-rearm`, global `--threads 12` (the
unchanged single-token policy uses 10 participants).

| 10-minute single-pass metric | A | B | Change |
|---|---:|---:|---:|
| Decode, pair 1 | 55.081 s | 41.490 s | -24.67% |
| Decode, pair 2 | 48.225 s | 40.142 s | -16.76% |
| Decode, pair 3 | 46.210 s | 38.971 s | -15.67% |
| Median decode | 48.225 s | 40.142 s | -16.76% |
| Median process wall time | 63.284 s | 55.613 s | -12.12% |
| Median process system CPU | 66.951 CPU s | 6.562 CPU s | -90.20% |
| Median involuntary context switches | 2,070,805 | 736,905 | -64.41% |

The median of paired wall deltas is -12.84%; it differs from the ratio of
side medians above. All three long pairs improved, all outputs matched,
and each run decoded 1,765 tokens. System CPU fell by about 90% in each pair.
CPU seconds and context switches cover the entire process and all its
threads, not decode alone; context switches are not a count of yield calls.

All measured long runs had zero system Swapins/Swapouts deltas. Existing
swap usage stayed at 18,961.38 MB. Decompression varied, including more
pages in the third B run despite faster decode. Existing swap occupancy
alone does not explain this comparison.

The 2-minute case is noisy: paired decode deltas were -36.44%, +31.45% and
-3.91%. Do not infer a uniform short-context speedup from its median.

### Thread-count and oversubscription controls

- Same pre-fix binary, global 12 threads, decode override 12 versus 10:
  median paired decode -1.52% for 2 minutes and -10.80% for 10 minutes.
  This loaded-machine result does not support reverting R15-3, and does not
  settle idle-machine peak performance or justify a KV-length threshold.
- Before versus after fix with global 16 threads and
  `QWEN_ASR_DECODE_THREADS=16`: three pairs each of `wav_offline` and
  `2min_offline`, identical outputs, no hangs. Median paired wall deltas
  -3.37% and -1.46%; decode -5.71% and -1.28%. The host has 15 logical cores;
  this exercises oversubscription without changing the existing 16-thread cap.


### Final main baseline versus rebuilt fix

The final release build was compared with main `37c1e35b` using three pairs
and a 6% median decode-regression threshold. The harness exited successfully;
all outputs matched, including warmups. An iPhone 17e iOS simulator was
booted when this comparison started, and external load varied substantially.
No unrelated process was stopped.

| Metric | A: main | B: final fix |
|---|---|---|
| Decode, seconds by pair | 49.569, 88.547, 54.649 | 54.503, 55.691, 40.560 |
| Median decode | 54.649 s | 54.503 s (-0.27%) |
| Median wall | 76.356 s | 71.560 s (-6.28%) |

The paired decode deltas (+9.95%, -37.11%, -25.78%) reveal the noise hidden
by a single median. The threshold check passed, but this is **not stable
idle-machine acceptance** or evidence for a 26% speed claim. The earlier
same-branch before/after experiment isolates the backoff change more directly.
The final binary includes the test additions (compiled out in release);
its distinct binary hash is recorded separately rather than treating it as
the earlier candidate binary.

## Validation

- Regression test fails with the old behavior: two additional spin/yield
  cycles yield 8,195 total calls instead of 3. Restoring the fix makes it pass.
  Short waits remain syscall-free and long waits still yield.
- A rotating delayed participant exercises barrier visibility across 16
  generations with four threads, plus the single-participant case.
- Full default release workspace tests: **95 passed, 3 intentionally ignored**.
- Relevant no-default-feature library/kernel/batch/lockstep/regression/verify
  tests: **61 passed**.
- Formatting and all-target clippy with `-D warnings` pass in both feature
  configurations. All **6 Python harness tests** pass.
- The change preserves arithmetic order. The completed A/B runs check exact
  transcript equality; the full 2,703-item WER gate was not run.

## Reproduction and evidence

```sh
python3 bench/r15/ab.py tmp/r15/bin/batch-static-safe tmp/r15/bin/backoff-rearm \
  --cases 2min_offline,10min_offline --runs 3 --threads 12 \
  --out bench/r15/results/long-backoff-rearm
python3 bench/r15/ab.py tmp/r15/bin/batch-static-safe tmp/r15/bin/backoff-rearm \
  --cases wav_offline,2min_offline --runs 3 --threads 16 \
  --env QWEN_ASR_DECODE_THREADS=16 \
  --out bench/r15/results/backoff-oversubscribed16
python3 bench/r15/ab.py tmp/r15/bin/base tmp/r15/bin/long-fixed-final \
  --cases 10min_offline --runs 3 --threads 12 \
  --max-decode-regression 6 --out bench/r15/results/long-final-vs-base
```

Build inputs and binary/source hashes: [provenance.json](provenance.json).
Each of `baseline-recheck/`, `cap12-cap10/`, `rearm/`,
`oversubscribed16/`, and `final-vs-base/` contains metadata, per-run evidence and summaries. Full
stdout/stderr and binaries remain in ignored local results/tmp directories.
The regression test's red/green logs and [validation counts](validation.json)
are preserved here.

Remaining acceptance work: idle-machine 2-/10-minute comparisons and thread
review, Linux/x86 CI, and validation on other Apple Silicon machines. No
push or merge was performed. Headline README performance figures are unchanged.
