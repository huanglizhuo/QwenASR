# Batch attention experiment — 2026-11-06

Dynamic task claiming is **not enabled** in the final code. The experiment
passed bitwise state/output checks but showed no repeatable end-to-end win.
The retained changes are disjoint per-GQA output borrows, diagnostic phase
counters and exactness tests.

A = static scheduling with disjoint slices; B = dynamic scheduling with the
same slices and instrumentation. Both use 12 pool threads, default features,
native CPU tuning, and the same Rust release configuration. Each case has
one discarded warmup per binary and AB/BA alternation. All outputs match.

| Unprofiled case (6 pairs) | Ratio of wall medians | Median paired wall delta | B faster pairs |
|---|---:|---:|---:|
| 2 min, -S30 | +0.33% | -0.72% | 3/6 |
| 10 min, -S30 | +1.47% | -0.49% | 4/6 |
| 12 files | +2.12% | +0.26% | 3/6 |

Three-pair profiling showed the 10-minute attention total falling from
1,389.0 to 1,129.4 ms (-18.69% by separate medians; -15.77% paired). This did
not yield a repeatable unprofiled latency win. The 2-minute profile had
92 vs 102 median batched calls: pipeline occupancy also changes with timing,
so summed stage time is not a controlled per-token kernel comparison.

Conditions: external AC power, persistent BTLEServer background load,
interactive applications present. Both 10-minute sides drifted from about
10 to 12 seconds during the unprofiled run. These are loaded-machine
experiments, not the deferred idle-machine R15-3/long-single-pass gate.
Caller-thread phase clocks also approximate global stage durations under
preemption.

`profile/` and `wall/` contain all run records (including environment snapshots),
summary metrics and binary hashes. Full per-run stdout/stderr remain locally
under ignored `bench/r15/results/batch-dynamic-{profile,wall}/`.
`dynamic-vs-static.patch` preserves the rejected task-claiming experiment;
`provenance.json` records builds and the retained source hash. No dynamic
scheduler flag or context-length heuristic was added to production.

`retained-check/` compares the final static/disjoint/profiling build against
the previous branch-head binary (three unprofiled pairs, 2-minute -S30 and
12 files). Outputs match; timings are validation data, not a new speed claim.
