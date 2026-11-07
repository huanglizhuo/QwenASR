# Round 15 performance tools

Run from the repository root. Python 3.10+ uses only the standard library.
The CPU probes target Apple Silicon; `poolbench` is an independent Cargo
workspace, not part of release builds. None of these probes enables an
experimental kernel in the library.

## Reproducible A/B

Build both revisions with the same Rust toolchain, Cargo release profile,
features and target CPU. Do not compile, run tests, or run other benchmarks
while collecting timings. Keep power mode and background load constant;
record idle and loaded results separately. Do not kill unrelated system
services to make a run look idle.

```sh
python3 -m unittest discover -s bench/r15 -p 'test_*.py'
python3 bench/r15/ab.py /path/to/baseline /path/to/candidate \
  --cases 2min_offline,10min_offline --runs 3 --threads 12 \
  --max-decode-regression 6 --out bench/r15/results/base-head

# Same binary, change only the number of decode participants.
python3 bench/r15/ab.py /path/to/candidate /path/to/candidate \
  --cases 2min_offline,10min_offline --runs 3 --threads 12 \
  --a-env QWEN_ASR_DECODE_THREADS=12 --b-env QWEN_ASR_DECODE_THREADS=10 \
  --out bench/r15/results/cap12-cap10

bench/r15/ident.sh /path/to/baseline /path/to/candidate bench/r15/results/identity
```

Each case runs one discarded warmup per side, followed by alternating AB/BA
pairs. The output directory must be new. Every invocation saves stdout,
stderr, return code, phase timings, wall time and before/after process,
load-average, power and thermal snapshots. On macOS, snapshots also preserve
full `vm_stat` and swap-usage output. Each row includes child-process
`resource_usage` deltas for user/system CPU seconds, minor/major faults, and
voluntary/involuntary context switches. These cover the whole process and
its threads, not just decode; context-switch counts are not yield-call counts.
When `/usr/bin/time` is available on macOS (BSD `-l`) or Linux (GNU `-v`),
`metrics.peak_rss_bytes` records each invocation's peak resident set in bytes.
It is not the cumulative `RUSAGE_CHILDREN` high-water mark. Raw timer output is
retained in stderr, and `measurement_cmd` records the wrapper; wall time includes
its small launch overhead. Without a supported timer, RSS is omitted. Timeouts
kill the complete benchmark process group, including the timer.
Compare VM counter deltas: existing swap usage alone does not prove active
swapping. `metadata.json` records binary
hashes and explicit environment overrides. Inherited `QWEN_ASR_*` variables
are removed; pass desired switches with `--env`, `--a-env`, or `--b-env`.

A failed process or empty transcript stops the run. Any differing transcript
(including warmups and within-side instability) fails the final check.
The optional decode gate fails when median B ms/token exceeds A by the
specified percentage, or when decode metrics are absent. Six percent is the
historical noisy-machine control band, **not** a universal acceptance limit.
Always inspect individual runs, token counts, end-to-end wall time and
background load before attributing a delta to code.

Wall time includes process launch, loading, resampling, inference and exit.
Existing debug lines expose encoder/prefill/decode; this harness does not
pretend to separate overlapped load/resampling or process teardown.

## Batch decode phase diagnosis

Add `--profile` to an A/B run to collect `batch_decode`, `batch_qkv`,
`batch_attention`, `batch_ffn`, and `batch_lm_head` timings and call counts.
These reuse the CLI's existing profiler. Layer phases record only the calling
thread's elapsed time through the existing completion barrier, not the sum of
worker times. Under preemption this is an approximate phase breakdown.
`batch_decode` covers full batched calls with B > 1; B = 1 uses the ordinary
single-session path and is excluded from those counters. FFN includes the
activation quantization between gate/up and down projections.

```sh
python3 bench/r15/ab.py /path/to/static /path/to/dynamic \
  --cases 2min_seg30,10min_seg30,multi12 --runs 3 --profile \
  --out bench/r15/results/batch-phase-profile
```

Use profiling to locate work, then run unprofiled A/B for end-to-end acceptance.
The lockstep scheduler can produce different batch sizes/step counts as
prefill and decode overlap; inspect call counts along with phase totals.
Serial `ms_tok` and its optional regression gate do not represent batched
steps. Alignment runs include separate ASR/aligner work in their debug lines;
use plain segmented input when diagnosing ASR batch attention.

## Inputs

Defaults: `qwen3-asr-0.6b/`, `qwen3-aligner-0.6b/`, `bench/samples/audio.wav`,
and `bench/long/samples/long-{2,10}min.wav`. See
`bench/long/build_long_samples.py` for long sample generation.
Use `--model`, `--aligner` and `--fixtures` to override local directories.

Optional short, push and multi-file cases use `fixtures/clip{4,28,60}s.wav`,
`fixtures/multi/m00.wav` ... `m11.wav`, and `fixtures/multi60/m00.wav` ...
`m59.wav`. The previous session's fixtures have been copied locally into
this ignored directory; WAVs, models, build outputs and fresh raw results
are not committed. The seven-case identity suite needs no optional fixtures.
`srt_2min` saves and compares the actual alignment JSON as well as stdout.

## Probes

```sh
clang -O3 -mcpu=native -pthread bench/r15/bw/bw.c -o bench/r15/bw/bw
clang -O3 -mcpu=native -pthread bench/r15/bw/stage.c -o bench/r15/bw/stage
clang -O3 -mcpu=native -pthread bench/r15/bw/stage2.c -o bench/r15/bw/stage2
bench/r15/bw/bw 10
bench/r15/bw/stage 10 1 5 100
bench/r15/bw/stage2 10 1 100
cargo run --release --manifest-path bench/r15/poolbench/Cargo.toml --bin poolbench -- 10
cargo run --release --manifest-path bench/r15/poolbench/Cargo.toml --bin attn -- p
cargo run --release --manifest-path bench/r15/poolbench/Cargo.toml --bin melcache -- bench/long/samples/long-2min.wav
cargo run --release --manifest-path bench/r15/poolbench/Cargo.toml --bin resamp -- bench/samples/audio.wav
```

`attn` uses QoS hints (`p`/`e`), not core affinity. `resamp` only accepts a
canonical 44-byte-header mono PCM16 44.1 kHz WAV; it reports numerical
differences and is **not** an accuracy gate. C probes approximate decode
memory traffic and drop sub-cache-line tails; their timings are not ASR
latency or evidence for adopting a production change.

`history/` preserves one reference copy of the earlier seven-case transcripts
and per-revision hashes (the saved revisions are identical),
alignment JSON, multi-file output and binary hashes. Historical binary names
are unverified labels; local copies are under ignored `tmp/r15/historical-bin/`.
Rebuild from Git for causal comparisons instead of trusting these names.

The interrupted November 6 rerun is preserved under
[`history/2026-11-06-loaded-ab/`](history/2026-11-06-loaded-ab/STATUS.md).
It is **not a performance acceptance result**: unrelated background jobs
changed during the run. Fresh base/head build provenance is in
[`history/rebuild-provenance.json`](history/rebuild-provenance.json).

The [long-audio investigation](history/long-audio/README.md) records the
macOS/aarch64 wait-backoff fix, paired measurements, thread-count controls
and validation. Loaded-machine results do not replace idle-machine gates.

## Comprehensive published run

The [2026-11-06 run](../../docs/benchmarks/2026-11-06/README.md) combines
17 alternating baseline/candidate cases, ten-run short-mode and external
implementation comparisons, full offline dev-clean WER and a separate
100-item streaming evaluation. New `mixed_offline`, `2min_stream` and
`10min_stream` cases complete file-streaming coverage. File/stdin runs are
not microphone-paced streaming latency tests.
