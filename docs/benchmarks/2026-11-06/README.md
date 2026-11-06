# Comprehensive benchmark — run started 2026-11-06

This run measures the `r15-speedups` working tree, including the Apple Silicon
wait-backoff fix. It is **not** a clean `ef62a79d` build. The immutable binary
SHA256, source hashes, model/sample hashes, toolchain and initial machine
snapshot are in [provenance.json](provenance.json); the production-source
diff is preserved as [source.patch](source.patch).

Hardware: Apple M5 Pro, 15 cores, 48 GiB. Default release features and
`target-cpu=native`. A simulator was booted, BTLEServer consumed approximately
one core and other desktop processes remained active. No unrelated processes
were stopped. These are loaded-machine observations, not idle-machine peak
claims. Benchmarks, builds and tests were serialized.

## Scope and measurements

- 17 main-versus-candidate cases: standard short offline/segmented/streaming,
  mixed Chinese/English offline, 4/28-second clips, 2/10-minute audio in three
  modes, stdin streaming, 12/60-file batches, and 2-minute forced alignment.
  One discarded warmup per binary/case, then three alternating AB/BA pairs;
  12 global threads, default single-token policy. Transcript hashes are checked
  across all runs, and alignment JSON is also compared byte-for-byte.
- Standard 28.16-second sample: 10 standalone runs in each of three modes.
  The five-implementation comparison separately takes 10 offline runs per
  implementation, using each implementation's own default thread policy.
  These scripts have no discarded warmup and their first invocation is included.
- Mixed-language segmented/streaming performance: three runs per mode; no
  reference transcript is available, so no WER is claimed for that sample.
- Full LibriSpeech dev-clean: 2,703 offline utterances, plus a separately
  labelled first-100 streaming evaluation. The 1,089-item `dev-clean-2` cache
  is a subset and is not used as the full dataset.
- Long-sample accuracy is scored against the generated `.ref.txt` files.
  These concatenated samples are a regression workload, not an independent
  accuracy corpus. Long-file streaming is processed from an existing file;
  stdin receives a complete WAV, not a real-time microphone-paced stream.

The current Rust CLI inference timer excludes model loading/resampling/exit.
MLX and second-state time file-path transcription APIs after model load,
including their input preprocessing. These native inference boundaries are
not identical. Process wall time includes the full lifecycle for each engine.
OS page caches are uncontrolled; these are not cold-cache measurements. Stage medians in the A/B report are computed from all measured runs;
`run.sh` phase fields instead belong to its representative inference run.
Throughput multiplier is audio duration divided by inference time (higher is
faster). It is not the inverse convention sometimes also called RTF.

The standard sample is **28.16 s / 44.1 kHz**, while the generated long samples
are **120.705 s and 602.465 s / 16 kHz**. Do not treat them as the same resampling
workload. CPU engines and MLX use the same model family but different runtime
and quantization defaults; cross-implementation timings do not establish
accuracy equivalence.

## Commands

Run from the repository root, one command at a time. Output directories must
be fresh. The binary was copied after `cargo build --release -p qwen-asr-cli`.
The main baseline was rebuilt from `37c1e35b` with the same toolchain/profile.

```sh
python3 bench/r15/ab.py tmp/r15/bin/base tmp/r15/comprehensive/qwen-asr \
  --cases wav_offline,wav_seg30,wav_stream,mixed_offline,c4_offline,c28_offline,2min_offline,10min_offline,2min_seg30,10min_seg30,2min_stream,10min_stream,push_c28,push_2min,multi12,multi60,srt_2min \
  --runs 3 --threads 12 --out bench/r15/results/comprehensive-2026-11-06

bash bench/run.sh --binary tmp/r15/comprehensive/qwen-asr \
  --samples-dir bench/samples/audio.wav --threads 12 --runs 10 \
  --label comprehensive-2026-11-06-short
bash bench/run.sh --binary tmp/r15/comprehensive/qwen-asr \
  --samples-dir bench/samples/mixed_zh_en.wav --threads 12 --runs 3 \
  --label comprehensive-2026-11-06-mixed

python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-full \
  --binary tmp/r15/comprehensive/qwen-asr --model-dir qwen3-asr-0.6b \
  --threads 12 --mode offline --label comprehensive-2026-11-06-full \
  --output-dir librispeech-wer-bench/wer-results
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-full \
  --binary tmp/r15/comprehensive/qwen-asr --model-dir qwen3-asr-0.6b \
  --threads 12 --mode streaming --limit 100 \
  --label comprehensive-2026-11-06-stream100 \
  --output-dir librispeech-wer-bench/wer-results
```

The cross-implementation command and exact versions are recorded with its
results. `--current-binary` measures the supplied working-tree build and records
its SHA256 instead of silently checking out HEAD. External source checkouts,
virtual environments, models, raw audio and compiled binaries are ignored.

## Limits

This run covers the locally available CPU 0.6B ASR and 0.6B aligner. There is
no local 1.7B model. Linux/x86, iOS/Android, live capture pacing, Metal experimental
kernels, idle-machine acceptance and full streaming-corpus accuracy are outside
this measured set. A matching corpus WER alone is not proof of full-corpus byte identity;
main and candidate hypotheses must also be compared by utterance ID. No optimization policy or inference arithmetic was changed for
this benchmark run.

## Cross-implementation invocation

```sh
printf 'mlx-audio==0.4.5\n' > tmp/r15/comprehensive/pip-constraints.txt
PIP_CONSTRAINT="$PWD/tmp/r15/comprehensive/pip-constraints.txt" \
  bash bench/benchmark-all.sh --runs 10 --modes offline \
  --current-binary "$PWD/tmp/r15/comprehensive/qwen-asr" \
  --tmp-dir "$PWD/tmp/r15/cross-2026-11-06" \
  --report-dir "$PWD/bench/compare-results/comprehensive-2026-11-06" \
  --mlx-audio-model /Users/lizhuo/.cache/huggingface/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit/snapshots/89e96d92ba34aca20b3e29fb10cc284097d1219f
```

Replace the cached model path with your local snapshot at the same revision.
The first attempt completed four implementations but failed second-state's
build because the Metal Toolchain compiler was missing. That failure is
retained in the raw logs; it is not a latency sample. The retry and final
validation are recorded alongside the completed results.

`run.sh` previously used the rounded CLI duration (28.2 s) for throughput.
Published short/mixed and cross summaries recalculate derived throughput from
WAV headers (28.16 s for the standard clip); no measured time is altered.
The script now reads the exact header duration itself and records reported
pool width (or null when unavailable) rather than assuming hardware core
count equals the engine's thread default.

### Metal recovery

The missing compiler was installed using `xcodebuild -downloadComponent
MetalToolchain` (27B47). `xcrun -sdk macosx metal --version` reports Apple metal
32023.922. MLX generated malformed JIT preamble files during the failed build;
a plain debug-profile clean did not remove them. The retry uses
`cargo clean --release -p qwen3_asr` **inside the temporary second-state clone**
followed by `cargo build --release --no-default-features --features mlx`.
The main project's target directory is unaffected. Generated files are checked
for the earlier error fragments before accepting the rebuilt binary.

## Full offline comparison with main

```sh
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-full \
  --binary tmp/r15/bin/base --model-dir qwen3-asr-0.6b \
  --threads 12 --mode offline --label comprehensive-2026-11-06-main-full \
  --output-dir librispeech-wer-bench/wer-results
```

Both full passes completed 2,703 items with WER 2.7591%. Joining the JSONL
rows on utterance ID and checking reference equality found one punctuation
difference before normalization and zero differences after normalization.
[Main comparison and corpus digests](wer-main-comparison.json).

[All 50 cross-implementation timings](cross/runs.json) reproduce the five published medians.
The Rust harness retains its representative transcript. The C and MLX logs
also retain each invocation’s transcript. The [targeted builds](wer-punctuation-revisions.json)
reproduce the comma difference across the R15-8 to R15-10 boundary.
