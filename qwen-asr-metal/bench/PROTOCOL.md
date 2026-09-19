# Persistent engine protocol v1

Run one process per engine and task. A process loads exactly one model, then
reads one JSON object per stdin line and writes one JSON object per stdout line.
Diagnostics go to stderr; flush every response. Unknown or invalid requests must
fail. One request is in flight at a time. No result/PCM/encoder-output cache
across requests: each request must perform fresh inference with an empty KV.
Reusable weights, tokenizer, work buffers and compiled pipelines are allowed.

The reference adapter wraps the **current Rust implementation** without modifying
its inference code. Threads=0 uses its own `get_num_cpus()` default. The benchmark
fixes offline, batch=1, English language forcing, no silence skip, no segmentation,
no past-text conditioning and no speculative decoding. Greedy ASR has the source
engine's 2048 generated-token limit; a candidate must expose a limit-hit as an
error, not return a truncated successful result. The aligner gets the fixed human
transcript, not the ASR hypothesis. Language forcing only applies to ASR; alignment
uses the source aligner's prompt and word tokenization.

## Ready

```json
{"schema":1,"ready":true,"contract":"qwen-asr-metal-bench-v1","task":"asr","backend":"metal","load_ms":123.4}
```

Task is `asr` or `align`. CPU reference identifies as `cpu`; candidate must identify
as `metal`. Include device, precision/quantization, model artifact hashes, kernel
build identity and supported feature range in the candidate handshake for audit.
No benchmark result may be emitted by a capability probe or mock adapter.

The native candidate also returns `package_path`, `kernel_directory`,
`precision`, and `weights_sha256`. The runner independently hashes the package
index, weights, tokenizer assets, and all four runtime `.metal` sources; the
package's source safetensors/config hashes must equal the CPU model identity.
Generation policy is fixed above rather than read from `generation_config.json`.
Kernel source changes are therefore visible even if the executable is unchanged.

## Request

```json
{"schema":1,"op":"infer","id":"unique-id","task":"asr","audio":"/absolute/input.wav","language":"English"}
```

For `align`, add `"text":"THE FIXED REFERENCE TEXT"`. ASR requests deliberately do
not contain reference text. Input is the complete frozen 16 kHz mono PCM16 WAV.

## Result

```json
{"schema":1,"ok":true,"id":"unique-id","task":"asr","backend":"metal","text":"recognized text","words":[],"elapsed_ms":100.0,"input_ms":0.2,"audio_samples":16000,"sample_rate":16000,"gpu_completed":true,"cpu_tensor_fallbacks":0}
```

For `align`, `words` contains one record per whitespace-delimited reference word:

```json
{"text":"THE","start_ms":0.0,"end_ms":160.0}
```

`elapsed_ms` is monotonic host wall time from **resident PCM before feature
extraction to the final returned text/timestamps**, including prompt processing,
Mel, encoder, prefill, decode/classification, synchronization, readback and
postprocessing. The GPU must finish before stopping the clock. `input_ms` measures
WAV decoding separately. Model loading/pipeline compilation is before readiness;
the first full corpus pass is retained as warmup, excluded from headline timing.
The runner also measures request wall time and requires its own 30% improvement.

Output declarations alone cannot prove execution provenance. A passing timing and
quality result still needs code inspection and a Metal System Trace showing the
neural network and feature extraction execute on GPU with no CPU tensor fallback.

Errors have `ok=false` and an `error` string; exit or timeout also fails the suite.
Missing cases, malformed output, changed PCM lengths and non-finite timings fail
closed. Finite but nonmonotonic/out-of-audio timestamps are retained with explicit
issues, so the rest of the corpus can still be measured. Any such issue in either
engine blocks alignment acceptance pending reference review; it does not justify
silently loosening the gate or dropping the clip. Nondeterministic output fails.

## Scoring and gate

- Exactly 20 cases, frozen PCM SHA256 and human transcript; 20 distinct speakers.
- NFKC + casefold; remove apostrophes, tokenize Unicode alphanumerics; punctuation
  and whitespace are not scored. CER uses the concatenated normalized tokens.
- Corpus WER/CER use total edit counts divided by total reference units. They are
  not unweighted averages of sentence rates. Every sentence must have no increase
  in either error count versus CPU; repeated outputs must be identical.
- Alignment requires identical reference words and **0 ms** CPU timestamp drift.
  This is regression parity, not human-labeled alignment accuracy. An 80 ms
  tolerance must not be introduced without explicitly changing the acceptance
  policy and corpus version. Structural validation allows the final predicted
  time bin to extend at most 80 ms beyond audio; CPU parity still applies.
- Each task separately: median total time across the 20 samples ≤ 0.70 × CPU;
  host request total also ≤ 0.70 × CPU; P95 suite time must not regress.
- P95/median suite-time spread must be ≤ 1.25 in both engines. Otherwise the
  timing run is retained but cannot certify a speedup; rerun without competing
  workload. This threshold is fixed before measuring any GPU implementation.
- Default: 1 warmup + 7 measured rounds. Seeded case order; alternate CPU/GPU order
  per case; no concurrent inference. Report each sample, each round, P50/P95,
  conventional RTF (inference/audio, lower is better), realtime multiple and
  paired bootstrap ratio interval. Seven rounds are a development gate, not a
  population-level latency or accuracy guarantee.
- Freeze the manifest before optimizing. Do not replace difficult cases. This
  English clean-speech smoke suite does not certify all languages/noise conditions.
- Baseline-only output is **not** a GPU success. A comparison passing still has
  `comparison_passed_requires_gpu_audit` status pending execution verification.
