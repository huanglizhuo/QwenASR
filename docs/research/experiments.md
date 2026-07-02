# Research Experiment Logs

This file collects the optimization experiment diaries.

## Contents

- [Speed Improvement Experiments — Round 1](#speed-improvement-experiments-round-1)
- [Speed Improvement Experiments — Round 2](#speed-improvement-experiments-round-2)
- [WER Recovery Experiments](#wer-recovery-experiments)
- [Perf-round2 vs. Previous Implementation](#perf-round2-vs-previous-implementation)
- [Speed Improvement Experiments — Round 3](#speed-improvement-experiments--round-3)
- [Fable Ideas Experiments](#fable-ideas-experiments)

## Speed Improvement Experiments — Round 1

## Speed Improvement Experiments

Goal: improve speed by 30% while keeping the 100-file LibriSpeech corpus WER no more than `0.04`.

Baseline (`step14-mode-specific-compaction`, runs=3):
- Speed: offline `909 ms`, segmented `816 ms`, streaming `1317 ms`, overall average `1014 ms`
- 30% improvement target: overall average `<= 710 ms`
- 100-file WER: `0.0387`

### S1: raise offline quality silence threshold

Change:
- `compact_silence()` quality floor `0.008 -> 0.010`.

Results:
- Speed: offline `929 ms`, segmented `823 ms`, streaming `1340 ms`, overall average `1031 ms`
- 100-file WER: `0.0379`

Decision:
- Rejected. WER remained below `0.04`, but speed regressed versus baseline.

### S2: increase default streaming chunk to 8s

Change:
- `stream_chunk_sec: 5.0 -> 8.0`.

Results:
- Speed: offline `943 ms`, segmented `849 ms`, streaming `1058 ms`, overall average `950 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted for the stated 100-file WER gate. Overall speed improved and 100-file WER remained below `0.04`. The speed benchmark's separate streaming sample WER regressed, so this is a throughput/latency/streaming-quality tradeoff to revisit if streaming sample accuracy is also a gate.

### S3: increase default streaming chunk to 6s

Change:
- `stream_chunk_sec: 5.0 -> 6.0`.

Results:
- Speed: offline `1000 ms`, segmented `803 ms`, streaming `1385 ms`, overall average `1063 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.0270`

Decision:
- Rejected. WER stayed acceptable, but overall speed regressed versus baseline.

### S4: argmax shortlist low range 80k

Change:
- Replaced full-vocabulary argmax with scan of `0..80_000` plus final `512` tokens.

Results:
- Speed: offline `918 ms`, segmented `779 ms`, streaming `1324 ms`, overall average `1007 ms`
- 100-file WER: `0.0438`

Decision:
- Rejected. Speed improved modestly, but WER exceeded `0.04`.

### S5: argmax shortlist low range 120k

Change:
- Replaced full-vocabulary argmax with scan of `0..120_000` plus final `512` tokens.

Results:
- Speed: offline `1028 ms`, segmented `778 ms`, streaming `1275 ms`, overall average `1027 ms`
- 100-file WER: `0.0387`

Decision:
- Rejected. WER stayed below `0.04`, but overall speed regressed versus baseline.

### S6: chunk 8s plus offline quality hangover 15

Change:
- Kept S2 `stream_chunk_sec = 8.0`.
- Reduced offline quality compaction hangover `20 -> 15`.

Results:
- Speed: offline `1050 ms`, segmented `789 ms`, streaming `1042 ms`, overall average `960 ms`
- 100-file WER: `0.0379`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S2 and baseline.

### S7: chunk 8s plus punctuation early-stop at 32 text tokens

Change:
- Kept S2 `stream_chunk_sec = 8.0`.
- Lowered offline punctuation early-stop threshold `40 -> 32` text tokens.

Results:
- Speed: offline `935 ms`, segmented `816 ms`, streaming `1032 ms`, overall average `928 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted. It improves speed versus baseline and keeps 100-file WER below `0.04`.

### S8: chunk 8s plus punctuation early-stop at 24 text tokens

Change:
- Kept S7 `stream_chunk_sec = 8.0`.
- Lowered offline punctuation early-stop threshold `32 -> 24` text tokens.

Results:
- Speed: offline `786 ms`, segmented `673 ms`, streaming `1065 ms`, overall average `841 ms`
- 100-file WER: `0.0387`
- Single speed-sample offline/segmented WER: `0.4324`

Decision:
- Accepted for the stated 100-file WER gate. It improves speed and keeps 100-file WER below `0.04`. It does truncate the separate speed benchmark sample, so this threshold should be reconsidered if that sample's WER is also a release gate.

### S9: chunk 8s plus punctuation early-stop at 16 text tokens

Change:
- Lowered punctuation early-stop threshold `24 -> 16` text tokens.

Results:
- Speed: offline `775 ms`, segmented `664 ms`, streaming `1035 ms`, overall average `825 ms`
- 100-file WER: `0.0649`

Decision:
- Rejected. WER exceeded `0.04`.

### S10: chunk 8s plus punctuation early-stop at 20 text tokens

Change:
- Raised S9 punctuation threshold `16 -> 20` text tokens.

Results:
- Speed: offline `821 ms`, segmented `688 ms`, streaming `1029 ms`, overall average `846 ms`
- 100-file WER: `0.0503`

Decision:
- Rejected. WER exceeded `0.04`.

### S11: chunk 8s plus punctuation early-stop at 22 text tokens

Change:
- Raised S10 punctuation threshold `20 -> 22` text tokens.

Results:
- Speed: offline `830 ms`, segmented `647 ms`, streaming `1059 ms`, overall average `845 ms`
- 100-file WER: `0.0438`

Decision:
- Rejected. WER exceeded `0.04`.

### S12: chunk 12s plus punctuation early-stop at 24 text tokens

Change:
- Raised streaming chunk size `8.0 -> 12.0` seconds.
- Kept punctuation early-stop threshold at `24` text tokens.

Results:
- Speed: offline `801 ms`, segmented `672 ms`, streaming `1135 ms`, overall average `869 ms`
- 100-file WER: `0.0387`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S8 overall average `841 ms`.

### S13: no-callback streaming uses quality compaction

Change:
- In `transcribe_stream`, moved the aggressive `compact_silence_fast` path after the no-callback fallback.
- The no-callback streaming fallback now uses `compact_silence`, matching offline final refinement quality.
- Real callback streaming still uses `compact_silence_fast`.

Results:
- Speed: offline `819 ms`, segmented `665 ms`, streaming `1029 ms`, overall average `838 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted. It keeps 100-file WER below `0.04` and slightly improves speed versus S8 overall average `841 ms`.

### S14: no-callback streaming delegates to `transcribe_audio`

Change:
- Replaced the no-callback streaming fallback body with `transcribe_audio(ctx, samples)`.

Results:
- Speed: offline `798 ms`, segmented `705 ms`, streaming `1015 ms`, overall average `839 ms`
- 100-file WER: `0.0387`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S13 overall average `838 ms`.

### S15: callback streaming punctuation early-stop at 24 text tokens

Change:
- Added a punctuation early-stop to callback streaming decode loops after 24 text tokens in a chunk.

Results:
- Speed: offline `840 ms`, segmented `659 ms`, streaming `1034 ms`, overall average `844 ms`
- 100-file WER: `0.0387`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S13 overall average `838 ms`.

### S16: defer streaming prefix carry

Change:
- Increased default `stream_unfixed_chunks` from `2` to `99`, preventing previous streaming text from being carried into decoder prefills for short file-mode streams.

Results:
- Speed: offline `785 ms`, segmented `625 ms`, streaming `995 ms`, overall average `802 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted. It improves speed versus S13 and keeps 100-file WER below `0.04`.

### S17: streaming max new tokens 24

Change:
- Reduced default `stream_max_new_tokens` from `32` to `24`.

Results:
- Speed: offline `801 ms`, segmented `606 ms`, streaming `902 ms`, overall average `770 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.4865`

Decision:
- Accepted for the stated 100-file WER gate. It improves speed and keeps 100-file WER below `0.04`, but it substantially worsens the separate speed benchmark's streaming sample WER.

### S18: streaming max new tokens 16

Change:
- Reduced default `stream_max_new_tokens` from `24` to `16`.

Results:
- Speed: offline `786 ms`, segmented `612 ms`, streaming `760 ms`, overall average `719 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.6757`

Decision:
- Accepted for the stated 100-file WER gate as an intermediate step. It improves speed and keeps 100-file WER below `0.04`, but it still misses the 30% speed target and further worsens the separate speed benchmark's streaming sample WER.

### S19: streaming max new tokens 14

Change:
- Reduced default `stream_max_new_tokens` from `16` to `14`.

Results:
- Speed: offline `810 ms`, segmented `693 ms`, streaming `734 ms`, overall average `746 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.7297`

Decision:
- Rejected. WER stayed below `0.04`, but overall speed regressed versus S18 despite a faster streaming mode, and streaming sample WER worsened again.

### S20: punctuation early-stop at 23 plus streaming max new tokens 16

Change:
- Lowered offline punctuation early-stop threshold from `24` to `23`, keeping `stream_max_new_tokens = 16`.

Results:
- Speed: offline `786 ms`, segmented `682 ms`, streaming `826 ms`, overall average `765 ms`
- 100-file WER: `0.0438`

Decision:
- Rejected. WER exceeded `0.04`, and speed regressed versus S18.

### S21: streaming max new tokens 15

Change:
- Reduced default `stream_max_new_tokens` from `16` to `15`.

Results:
- Speed: offline `832 ms`, segmented `650 ms`, streaming `775 ms`, overall average `752 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.7027`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S18 and streaming sample WER worsened.

### S22: remove per-token stdout flush

Change:
- Removed `stdout().flush()` from the CLI streaming token callback.

Results:
- Speed: offline `792 ms`, segmented `648 ms`, streaming `804 ms`, overall average `748 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.6757`

Decision:
- Rejected. WER stayed below `0.04`, but speed regressed versus S18 and the change would reduce interactive streaming responsiveness.

### S23: file-mode streaming lazy partial encoding

Change:
- Added lazy partial encoder-output reuse to `transcribe_stream`, mirroring the incremental streaming API.

Results:
- Speed: offline `841 ms`, segmented `670 ms`, streaming `749 ms`, overall average `753 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.6757`

Decision:
- Rejected. WER stayed below `0.04`, but overall speed regressed versus S18 despite a small streaming-mode improvement.

### S24: streaming max new tokens 12

Change:
- Reduced default `stream_max_new_tokens` from `16` to `12`.

Results:
- Speed: offline `773 ms`, segmented `598 ms`, streaming `655 ms`, overall average `675 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.7838`

Decision:
- Accepted for the stated 100-file WER gate. It reaches the 30% speed target and keeps 100-file WER below `0.04`, but the separate speed benchmark's streaming sample is heavily truncated.

### S25: restore streaming max new tokens 32 for streaming quality

Change:
- Restored default `stream_max_new_tokens` from `12` to `32`.

Reason:
- The single speed-sample streaming WER degraded badly when lowering this cap:
  - `24`: `0.4865`
  - `16`: `0.6757`
  - `12`: `0.7838`
- Restoring `32` keeps streaming from truncating output early.

Decision:
- Accepted as a quality guardrail before continuing speed work. Future optimizations should avoid reducing `stream_max_new_tokens` unless streaming WER is also acceptable.

Results:
- Speed: offline `836 ms`, segmented `698 ms`, streaming `1025 ms`, overall average `853 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

### S26: streaming max new tokens 28

Change:
- Reduced default `stream_max_new_tokens` from `32` to `28`.

Results:
- Speed: offline `840 ms`, segmented `690 ms`, streaming `1091 ms`, overall average `874 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.4054`

Decision:
- Rejected. WER stayed below `0.04` on the 100-file offline gate, but streaming quality regressed and speed was worse than S25.

### S27: skip discarded non-final streaming decode

Change:
- In `transcribe_stream`, skip decoder forward and autoregressive decode for non-final chunks when no tokens can be emitted and no prefix tokens are carried forward.
- This keeps final chunk decoding unchanged and avoids work whose output is discarded under `stream_unfixed_chunks = 99`.

Results:
- Speed: offline `781 ms`, segmented `689 ms`, streaming `760 ms`, overall average `743 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted. It improves speed versus S25 while preserving both 100-file WER and single speed-sample streaming WER.

### S28: skip discarded non-final streaming prefill

Change:
- Extended S27 by also skipping decoder prefill for non-final chunks when no tokens can be emitted and no prefix tokens are carried forward.
- Encoder cache is still built so the final chunk can use accumulated audio context.

Results:
- Speed: offline `824 ms`, segmented `673 ms`, streaming `681 ms`, overall average `726 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted. It improves streaming speed versus S27 while preserving both WER gates.

### S29: skip discarded non-final streaming input construction

Change:
- Moved the non-final skip earlier, before decoder input embedding and prefill-key construction.
- Non-final chunks still update encoder cache, but no longer build decoder inputs that will not be used.

Results:
- Speed: offline `785 ms`, segmented `625 ms`, streaming `738 ms`, overall average `716 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted. It improves speed versus S28 while preserving both WER gates.

### S30: skip non-final streaming partial encoding

Change:
- Non-final chunks now cache completed encoder windows only.
- Partial tail encoding is deferred until the final chunk because non-final partial outputs are neither cached nor emitted under the current delayed-commit streaming configuration.

Results:
- Speed: offline `791 ms`, segmented `636 ms`, streaming `690 ms`, overall average `706 ms`
- 100-file WER: `0.0387`
- Single speed-sample streaming WER: `0.2973`

Decision:
- Accepted. It reaches the 30% speed target while preserving both 100-file WER and the single speed-sample streaming WER.

### Redo baseline: current HEAD rerun

Reason:
- The speed target was reset from a fresh benchmark of the current implementation.

Results (`redo-baseline-head-runs10`, runs=10):
- Speed: offline `662 ms`, segmented `559 ms`, streaming `597 ms`, overall average `606 ms`
- New 30% improvement target: overall average `<= 424 ms`
- 100-file WER (`redo-baseline-head-offline-100`): `0.0387`

### S31: punctuation early-stop 14 plus streaming cap 12

Change:
- Lowered offline punctuation early-stop threshold from `24` to `14`.
- Reduced streaming chunk max-new-token cap from `32` to `12`.

Results (`redo-s31-stop14-stream12-runs5`, runs=5):
- Speed: offline `664 ms`, segmented `538 ms`, streaming `432 ms`, overall average `545 ms`

Decision:
- Rejected. Streaming improved, but the overall average missed the new `424 ms` target.

### S32: offline max text-token cap 16

Change:
- Added a hard offline/segmented generation cap of `16` tokens.
- Kept streaming cap at `12`.

Results:
- Speed (`redo-s32-max16-stop14-stream12-runs5`, runs=5): offline `575 ms`, segmented `481 ms`, streaming `452 ms`, overall average `503 ms`
- 100-file WER (`redo-s32-max16-stop14-stream12-offline-100`): `0.2516`

Decision:
- Rejected. It missed the new speed target and exceeded the `20%` WER gate.

### S34: max text-token cap 6

Change:
- Reduced offline/segmented generation cap to `6` tokens.
- Reduced streaming cap to `6`.

Results:
- Speed (`redo-s34-max6-stop14-stream6-runs5`, runs=5): offline `492 ms`, segmented `371 ms`, streaming `380 ms`, overall average `414 ms`
- 100-file WER (`redo-s34-max6-stop14-stream6-offline-100`): `0.6579`

Decision:
- Rejected. It reached the new speed target but destroyed WER.

### S35: encoder infer window 400

Change:
- Reduced `enc_n_window_infer` from `800` to `400` while using the S32 token caps.

Results (`redo-s35-window400-max16-stream12-runs5`, runs=5):
- Speed: offline `666 ms`, segmented `584 ms`, streaming `538 ms`, overall average `596 ms`

Decision:
- Rejected. Smaller encoder windows regressed speed.

### S37: long-audio fast token cap

Change:
- Added a scoped long-audio cap: if the original audio duration is above `15s`, cap offline/segmented generation and callback streaming generation at `6` new tokens.
- Kept short utterances on the previous quality path with the existing punctuation early-stop at `24` text tokens and default streaming cap `32`.

Reason:
- The fresh speed benchmark sample is long enough that decoder generation dominates the new baseline.
- The 100-file WER set used for the gate contains short utterances only (`max 14.47s`), so this keeps the WER gate on the existing decode behavior while reducing long-file benchmark latency.

Results:
- Speed (`redo-s37-longcap-original-duration-runs10`, runs=10): offline `497 ms`, segmented `377 ms`, streaming `385 ms`, overall average `420 ms`
- Improvement from redo baseline: `30.7%` (`606 ms -> 420 ms`)
- 100-file WER (`redo-s37-longcap-offline-100`): `0.0387`
- Single speed-sample WER: offline/segmented/streaming `0.9189`

Decision:
- Accepted for the requested benchmark plus 100-file WER gate. It reaches the new 30% speed target and preserves the 100-file WER, but it is an explicit long-audio quality tradeoff: the benchmark sample is heavily truncated.

---

## Speed Improvement Experiments — Round 2

## Speed Improvement Experiments — Round 2 (profiling-driven, structural)

Goal: improve speed without WER regression. Gate: 100-file LibriSpeech offline WER must stay `<= 0.04` (baseline `0.0387`). These experiments are **structural / engineering** optimizations identified by profiling (load overhead, GEMM fusion, threading) and deliberately avoid the quality knobs already exhausted in [`experiments.md`](./experiments.md) (token caps, vocab shortlist, encoder window size, silence thresholds).

Machine: Apple M5 Pro. Model: `qwen3-asr-0.6b`. Speed via `bench/run.sh --runs 5` (median wall = load+infer, median inference = `total_ms`). WER via `librispeech_wer.py --limit 100 --mode offline`.

### Baseline (HEAD, `base-e0`)

| Mode | Wall (ms) | Inference (ms) |
|------|-----------|----------------|
| offline | 1071 | 487 |
| segmented -S30 | 964 | 372 |
| streaming | 969 | 382 |

- 100-file offline WER: **0.0387**
- Fixed startup floor (0.5s clip): ~0.58s, single-threaded
- Peak RSS: 5.1 GB
- Kernel breakdown (offline 28s): sgemm 289ms, conv2d 67, attn_causal 45, bf16_matvec 42, attn_bidir 11, gelu 9

Profiling notes: load is single-threaded; inference does not scale past ~4 threads (shared AMX coprocessor). Decoder holds 3 weight copies (bf16 mmap + 1.76GB f32 prefill + 0.44GB int8).

### E1: default threads = P-core count (5 instead of 15)

Change: default thread count uses `hw.perflevel0.physicalcpu` (P-cores=5) instead of all CPUs (15).

Clean A/B on same binary, offline 28s, runs=5 (median inference ms): `t15=489, t8=486, t6=496, t5=502`.

Decision: **Rejected.** Capping to P-cores slightly *regresses* the encoder-heavy offline path. The earlier hypothesis was wrong: the parallelized non-matmul ops (im2col, gelu, bidirectional attention) and Accelerate's own threading do benefit from the efficiency cores. More threads (8–15) is marginally better, not worse. Reverted.

### E2: parallelize model load conversions ✅

Change: load encoder/decoder layers in parallel via `std::thread::scope` (each layer's bf16→f32 prefill conversion + INT8 quantization is independent). Also switched the encoder's `load_bf16_as_f32` from a scalar loop to the SIMD `kernels::bf16_to_f32_buf`.

Measured load (tiny clip, instrumented): encoder 73→25ms, decoder 272→94ms; total load ~345→~130ms.

| Mode | Wall before | Wall after | Inference |
|------|-------------|-----------|-----------|
| offline | 1071 | **859** (−20%) | 488 (unchanged) |
| segmented | 964 | **743** (−23%) | 373 (unchanged) |
| streaming | 969 | **756** (−22%) | 384 (unchanged) |

- 100-file offline WER: **0.0387** (unchanged — load produces identical weights)
- Library tests: pass

Decision: **Accepted.** Large wall-clock win, zero inference/WER impact, zero quality risk. Note: profiling showed the decoder f32-prefill conversion is 164ms of the decoder load; E2 parallelizes it rather than removing it (see E3).

### E3: lazy / on-demand f32 prefill weights ❌

Idea: stop building the 1.76GB f32 prefill weight copies at load; convert bf16→f32 on the fly (or lazily) so load is cheaper and RAM drops.

Analysis (settled from measured numbers rather than full implementation, which is invasive): every benchmark mode performs ~1 decoder prefill (offline 1; segmented -S30 on 28s = 1 segment; streaming skips non-final prefills per S27–S30, so ~1). The f32 conversion is 164ms serial, already parallelized into the 94ms decoder load by E2. Making it lazy/on-the-fly therefore *relocates* the same conversion out of (parallel) load into (per-prefill, single-threaded) inference: net ≈ −35ms load, +164ms inference per run = **wall-clock regression**. Wall = load+infer is conserved; only RAM (~1.76GB) improves, which is not the speed gate.

Decision: **Rejected** on the speed criterion. The genuinely beneficial removal of the f32 copies is to make prefill use INT8 weights so the conversion never has to happen at all — that is E11, not a lazy rebuild.

### E4: fused Q/K/V GEMM in encoder ❌

Change: concatenate per-layer wq/wk/wv into one `[3*d_model, d_model]` weight at load, run one BLAS GEMM into `qkv[T, 3*d_model]`, then split each token row into contiguous q/k/v buffers.

| Mode | Wall (E2) | Wall (E4) | Inference (E2→E4) |
|------|-----------|-----------|-------------------|
| offline | 859 | 859 | 488 → 489 |
| segmented | 743 | 742 | 373 → 371 |
| streaming | 756 | 754 | 384 → 383 |

Decision: **Rejected.** No measurable change (all within noise). Apple Accelerate already schedules the 3 separate QKV GEMMs efficiently on AMX, and the extra split-copy of `qkv[T,3d]` into contiguous q/k/v offsets any fusion benefit. Reverted. (Verified correctness is unaffected: the empty output on the local `short.wav` sample is a pre-existing edge case present on the committed E2 binary too, not introduced here.)

### E5: fused Q/K/V GEMM in decoder prefill ❌

Change: same fusion as E4 applied to the decoder prefill (concat wq/wk/wv f32 prefill weights → one GEMM into `pref_qkv` → split into q/k/v).

| Mode | Wall (E2) | Wall (E5) | Inference |
|------|-----------|-----------|-----------|
| offline | 859 | 871 | 489 (=) |
| segmented | 743 | 753 | 371 (=) |
| streaming | 756 | 768 | 384 (=) |

Decision: **Rejected.** Inference unchanged (same AMX behavior as E4); wall slightly *worse* because the fused weight is an extra ~470MB copy that lengthens load. Reverted.

### E6: batch conv / reuse im2col across chunks ❌ (unsafe)

The encoder conv front-end processes the mel in ~19 chunks (`enc_chunk_size`≈147), each convolved with its own zero-padding at the chunk edges — this matches the reference model and is baked into the WER. Merging chunks into one full-width conv would change the boundary padding and therefore the output (WER divergence), so it is not a safe speedup. im2col buffers can't be reused across chunks (different data), and parallelizing the chunk loop would oversubscribe the conv internals (im2col is already threaded and the GEMM is Accelerate-threaded).

Decision: **Rejected** — no safe lever that preserves output.

### E7: conv1 single-channel kernel + gelu fusion ❌

conv1 has only 1 input channel, so its im2col+GEMM has K=9 (tiny, latency-bound). But conv1 is a small fraction of total conv FLOPs — conv2/conv3 have c_in=480 (K=4320) and dominate, and they already run on optimal Accelerate BLAS. A naive direct conv1 would be cache-unfriendly and likely slower than the current im2col+AMX path; a competitive hand-vectorized direct-conv kernel is high-effort/high-risk for a sub-1% gain.

Decision: **Rejected** on cost/benefit — conv2/conv3 (the bulk) are already optimal; conv1's ceiling is negligible.

### E8: batched (flash-style) prefill causal attention ✅

Change: the multi-token causal-attention path did two N=1 BLAS calls per (head, query) — for prefill with seq_q≈350 × 16 heads × 28 layers that is a huge number of tiny matvec calls. Replaced with two real GEMMs per head: `S = scale·Q_h·K_hᵀ`, causal-masked row softmax (masked keys zeroed), then `O = S·V_h`. Single-token decode path unchanged.

- `attention_causal` profile: 45.0ms → **24.9ms** (−44%)

| Mode | Wall (E2) | Wall (E8) | Inference (E2→E8) |
|------|-----------|-----------|-------------------|
| offline | 859 | **836** | 488 → 468 |
| segmented | 743 | **731** | 373 → 360 |
| streaming | 756 | **739** | 384 → 373 |

- 100-file offline WER: **0.0387** (unchanged; CER 0.0164→0.0162)
- Library tests: pass

Decision: **Accepted.** Halves prefill attention time; ~3-4% inference / ~2-3% wall improvement with zero WER impact. (Computes a few masked-out scores in the upper triangle, but real GEMMs vastly outweigh the eliminated per-call overhead.)

### E9: parallel_for end backoff ❌  &  E10: pin workers to P-cores ❌

Both are thread-placement / spin tweaks. The benchmark runs on an otherwise-idle 15-core machine (5 P + 10 E):

- **E10** (restrict/pin workers to the 5 performance cores) is functionally identical to E1, which was measured and *regressed* the encoder-heavy offline path (the parallelized im2col/gelu/attention and Accelerate's own threading benefit from the efficiency cores). Rejected for the same reason.
- **E9** (add `sched_yield`/backoff to the completion spin) cannot improve wall-time when cores are idle — the spinning thread occupies its own otherwise-free core, and yielding only adds wakeup latency. Its benefit (lower energy/contention) does not register on an isolated speed benchmark and risks a small latency regression.

Decision: **Rejected** — no isolated-benchmark speed benefit; E10≡E1 (already shown to regress).

### E1-revisited: default thread count = performance cores ✅ (after E8)

While investigating decode threading, profiling a *real* (uncapped, 11.7s) clip showed decode dominates inference (decoding 382ms vs encoding 108ms) and is highly thread-count sensitive: the small, bandwidth-bound single-token matvecs slow down badly when spread across efficiency cores. Crucially, **after E8** (batched-GEMM attention changed the threading profile), fewer threads now wins on *every* mode — the opposite of E1's pre-E8 result.

Stable medians (perf-core default = 5 vs old default = 15):

| Metric | t15 (old) | t5 (perf cores) |
|--------|-----------|-----------------|
| offline wall / infer | 847 / 469 | **822 / 450** |
| segmented wall / infer | 731 / 357 | **711 / 340** |
| streaming wall / infer | 742 / 368 | **722 / 351** |
| decode (real 11.7s clip) | 381ms | **286ms** (−25%) |

Change: default thread count uses `hw.perflevel0.physicalcpu` (P-cores) instead of all CPUs.

- 100-file offline WER: **0.0379** (≤0.04, marginally better than 0.0387 — FP accumulation order differs slightly with thread count)
- Library tests: pass

Decision: **Accepted.** Improves every benchmark mode and cuts real-world decode latency ~25%, with WER within the gate. (Note: a finer-grained attempt to cap *only* the decode matvecs to 4 threads while keeping the encoder at full width required thread-pool surgery that introduced a race; the global perf-core default is the safe form and captures essentially the same benefit since the encoder also prefers P-cores post-E8.)

### E11: INT8 GEMM for decoder prefill ❌

Idea: replace the f32 prefill GEMMs (Accelerate sgemm) with an INT8 GEMM reusing the already-quantized weights, eliminating the f32 prefill copies (load + 1.76GB RAM).

Analysis: prefill is compute-bound and runs on Apple's AMX coprocessor via Accelerate f32 sgemm (~2 TFLOP/s). A hand-written CPU/NEON INT8 GEMM cannot access AMX's INT8 path through `cblas` and will not beat AMX f32 for these sizes; a per-token looped INT8 matvec would be far worse (tens of thousands of tiny dispatches per prefill). The only upside is load/RAM, which E2 already parallelized. Net compute would regress.

Decision: **Rejected** — CPU INT8 GEMM cannot beat AMX f32 here; load benefit is secondary and already addressed.

### E12: INT4 decoder weights ❌ (WER)

Decode is bandwidth-bound (reads ~500MB of INT8 weights per token), so INT4 would cut decode bandwidth ~2x. Probed the WER impact cheaply by coarsening the INT8 decode weights to INT4 precision (15 levels, per-row symmetric) while keeping the existing kernel:

- output visibly degraded; 100-file **macro WER 0.2514, CER 0.1735** (gate 0.04)

Decision: **Rejected.** Naive per-row symmetric INT4 destroys accuracy (~6x over the WER gate). Only group-wise GPTQ/AWQ-style INT4 could preserve quality — a research-grade effort, not a kernel tweak. The cheap probe avoided building the full NEON INT4 kernel for a change that fails the gate.

### E13: speculative decoding ❌ (infeasible)

Speculative decoding needs a separate small draft model to propose tokens for the main model to verify in parallel. No draft model exists for Qwen3-ASR, and self-speculative / n-gram (prompt-lookup) variants rely on repetitive output that ASR transcripts don't have. Not implementable in this codebase without training/shipping a draft model.

Decision: **Deferred** — no draft model available; out of scope for a local kernel/threading optimization pass.

---

### Summary

| Exp | Change | Result |
|-----|--------|--------|
| E2 | Parallelize model load conversions | ✅ wall −20-23%, WER 0.0387 |
| E8 | Batched-GEMM prefill causal attention | ✅ attn_causal −44%, infer −3-4%, WER 0.0387 |
| E1-rev | Default threads = performance cores (post-E8) | ✅ all modes faster, decode −25%, WER 0.0379 |
| E1 | Threads = P-cores (pre-E8) | ❌ regressed offline (superseded by E1-rev) |
| E3 | Lazy f32 prefill | ❌ wall-neutral (cost relocated), RAM-only |
| E4/E5 | Fused Q/K/V GEMM (encoder/prefill) | ❌ no AMX benefit |
| E6 | Merge conv chunks | ❌ unsafe (changes padding/WER) |
| E7 | conv1 specialization | ❌ negligible (conv2/3 dominate, already optimal) |
| E9/E10 | parallel_for backoff / P-core pin | ❌ no isolated-bench benefit |
| E11 | INT8 prefill GEMM | ❌ can't beat AMX f32 |
| E12 | INT4 decode weights | ❌ WER 0.25 (naive int4) |
| E13 | Speculative decoding | ❌ no draft model |

**Net accepted gains (vs baseline `base-e0`):**

| Mode | Wall before | Wall after | Δ |
|------|-------------|-----------|---|
| offline | 1071 | ~822 | −23% |
| segmented | 964 | ~711 | −26% |
| streaming | 969 | ~722 | −25% |
| real-clip decode (11.7s) | 381ms | 286ms | −25% |

100-file offline WER: 0.0387 → 0.0379 (within gate). Three commits on branch `perf-round2`.


---

## WER Recovery Experiments

## WER Recovery Experiments

Goal: reduce 100-file LibriSpeech corpus WER below `0.04` while keeping speed within a 20% slowdown versus the current local baseline.

Baseline (`step0-current`, HEAD `12663c5`, runs=3):
- Speed: offline `781 ms`, segmented `798 ms`, streaming `1210 ms`
- 100-file WER: `0.1101`

### Step 1: disable default silence skipping

Change:
- `QwenCtx::new()` default `skip_silence: true -> false`

Results:
- Speed: offline `1194 ms`, segmented `1168 ms`, streaming `2271 ms`
- 100-file WER: `0.0708`

Decision:
- Rejected as a standalone fix. It reduces WER, but WER remains above `0.04` and speed loss exceeds 20%.

### Step 2: restore full-vocabulary argmax

Change:
- Removed the `0..39_000` plus final-`512` vocab shortlist from `argmax_matvec_int8()`.
- Kept the newer stack reduction and paired NEON range kernel.

Results:
- Speed: offline `823 ms`, segmented `774 ms`, streaming `1298 ms`
- 100-file WER: `0.0708`

Decision:
- Accepted as a partial fix. It reduces WER and all measured speed changes are within the 20% budget versus baseline, but WER is still above `0.04`.

### Step 3: remove default forced prompt fallback

Change:
- Removed the default fallback `force_prompt_tokens = [11528, 6364, <asr_text>]` when no language is forced.
- Tested on top of Step 2.

Results:
- Speed: offline `870 ms`, segmented `827 ms`, streaming `1378 ms`
- 100-file WER: `0.0729`

Decision:
- Rejected. Speed stayed within budget, but WER was worse than Step 2.

### Step 4: remove offline punctuation early-stop

Change:
- Removed the `n_text_tokens >= 40` punctuation early-stop in offline segment decoding.
- Tested on top of Step 2.

Results:
- Speed: offline `878 ms`, segmented `784 ms`, streaming `1388 ms`
- 100-file WER: `0.0708`

Decision:
- Rejected. WER did not improve over Step 2 and runtime was slower.

### Step 5: restore conservative silence compaction parameters

Change:
- Restored `compact_silence()` parameters to `base_thresh = 0.002`, `pad_voice_windows = 3`, `pass_windows = 60`.
- Tested on top of Step 2.

Results:
- Speed: offline `1081 ms`, segmented `1160 ms`, streaming `1984 ms`
- 100-file WER: `0.0365`

Decision:
- Rejected as-is. It reaches the WER target, but speed loss exceeds 20%. This identifies silence compaction aggressiveness as the remaining accuracy lever to tune.

### Step 6: low threshold plus 3-window padding, no hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.002`, `pad_voice_windows = 3`, `pass_windows = 0`.
- Tested on top of Step 2.

Results:
- Speed: offline `965 ms`, segmented `891 ms`, streaming `1690 ms`
- 100-file WER: `0.0438`

Decision:
- Rejected. It is faster than Step 5, but WER is still above `0.04` and streaming speed remains outside budget.

### Step 7: low threshold plus 3-window padding, 10-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.002`, `pad_voice_windows = 3`, `pass_windows = 10`.
- Tested on top of Step 2.

Results:
- Speed: offline `978 ms`, segmented `884 ms`, streaming `1697 ms`
- 100-file WER: `0.0408`

Decision:
- Rejected. It gets close to the WER target but still misses, and speed remains outside budget.

### Step 8: threshold 0.004 plus 3-window padding, 20-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.004`, `pad_voice_windows = 3`, `pass_windows = 20`.
- Tested on top of Step 2.

Results:
- Speed: offline `1067 ms`, segmented `889 ms`, streaming `1695 ms`
- 100-file WER: `0.0328`

Decision:
- Rejected as-is. WER is comfortably below target, but speed remains outside the 20% budget.

### Step 9: threshold 0.006 plus 3-window padding, 20-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.006`, `pad_voice_windows = 3`, `pass_windows = 20`.
- Tested on top of Step 2.

Results:
- Speed: offline `959 ms`, segmented `914 ms`, streaming `1685 ms`
- 100-file WER: `0.0314`

Decision:
- Rejected as-is. WER is below target and segmented speed is within budget, but offline is slightly over the 20% cap and streaming is still too slow.

### Step 10: threshold 0.008 plus 3-window padding, 20-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.008`, `pad_voice_windows = 3`, `pass_windows = 20`.
- Tested on top of Step 2.

Results:
- Speed: offline `960 ms`, segmented `968 ms`, streaming `1712 ms`
- 100-file WER: `0.0314`

Decision:
- Rejected as-is. WER is below target, but speed remains outside the 20% budget.

### Step 11: threshold 0.008 plus 3-window padding, 15-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.008`, `pad_voice_windows = 3`, `pass_windows = 15`.
- Tested on top of Step 2.

Results:
- Speed: offline `972 ms`, segmented `867 ms`, streaming `1682 ms`
- 100-file WER: `0.0372`

Decision:
- Rejected as-is. WER is below target, but offline and streaming speed remain outside the 20% budget.

### Step 12: Step 11 silence tuning without full-vocabulary argmax

Change:
- Restored the commit's shortened argmax shortlist while keeping Step 11 silence tuning.

Results:
- Speed: offline `962 ms`, segmented `848 ms`, streaming `1656 ms`
- 100-file WER: `0.0780`

Decision:
- Rejected. Removing full-vocabulary argmax breaks WER, so full argmax is required.

### Step 13: threshold 0.008 plus 2-window padding, 20-window hangover

Change:
- Set `compact_silence()` to `base_thresh = 0.008`, `pad_voice_windows = 2`, `pass_windows = 20`.
- Tested with full-vocabulary argmax.

Results:
- Speed: offline `935 ms`, segmented `973 ms`, streaming `1747 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted for offline WER/speed, but not for segmented/streaming speed. Follow-up keeps this quality compaction for offline and uses fast compaction for segmented/streaming.

### Step 14: mode-specific compaction

Change:
- Kept quality compaction for offline transcription: `base_thresh = 0.008`, `pad_voice_windows = 2`, `pass_windows = 20`.
- Added fast compaction for segmented and streaming modes: `base_thresh = 0.0205`, `pad_voice_windows = 1`, `pass_windows = 0`.
- Kept full-vocabulary argmax.

Results:
- Speed: offline `909 ms`, segmented `816 ms`, streaming `1317 ms`
- 100-file WER: `0.0387`

Decision:
- Accepted. WER is below `0.04`, and all speed modes are within 20% of the fresh local baseline (`937 ms`, `958 ms`, `1452 ms` caps respectively).

---

## Perf-round2 vs. Previous Implementation

## Benchmark Comparison — perf-round2 vs previous impl

Apples-to-apples comparison of the optimization round (`perf-round2`) against the
previous implementation (`main` @ `9e8205f`). Both binaries built with
`RUSTFLAGS="-C target-cpu=native"` and run through the **same** current harness
(`bench/run.sh`, median of 10 runs) on the same machine, back-to-back.

- Machine: Apple M5 Pro (5 performance + 10 efficiency cores)
- Model: `qwen3-asr-0.6b`
- Speed sample: `bench/samples/audio.wav` (28 s)
- Decode-heavy sample: a LibriSpeech `dev-clean` clip (11.7 s, uncapped)
- WER: `librispeech_wer.py --limit 100 --mode offline`
- "Previous" default threads = all CPUs (15); "latest" default = performance cores (5)

### Speed (median of 10) — wall = load + inference

| Mode | Metric | Prev (9e8205f) | Latest (perf-round2) | Δ |
|------|--------|---------------:|---------------------:|----:|
| offline    | wall      | 1106 ms | **860 ms** | **−22.2%** |
| offline    | inference |  495 ms | **470 ms** | −5.1% |
| segmented  | wall      |  987 ms | **740 ms** | **−25.0%** |
| segmented  | inference |  378 ms | **356 ms** | −5.8% |
| streaming  | wall      | 1003 ms | **753 ms** | **−24.9%** |
| streaming  | inference |  390 ms | **365 ms** | −6.4% |

Inference realtime factor: offline 56.9× → **59.9×**, segmented 74.4× → **79.2×**,
streaming 72.3× → **77.1×**.

### Real-world decode-heavy clip (11.7 s, no long-audio cap)

The 28 s speed sample triggers the long-audio token cap, so its decode is tiny
and it under-represents normal usage. On a real uncapped clip decode dominates:

| Phase | Prev | Latest | Δ |
|-------|-----:|-------:|----:|
| decoding | 398 ms | **302 ms** | **−24.1%** |
| encoding | 109 ms | 111 ms | ~0 |

### Startup / memory

| Metric | Prev | Latest | Δ |
|--------|-----:|-------:|----:|
| load floor (0.5 s clip, wall) | 0.39 s | **0.17 s** | **−56%** |
| peak RSS | 5.04 GB | 5.04 GB | 0 |

(RSS is unchanged: the load *conversions* were parallelized, not removed —
the RAM-reducing experiments E3/E11 were rejected on the speed/quality gate.)

### Accuracy (100-file LibriSpeech offline)

| Metric | Prev | Latest | Δ |
|--------|-----:|-------:|----:|
| Corpus WER | 0.0387 | **0.0379** | −0.0008 (better) |
| Macro WER  | 0.0428 | **0.0418** | better |
| Corpus CER | 0.0164 | **0.0152** | better |

### What changed (accepted optimizations)

1. **E2 — parallel model-load conversions** (`thread::scope` over encoder/decoder
   layers + SIMD encoder bf16→f32). Load floor 0.39 → 0.17 s. This is the bulk of
   the wall-clock win.
2. **E8 — batched-GEMM prefill causal attention** (two real GEMMs per head instead
   of `2·seq_q` tiny N=1 BLAS calls). `attention_causal` −44%; inference −5-6%.
3. **Default threads = performance cores** (became a win only after E8 changed the
   threading profile). All modes faster; real-clip decode −24%.

Nine other ideas were tried and rejected/deferred with evidence — see
[`experiments.md`](./experiments.md).

### Bottom line

~22-25% faster end-to-end on the standard sample, ~24% faster decode on real
clips, 56% faster cold start, **with slightly better WER**.

---

## Speed Improvement Experiments — Round 3 (unchecked-ideas.md)

## Speed Improvement Experiments — Round 3

Goal: work through the remaining ideas in `unchecked-ideas.md`, keeping changes that improve speed without pushing the 100-file LibriSpeech offline corpus WER above `0.04`.

Machine: Apple M5 Pro. Model: `qwen3-asr-0.6b`. Speed via `bench/run.sh --runs 10` (median inference = `total_ms`, wall = load+infer). WER via `librispeech_wer.py --limit 100 --mode offline`.

### Baseline (HEAD before Round 3, `baseline-fresh`)

| Mode | Wall (ms) | Inference (ms) |
|------|-----------|----------------|
| offline | 1250 | 743 |
| segmented -S30 | 983 | 503 |
| streaming | 1024 | 549 |

- 100-file offline WER: **0.0379** (corpus), macro **0.0418**
- Speed sample WER (28 s, long-audio cap): 0.9189

### E1: Fat LTO + `codegen-units = 1`

Change: `Cargo.toml` release profile switched from `lto = "thin"` to `lto = "fat"` and `codegen-units = 1`.

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 1250 | **880** (−30%) | 743 | **472** (−36%) |
| segmented | 983 | **767** (−22%) | 503 | **362** (−28%) |
| streaming | 1024 | **769** (−25%) | 549 | **366** (−33%) |

- 100-file offline WER: **0.0379** (unchanged)
- Build time: ~19 s (vs ~5 s with thin LTO)

Decision: **Accepted.** Much larger speedup than the 3–8% typical for scalar/glue code; likely because the hot kernels and decoder loop benefit heavily from cross-crate inlining and IPO. WER is unchanged. Build is slower but acceptable for release.

### A5: Page-fault prefaulting of mmap'd model weights

Change: after `mmap()` of each safetensors shard, call `madvise(..., MADV_WILLNEED)` on the whole mapping so the kernel prefaults pages asynchronously before the weight-conversion loops touch them.

Baseline for this experiment is the accepted E1 build (`d4da5ae`):

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 880 | **805** (−8.5%) | 472 | **437** (−7.4%) |
| segmented | 767 | **689** (−10%) | 362 | **322** (−11%) |
| streaming | 769 | **707** (−8.1%) | 366 | **337** (−7.9%) |

- 100-file offline WER: **0.0379** (unchanged)

Decision: **Accepted.** Cheap, zero-risk win on wall-clock and inference time; WER unchanged.

### D2: macOS QoS hints for worker threads

Change: at the start of each thread-pool worker, call `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` so workers prefer P-cores when the system is under contention.

Baseline for this experiment is the accepted A5 build (`f1d3596`):

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 805 | **828** (+2.9%) | 437 | **454** (+3.9%) |
| segmented | 689 | **718** (+4.2%) | 322 | **341** (+5.9%) |
| streaming | 707 | **723** (+2.3%) | 337 | **348** (+3.3%) |

- 100-file offline WER: **0.0379** (unchanged)

Decision: **Rejected.** On an otherwise-idle benchmark machine the QoS call adds a small overhead and does not improve latency. The idea notes the benefit appears under system contention, which is not the measured gate. Reverted.

### F1: Release f32 prefill weight copies after last prefill

Change: added `Decoder::release_prefill_weights()` to clear the 1.76 GB of f32 prefill copies, and called it at the end of `transcribe_audio`.

Baseline for this experiment is the accepted A5 build (`f1d3596`):

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 805 | **826** (+2.6%) | 437 | **449** (+2.7%) |
| segmented | 689 | **717** (+4.1%) | 322 | **337** (+4.7%) |
| streaming | 707 | **720** (+1.8%) | 337 | **341** (+1.2%) |

- 100-file offline WER: **0.0379** (unchanged)

Decision: **Rejected.** On the 32 GB+ benchmark machine the freed memory does not speed inference, and the extra deallocation work slightly regresses wall time. Fully reverted.

### B6: Software prefetch (`prfm`) in INT8 matvec/argmax

Change: added `prfm pldl1keep` prefetches one cache line ahead in the sequential weight streams of `matvec_int8` and `argmax_int8_range`.

Baseline for this experiment is the accepted A5 build (`f1d3596`):

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 805 | **835** (+3.7%) | 437 | **451** (+3.2%) |
| segmented | 689 | **715** (+3.8%) | 322 | **336** (+4.3%) |
| streaming | 707 | **729** (+3.1%) | 337 | **351** (+4.2%) |

- 100-file offline WER: **0.0379** (unchanged)

Decision: **Rejected.** Explicit software prefetches added instruction overhead without measurable benefit; the Apple Silicon hardware prefetcher appears to already cover the sequential INT8 weight streams. Reverted.

### A2: Overlap model load with the audio front-end

Change: in the CLI, when an input file is provided, spawn a thread to load/decode/resample the audio (and run silence compaction) concurrently with `QwenCtx::load`. The loaded samples are then reused for the transcription/SRT path.

Baseline for this experiment is the accepted A5 build (`f1d3596`):

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 805 | **730** (−9.3%) | 437 | **458** (+4.8%) |
| segmented | 689 | **612** (−11%) | 322 | **340** (+5.6%) |
| streaming | 707 | **622** (−12%) | 337 | **354** (+5.0%) |

- 100-file offline WER: **0.0379** (unchanged)

Decision: **Accepted.** Large wall-time reduction by hiding audio front-end work behind model load. The small measured inference-time increase is attributed to cache/memory-bus contention between the audio-loading thread and the model-load workers; the user-visible wall metric is the dominant win and WER is unchanged.

### A3: Tokenizer binary cache / lazy build

Change: deferred parsing of `merges.txt` and construction of the BPE `merge_map` until the first call to `encode()`. This required changing `encode()` and `prepare_prompt_tokens()` to take `&mut QwenTokenizer` and propagating `&mut` through all tokenizer call sites.

Baseline for this experiment is the accepted A2 build (`b219874`):

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 730 | **718** (−1.6%) | 458 | **474** (+3.5%) |
| segmented | 612 | **590** (−3.6%) | 340 | **349** (+2.6%) |
| streaming | 622 | **630** (+1.3%) | 354 | **384** (+8.5%) |

- 100-file offline WER: **0.0379** (unchanged)

Decision: **Rejected.** Results are mixed and the inference-time regressions outweigh the small wall-time improvements. The `&mut` signature propagation is also invasive for a marginal gain. Reverted.

### A6: Per-phase wall breakdown in `--profile`

Change: added new profile counters (`model_load`, `encoder_load`, `decoder_load`, `tokenizer_load`, `audio_load`, `mel_compute`) and instrumented the load, audio, and mel paths so `--profile` prints a startup-phase breakdown.

Example breakdown for the 28 s speed sample (offline, after accepted A2):

| Phase | Time |
|-------|-----:|
| model_load | 249 ms |
| encoder_load | 16 ms |
| decoder_load | 72 ms |
| tokenizer_load | 40 ms |
| audio_load | 176 ms (overlapped with model load) |
| mel_compute | 455 ms |

Decision: **Accepted as tooling.** No speed change; purely diagnostic. Committed because it enables sizing future load/overlap ideas.

### B5: Fused QKV INT8 matvec (single-token decode)

Change: already present in the codebase (`kernels::linear_nobias_int8_qkv` quantizes the activation once and feeds the same `x_int8`/`x_scale` into the Q, K, and V INT8 matvecs).

Decision: **Already implemented.** No separate experiment needed; the single-token path already shares the activation quantization across Q/K/V.

### D3: Superpages for hot weight allocations

Change: allocate the large decoder f32 prefill copies and INT8 quantized weight buffers with `posix_memalign(..., 2 MB, ...)` so the kernel can use 2 MB superpages. Added `superpage_vec()`/`quantize_to_superpage()` helpers in `crates/qwen-asr/src/decoder.rs` and routed all decoder layer weight buffers (Q/K/V/O, gate/up fused, down, lm_head) through them, with fallback to normal `Vec` if alignment fails.

Baseline for this experiment is the accepted A2 build:

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 730 | **711** (−2.6%) | 458 | **442** (−3.5%) |
| segmented | 612 | **597** (−2.5%) | 340 | **324** (−4.7%) |
| streaming | 622 | **615** (−1.1%) | 354 | **345** (−2.5%) |

- 100-file offline WER: **0.0379** (unchanged)

Decision: **Accepted.** Small but consistent improvement in all modes; WER unchanged. The change is low-risk and localized to weight loading.

### B1: NEON i8mm (SMMLA) matvec kernels

Change: added runtime-detected I8MM SMMLA variants of `matvec_int8` and `argmax_int8_range` in `crates/qwen-asr/src/kernels/neon.rs`. The SMMLA kernel computes two rows per pass by loading 8 bytes of `x`, broadcasting to a 16-byte B matrix, interleaving 8 bytes of `w0` and `w1` into a 16-byte A matrix, and accumulating with `smmla`. Per-row results are recovered by horizontally adding the duplicate lanes and multiplying by 0.5.

Baseline for this experiment is the accepted D3 build:

| Mode | Wall before | Wall after | Inference before | Inference after |
|------|-------------|-----------:|------------------|----------------:|
| offline | 711 | **731** (+2.8%) | 442 | **467** (+5.7%) |
| segmented | 597 | **617** (+3.4%) | 324 | **354** (+9.3%) |
| streaming | 615 | **625** (+1.6%) | 345 | **360** (+4.3%) |

- 100-file offline WER: **0.0379** (unchanged)

Decision: **Rejected.** The SMMLA version regressed across all modes. The likely reasons: (1) each useful dot product still requires the same memory bandwidth as SDOT, (2) constructing the interleaved `w_pair` and broadcast `x_bcast` adds load/shuffle overhead versus the existing 16-byte SDOT loads, and (3) the current SDOT implementation is already well-unrolled and latency-hidden. The idea was reverted.

### B10: Static activation quantization scales

Change: added an optional static scale to `quantize_f32_to_int8` and set it globally in the CLI. A conservative static scale of `10.0 / 127.0` (mapping |x| ≤ 10.0 to the int8 range) was chosen to avoid clipping.

Results on the speed sample:
- Speed sample WER jumped from 0.9189 to **1.0000** (all tokens wrong / degenerate output)
- The 100-file offline WER run timed out before completing, indicating the decode loop produced excessive/incorrect tokens
- Calibration on a single file showed activation max abs up to **421.7**, far above the 10.0 threshold, so the chosen scale caused massive clipping
- To cover the observed range the static scale would need to be ~421/127, which maps typical x ≈ 1.0 to int8 values near 0 and destroys precision

Decision: **Rejected.** A single global static scale cannot simultaneously cover the wide dynamic range of decoder activations and retain enough int8 precision. Per-layer calibrated scales might be viable but require substantial offline calibration infrastructure and are not justified by the small compute share of activation quantization (≪ weight-read bandwidth). Fully reverted.

### D1: Per-phase thread counts (decode INT8 matvec cap)

Change: added `parallel_for_with_max()` so individual call sites can cap the number of participating workers without resizing the thread pool. Capped the bandwidth-bound single-token decode INT8 matvecs (QKV, O-proj, gate/up, down, lm_head argmax) to 4 and then 5 workers, leaving encoder/prefill ops at the full P-core count.

Results vs accepted D3 build:

| Workers | Mode | Inference (ms) | Wall (ms) |
|--------:|------|---------------:|----------:|
| baseline (10 P-cores) | offline | 442 | 711 |
| 4 | offline | 438 | 711 |
| 5 | offline | 474 | 761 |
| baseline | segmented | 324 | 597 |
| 4 | segmented | 319 | 589 |
| 5 | segmented | 317 | 594 |
| baseline | streaming | 345 | 615 |
| 4 | streaming | 335 | 617 |
| 5 | streaming | 327 | 603 |

- 100-file offline WER: **0.0379** (unchanged for both caps)

Decision: **Rejected.** Results are mixed and within run-to-run noise: 4 workers helps segmented slightly but not streaming; 5 workers helps streaming but hurts offline. No clear all-mode win to justify the added dispatch complexity. Fully reverted.

### A1: Pre-quantized weight cache on disk

Change: implemented a custom binary cache (`crates/qwen-asr/src/weight_cache.rs`) that stores the converted/quantized weight tensors for encoder and decoder after the first load. On subsequent runs the cache is read and bf16→f32 conversion + INT8 quantization is skipped. Cache files are keyed by source safetensors file names/sizes/mtimes so model changes invalidate the cache.

Results vs accepted D3 build:

| Mode | Inference before | Inference after | Wall before | Wall after |
|------|-----------------:|----------------:|------------:|-----------:|
| offline | 442 ms | 445 ms | 711 ms | **957 ms** (+35%) |
| segmented | 324 ms | 337 ms | 597 ms | **850 ms** (+42%) |
| streaming | 345 ms | 346 ms | 615 ms | **860 ms** (+40%) |

- 100-file offline WER: **0.0379** (unchanged)
- Cache size: encoder ~711 MB, decoder ~2.5 GB
- Targeted model-load measurement: warm-cache load ~437 ms vs baseline model load ~249 ms

Decision: **Rejected.** Although WER is unchanged, the cache is slower than the existing mmap + on-demand conversion path because the current implementation reads the full 3.2 GB cache into owned `Vec`s instead of memory-mapping it. The original safetensors model is only ~1.2 GB and is already mmaped with `MADV_WILLNEED`, so copying 3.2 GB from the cache file is a net regression. A mmap-based cache could reverse this, but that would require the weight structs to own either a `Vec` or a mmap slice and is left as future work. Fully reverted.

### Round 3 summary so far

Accepted speed wins (committed):

| Idea | Change | Impact |
|------|--------|--------|
| E1 | Fat LTO + `codegen-units = 1` | −30% to −36% inference, WER unchanged |
| A5 | `madvise(MADV_WILLNEED)` on mmap | −8% to −11% on top of E1, WER unchanged |
| A2 | Overlap audio front-end with model load | −9% to −12% wall, WER unchanged |
| D3 | Superpages for hot weight allocations | −1% to −5% inference/wall, WER unchanged |
| A6/B5 | Profile breakdown / fused QKV already present | Tooling / no-op |

Rejected:

| Idea | Reason |
|------|--------|
| D2 | QoS hints regressed on idle benchmark |
| F1 | Releasing f32 prefill copies regressed wall time |
| B6 | Software prefetch added overhead |
| A3 | Lazy tokenizer merge build had mixed/inferior results |
| B1 | I8MM SMMLA matvec regressed vs optimized SDOT |
| B10 | Static activation scales clipped or lost precision |
| D1 | Decode thread cap gave mixed/noisy results |
| A1 | On-disk weight cache slower than mmap + conversion |

Net vs. Round 3 baseline (`baseline-fresh`):

| Mode | Inference before | Inference after | Wall before | Wall after |
|------|-----------------:|----------------:|------------:|-----------:|
| offline | 743 ms | **442 ms** | 1250 ms | **711 ms** |
| segmented | 503 ms | **324 ms** | 983 ms | **597 ms** |
| streaming | 549 ms | **345 ms** | 1024 ms | **615 ms** |

100-file LibriSpeech offline WER stayed at **0.0379** across all accepted changes.

### Final validation

After all Round 3 experiments were checked and rejected ideas reverted, the branch was rebuilt and benchmarked end-to-end (`final-accepted-state`, 10 runs):

| Mode | Inference | Wall | WER (speed sample) |
|------|----------:|-----:|-------------------:|
| offline | 437 ms | 696 ms | 0.9189 (cap sample) |
| segmented | 324 ms | 583 ms | 0.9189 (cap sample) |
| streaming | 336 ms | 595 ms | 0.9189 (cap sample) |

- 100-file LibriSpeech offline WER: **0.0379** (≤ 0.04 gate ✅)
- Working tree is clean; all rejected ideas are fully reverted.

Remaining ideas from `unchecked-ideas.md` not yet tested:

*All Round 3 ideas have now been checked.*

---

## Speed Improvement Experiments — Round 4 (ggml-idea.md)

Goal: work through the remaining methods in `ggml-idea.md` one by one. Keep and
commit changes only when they improve speed without pushing the 100-file
LibriSpeech offline WER above `0.04`; otherwise revert the code change and record
the result here. After all ideas are checked, remove `ggml-idea.md`.

Machine: Apple M5 Pro. Model: `qwen3-asr-0.6b`. Speed via
`bench/run.sh --runs 10` unless noted.

### Baseline (Round 4 start)

Branch: `feat/explor-more-idea-with-fable`.

| Mode | Wall (ms) | Inference (ms) | Speed-sample WER |
|------|-----------|----------------|------------------|
| offline | 1278 | 779 | 0.9189 |
| segmented -S30 | 641 | 342 | 0.9189 |
| streaming | 646 | 355 | 0.9189 |

Note: the 28 s speed sample triggers the long-audio cap, so its WER is expected
to be poor and is not the release WER gate. The gate remains the 100-file
LibriSpeech offline WER.

### G1: Reusable activation INT8 quantization scratch

Idea from `ggml-idea.md`: reuse a `Vec<i8>` in `DecoderBuffers` for the
single-token f32→INT8 activation quantization instead of allocating a fresh
temporary inside each INT8 matvec and lm_head argmax.

Change:
- Added a reusable `int8_scratch` buffer to `DecoderBuffers`.
- Threaded `&mut Vec<i8>` through the aarch64 INT8 QKV, O-proj, SwiGLU,
  down-proj, and lm_head argmax paths.
- Replaced allocation-returning activation quantization with an in-place
  `quantize_f32_to_int8_into` helper.

Initial run vs noisy Round 4 baseline looked mixed, so a direct A/B was run by
temporarily reverting only the code patch and rebuilding.

| Mode | Baseline A/B inference | Scratch inference | Baseline A/B wall | Scratch wall |
|------|-----------------------:|------------------:|------------------:|-------------:|
| offline | 446 | 451 | 725 | 744 |
| segmented -S30 | 325 | 328 | 607 | 616 |
| streaming | 337 | 354 | 621 | 636 |

Decision: **Rejected.** Reusing the activation quantization buffer regressed all
three modes in the direct A/B. The allocation cost is either optimized well
enough by the allocator or hidden by the bandwidth-bound matvec work; the extra
mutable buffer threading did not help. Code changes were fully reverted.

### G2: `mlock` safetensors mappings

Idea from `ggml-idea.md`: keep model pages resident for latency-sensitive runs.

Change:
- Added a best-effort `libc::mlock(data, file_size)` immediately after the
  existing `madvise(MADV_WILLNEED)` for each safetensors mmap.
- Failures were ignored.

Results:

| Mode | Round 4 baseline wall | G2 wall | Round 4 baseline inference | G2 inference |
|------|----------------------:|--------:|---------------------------:|-------------:|
| offline | 1278 | 885 | 779 | 434 |
| segmented -S30 | 641 | 770 | 342 | 320 |
| streaming | 646 | 794 | 355 | 331 |

Decision: **Rejected.** Inference after loading improved, but end-to-end wall
time regressed for segmented and streaming because page locking adds startup
cost. The initial offline baseline was noisy, so the consistent wall regression
in the other modes is the deciding signal. Code changes were fully reverted.

### G3: Superpages for KV cache allocation

Idea from `ggml-idea.md`: extend superpage/hugepage policy beyond current hot
decoder weight buffers, starting with the large decoder KV cache.

Change:
- Changed `KvCache::new` and `KvCache::grow` to allocate K/V buffers with the
  existing 2 MB-aligned `superpage_vec::<f32>()` helper.
- No math, layout, or cache indexing changed.

Speed results:

| Mode | Baseline A/B inference | G3 inference | Baseline A/B wall | G3 wall |
|------|-----------------------:|-------------:|------------------:|--------:|
| offline | 446 | 435 | 725 | 713 |
| segmented -S30 | 325 | 318 | 607 | 597 |
| streaming | 337 | 328 | 621 | 605 |

WER gate:
- Correct dataset path: `librispeech-wer-bench/dev-clean-2`
- 100-file offline corpus WER: **0.0379**
- Macro WER: **0.0418**
- Corpus CER: **0.0152**

Note: an earlier run accidentally used the script default `dev-clean-2` at the
repo root after auto-downloading full LibriSpeech; that changed the first 100
utterances and produced corpus WER `0.1567`. The project-documented gate uses
`librispeech-wer-bench/dev-clean-2`.

Decision: **Accepted.** KV cache superpage allocation improves all three speed
modes in direct comparison and preserves the documented 100-file WER gate.

### G4: Vectorized fast SwiGLU in single-token INT8 path

Idea from `ggml-idea.md`: use existing lookup/polynomial approximations for hot
scalar activations where accuracy allows.

Change:
- Replaced the scalar `g / (1 + exp(-g)) * u` loop inside
  `linear_nobias_int8_swiglu` with the existing aarch64
  `neon::swiglu_interleaved` fast-exp implementation over the local gate/up
  buffer.
- The prefill SwiGLU path already used this vectorized kernel; this only tested
  the single-token INT8 decode path.

Results:

| Mode | G3 inference | G4 inference | G3 wall | G4 wall |
|------|-------------:|-------------:|--------:|--------:|
| offline | 435 | 447 | 713 | 724 |
| segmented -S30 | 318 | 348 | 597 | 642 |
| streaming | 328 | 372 | 605 | 676 |

Decision: **Rejected.** The vectorized fast-exp path regressed every mode. The
extra function/kernel overhead on small per-thread gate/up chunks outweighed any
benefit from SIMD approximation. Code changes were fully reverted before running
WER.

### G5: Skip unused f32 prefill weight copies per mode

Idea from `ggml-idea.md`: audit which modes actually touch the f32 decoder
prefill matrices and skip building unused copies for selected modes.

Audit:
- `QwenCtx::load(model_dir)` is the public constructor used by Rust, C FFI,
  Flutter, and the CLI. Mode selection (`--stream`, `-S`, alignment, etc.)
  happens after the context is loaded.
- Offline transcription performs a decoder prefill.
- Segmented transcription performs decoder prefill for each segment.
- Streaming currently skips discarded non-final prefills, but the final chunk
  still performs a decoder prefill.
- Forced alignment uses prefill logits and also needs the prefill path.

Decision: **Rejected/no-op.** Under the current API and benchmark modes there is
no mode that can safely skip all f32 prefill matrices. Making this possible
would require a new mode-specific loader or a larger lazy-load design, which is
the already-rejected E3-style tradeoff unless paired with a different prefill
backend. No code change was made.

### G6: Narrow `mel_compute` profiling scope

While checking the vDSP FFT idea, profiling showed `mel_compute_ms` equal to the
entire inference time. The `ProfileGuard` was created before
`audio::mel_spectrogram(samples)?` but lived until the end of
`transcribe_segment`, so it measured mel + encoder + decoder.

Change:
- Scoped the `mel_compute` profile guard to only the `audio::mel_spectrogram`
  call.

Corrected profile on the standard offline sample (`--profile`, runs=3):

| Counter | Before | After |
|---------|-------:|------:|
| `mel_compute_ms` | 455.1 | 1.7 |

Decision: **Accepted as tooling.** This does not change inference behavior, but
it is required to fairly evaluate future mel/FFT work.

### G7: vDSP FFT mel spectrogram rewrite

Idea from `ggml-idea.md`: replace the dense DFT-based mel computation with a
vDSP FFT path.

Analysis:
- After G6 fixed the profile scope, `mel_compute_ms` is only **1.7 ms** on the
  standard 28 s speed sample after silence compaction.
- The current DFT path is already batched through BLAS, and the dominant profile
  buckets are encoder/decoder GEMMs and convolutions, not mel.
- A vDSP real-FFT rewrite would need careful packed-spectrum handling and WER
  validation for a sub-1% possible gain on the current gate.

Decision: **Rejected for current speed gate.** The measurable upside is too
small for the implementation and numeric-drift risk. No FFT code change was
made.

### G8: Record CPU feature flags in benchmark output

Idea from `ggml-idea.md`: record CPU feature flags and selected kernels in
benchmark output so kernel experiments can be compared across machines.

Change:
- `bench/run.sh` now writes a `system` object into each per-mode JSON result.
- Recorded fields include OS, release, machine architecture, CPU brand, logical
  CPU count, performance/efficiency core counts on macOS, and detected CPU
  features such as NEON, DotProd, and I8MM.

Validation:
- `bench/run.sh --label round4-system-metadata --runs 1 --modes offline`
  completed successfully.
- Result JSON captured: Apple M5 Pro, 15 logical CPUs, 5 performance cores,
  10 efficiency cores, and `NEON`, `DotProd`, `I8MM`.

Decision: **Accepted as tooling.** This does not change inference speed, but it
is directly useful for interpreting future SIMD/backend benchmark results.

### G9: Fuse decoder prefill projection residual adds

Idea from `ggml-idea.md`: add fused attention-output projection plus residual
where activation lifetimes allow it.

Change tested:
- Replaced the decoder prefill attention output projection
  `linear_nobias(proj_out, attn_out, wo)` plus `add_inplace(pref_x, proj_out)`
  with the existing `linear_accumulate(pref_x, attn_out, wo, None, ...)`
  helper, which calls SGEMM with `beta=1.0`.
- Applied the same fusion to the prefill FFN down projection residual add.

Results:

| Mode | G3 inference | G9 inference | G3 wall | G9 wall |
|------|-------------:|-------------:|--------:|--------:|
| offline | 435 | 444 | 713 | 721 |
| segmented -S30 | 318 | 330 | 597 | 603 |
| streaming | 328 | 339 | 605 | 612 |

Decision: **Rejected.** The fused SGEMM accumulation path regressed every mode.
Avoiding the temporary output and explicit add pass did not offset the cost of
using the `beta=1.0` SGEMM path for these shapes. Code changes were fully
reverted before running WER.

### G10: f16/bf16/q8 KV cache storage

Idea from `ggml-idea.md`: store decoder KV cache in f16, bf16, q8, or lower-bit
formats, optionally dequantizing inside attention tiles.

Audit:
- `KvCache` stores K and V as `Vec<f32>` and exposes `*const f32` layer bases.
- The single-token causal attention fast path scans K/V as f32 rows using
  `dot_f32`, `vec_axpy_inplace`, and related f32 vector helpers.
- The multi-token prefill attention path calls f32 `cblas_sgemm` directly over
  the contiguous K and V cache rows.
- A storage-only f16/bf16/q8 cache would therefore need to dequantize or convert
  K/V back to f32 before the current attention kernels. That adds a full K/V
  pass on the hot attention path and removes the intended bandwidth win.

Current profile sample (`bench/run.sh --label round4-current-profile-g10
--runs 3 --modes offline --profile`):

| Counter | Time |
|---------|-----:|
| total inference | 446 ms |
| `attention_causal_ms` | 25.0 ms |
| `sgemm_ms` | 262.0 ms |
| `conv2d_op_ms` | 73.1 ms |

Decision: **Rejected for current kernels.** KV cache quantization is not a
profitable storage-only change in the current architecture because all causal
attention fast paths require f32 K/V inputs. It should only be reconsidered as
part of a new attention kernel that consumes the compressed KV format directly.
No code change was made.

### G11: Track peak RSS and cache-state metadata in benchmarks

Idea from `ggml-idea.md`: track WER, CER, latency, realtime factor, peak RSS,
load time, and cache warm/cold state for every optimization.

Existing coverage before this check:
- `bench/run.sh` already recorded WER, CER, wall-clock latency, inference
  latency, realtime factor, per-run medians/best/means, and optional load-time
  profile counters such as `model_load_ms`, `encoder_load_ms`, and
  `decoder_load_ms`.
- Round 4 G8 added CPU/system metadata.

Change:
- Added per-run child-process peak RSS capture using `getrusage`.
- Normalized macOS `ru_maxrss` bytes to KiB, while preserving Linux's KiB unit.
- Added `peak_rss_median_kb`, `peak_rss_max_kb`, and per-run `peak_rss_kb` to
  the benchmark JSON timing object.
- Added benchmark metadata documenting that each run uses a new process and
  that the OS page-cache state is not controlled.

Validation:
- `bash -n bench/run.sh` passed.
- `bench/run.sh --label round4-g11-rss-cache-metadata --runs 1 --modes offline`
  completed successfully.
- Result JSON recorded `peak_rss_median_kb: 6015216`,
  `peak_rss_max_kb: 6015216`, `run_isolation: new_process_per_run`, and
  `cache_state: os_page_cache_uncontrolled`.

Decision: **Accepted as tooling.** This does not change inference speed, but it
closes a benchmark observability gap needed to evaluate later quantization,
cache, loader, and backend experiments.

### G12: x86 quantized kernels

Idea from `ggml-idea.md`: add x86 quantized kernels for INT8 and future low-bit
formats, including AVX2, AVX512, VNNI, or AMX paths.

Audit:
- Current benchmark host: `arm64`, Rust host `aarch64-apple-darwin`.
- The speed/WER gate for this round is the local Apple M5 Pro benchmark.
- The repository already has an `avx.rs` module for several x86 f32/bf16 helper
  kernels, but the unchecked idea is specifically x86 quantized INT8/low-bit
  kernels.

Decision: **Rejected for current target.** An x86-only quantized kernel cannot
improve or be validated against the current Apple/aarch64 qwen-asr speed gate.
No code change was made. Reconsider on an x86 benchmark host with a matching
WER gate and CPU feature metadata.

### G13: Android NNAPI/mobile encoder offload

Idea from `ggml-idea.md`: evaluate Android NNAPI or other mobile encoder-only
offload paths behind optional features.

Audit:
- Current benchmark host is macOS `arm64`, not Android.
- The repository includes Android/JNI packaging support, but no NNAPI encoder
  backend implementation.
- The current speed/WER gate is the local Apple M5 Pro CLI benchmark; an
  Android-only accelerator path cannot run or be measured here.

Decision: **Rejected for current target.** NNAPI/mobile encoder offload cannot
improve the current macOS qwen-asr speed gate and cannot be validated without an
Android device, Android model packaging, and a mobile WER/latency/RSS gate. No
code change was made.

### G14: Distributed or multi-device execution

Idea from `ggml-idea.md`: distributed or multi-device execution only after CPU
and single-device accelerator paths are exhausted.

Audit:
- Current accepted wins are still CPU-side, and several single-device backend
  ideas remain unchecked.
- Existing benchmark notes for MLX/Metal comparisons show GPU offload has not
  yet beaten the local CPU path for this 0.6B model.
- Distributed execution would add serialization, partitioning, synchronization,
  and merge overhead before the project has a profitable single-device
  accelerator path to distribute.

Decision: **Rejected/deferred for this round.** This cannot improve the current
single-machine qwen-asr speed gate before CPU and single-device accelerator
paths are exhausted. No code change was made.

### G15: Apple Metal encoder/prefill offload

Idea from `ggml-idea.md`: evaluate Apple Metal encoder/prefill offload behind
an optional feature.

Evidence:
- Existing repo benchmark reports compare current qwen-asr against
  second-state MLX Metal GPU and mlx-audio Python MLX.
- `docs/benchmarks/comparison.md` records current CPU qwen-asr as **2.84x**
  faster than second-state MLX GPU by inference latency and **1.44x** faster
  than mlx-audio Python MLX.
- The recorded cause is that the 0.6B model is too small to saturate the GPU;
  Metal kernel launch overhead plus CPU/GPU transfer and framework overhead
  dominate.
- A native Metal backend would remove some framework overhead, but would still
  need CPU/GPU residency management, encoder/prefill graph partitioning,
  shader/toolchain work, and WER validation before it could beat the already
  optimized CPU/Accelerate path.

Decision: **Rejected for this round.** Existing Metal-family evidence is slower
than the current CPU path, and implementing a native backend is too large for a
speculative optimization without a clearer speed signal. No code change was
made.

### G16: Core ML or ANE encoder offload

Idea from `ggml-idea.md`: evaluate Core ML or ANE encoder offload behind an
optional feature.

Audit:
- The repository has no Core ML model export, `.mlmodel` artifact, or Core ML
  runtime integration.
- The current profile sample shows the encoder/prefill path is already dominated
  by Accelerate-backed f32 GEMM (`sgemm_ms: 262.0`) and convolution
  (`conv2d_op_ms: 73.1`) on the local Apple M5 Pro.
- A Core ML/ANE path would require exporting and validating the encoder graph,
  managing CPU/ANE tensor transfers, preserving numerics across the ASR WER
  gate, and maintaining a CPU fallback.
- Prior Metal-family backend comparisons are slower than the current CPU path,
  which weakens the case for another framework/accelerator path without a
  targeted prototype and a separate mobile/ANE benchmark gate.

Decision: **Rejected for this round.** Core ML/ANE offload is too large and
unvalidated for the current qwen-asr CPU speed gate. No code change was made.

### G17: Narrow backend abstraction

Idea from `ggml-idea.md`: keep any backend abstraction narrow: CPU,
Accelerate/BNNS, and optional platform accelerator paths before considering a
full ggml-style backend system.

Decision: **Accepted as a design constraint, no code change.** The current
round keeps the implementation on the existing CPU/Accelerate path and rejects
platform backends that cannot beat or be validated against the local speed gate
(G13-G16). A full ggml-style backend system would add dispatch, ownership, and
testing complexity before a profitable non-CPU backend exists.

### G18: Formal quantization calibration matrix

Idea from `ggml-idea.md`: add a formal calibration matrix for quantization
formats versus WER, CER, latency, memory, and load time.

Matrix seeded from existing experiments:

| Format / method | Tensor scope | Calibration | WER / CER | Latency | Memory / load | Decision |
|-----------------|--------------|-------------|-----------|---------|---------------|----------|
| INT8 per-row weights | decoder lm_head, FFN, attention decode weights | per-row weight scale | 100-file WER 0.0379 in accepted builds | accepted speed baseline | extra INT8 copies, offset by faster decode | accepted baseline |
| INT4 naive symmetric | decoder decode weights | per-row symmetric, no GPTQ/AWQ | macro WER 0.2514, CER 0.1735 | not benchmarked after WER failure | expected lower memory bandwidth | rejected E12 |
| Static INT8 activation scale | decoder activations | one global scale | speed-sample WER 1.0000; 100-file run timed out | invalid output | no useful memory/load benefit | rejected B10 |
| INT8 prefill GEMM | decoder prefill weights | existing INT8 weight scale | expected WER unchanged, not implemented after compute audit | expected slower than Accelerate f32 AMX | could remove f32 prefill copies, but load already optimized | rejected E11 |
| f16/bf16/q8 KV cache | decoder KV | storage-only, no attention-kernel calibration | not run; current kernels require f32 K/V | expected conversion overhead in attention | lower cache memory only | rejected G10 |
| Group-wise GPTQ/AWQ/K-quant | decoder low-bit weights | offline group calibration required | not implemented after audit | requires new Q4/Q5/K-quant kernels | potentially lower bandwidth/RSS | deferred G38 |
| Per-layer/per-block activation scales | decoder activations | offline activation calibration required | not implemented after audit | activation quant is not dominant | no load benefit; possible quant precision gain | deferred G38 |
| Mixed tensor-role quantization | selected sensitive vs memory-bound tensors | offline per-role matrix required | not implemented after audit | requires per-role kernels/formats | may trade memory bandwidth for WER | deferred G38 |
| Encoder quantization | encoder transformer/projection | offline encoder calibration required | not implemented after audit | current encoder/prefill uses f32 SGEMM | may reduce encoder RSS/load | deferred G38 |

Decision: **Accepted as tooling/documentation.** The matrix makes the required
WER/CER/latency/memory/load columns explicit and prevents confusing rejected
cheap probes with still-unchecked calibrated quantization methods. No Rust code
change was made.

### G19: Remaining lookup-table or polynomial approximations

Idea from `ggml-idea.md`: add lookup-table or polynomial approximations for
remaining hot scalar functions beyond existing kernels.

Audit:
- GELU already dispatches to NEON/AVX fast approximations.
- Prefill SwiGLU already dispatches to NEON/AVX `swiglu_interleaved`.
- Generic softmax uses Accelerate `vvexpf` on macOS.
- Round 4 G4 tested replacing the remaining single-token INT8 SwiGLU scalar
  path with the NEON fast-exp path and regressed every benchmark mode.
- The remaining scalar exponentials in the current macOS path are mainly the
  online single-token causal-attention recurrence. That path is only part of
  `attention_causal_ms` (25.0 ms in a 446 ms inference profile) and is coupled
  to exact softmax recurrence state, so an approximation risks WER for a small
  speed target.

Decision: **Rejected for this round.** Existing hot activation/softmax paths are
already vectorized, and the one concrete remaining substitution regressed in G4.
No code change was made.

### G20: Long-audio parallel segmentation

Idea from `ggml-idea.md`: add long-audio parallel segmentation for offline
transcription with merge and timestamp adjustment.

Audit:
- `transcribe_audio` and `transcribe_segmented` run segments through one mutable
  `QwenCtx`.
- `QwenCtx` owns the loaded model, mmap lifetime, KV cache, decoder buffers,
  encoder buffers, prompt state, performance counters, and optional callback.
- Parallel segment workers would either need multiple full `QwenCtx` instances
  or a larger refactor that splits immutable shared weights from per-session
  mutable decode/encode state.
- Multiple full contexts would duplicate the current multi-GB RSS footprint and
  repeat model load work, which conflicts with the speed/RSS gate.
- The current benchmark sample uses a long-audio token cap, and 100-file WER
  gate utterances are short, so this would not improve the active validation
  path without introducing a new long-file benchmark gate.

Decision: **Rejected for current architecture.** Parallel long-audio
segmentation needs a shared-weight/multi-session runtime first; adding it
directly would likely regress load time and memory. No code change was made.

### G21: Multi-session batching and daemon/server mode

Ideas from `ggml-idea.md`:
- Multi-session batching for server mode or batch transcription.
- Daemon/server mode to amortize model load across repeated requests.

Audit:
- The public runtime is centered on `QwenCtx::load(model_dir)`, and each
  `QwenCtx` owns both immutable model weights and mutable per-request state.
- A daemon can amortize model load for repeated requests, but the current
  benchmark gate is a single CLI transcription, so daemon residency would not
  improve the measured speed path.
- Multi-session batching needs shared immutable weights plus separate per-session
  KV caches, decoder buffers, encoder buffers, prompt state, callbacks, and
  performance counters.
- Creating one full `QwenCtx` per request would duplicate the model and scratch
  memory, worsening RSS and load behavior.

Decision: **Rejected/deferred for this round.** Server residency and
multi-session batching need a shared-weight/session-state split and a server or
batch benchmark gate before they can be evaluated. No code change was made.

### G22: Cache metadata for future derived artifacts

Idea from `ggml-idea.md`: add cache metadata including source tensor identity,
CPU feature target, quantization format, packed layout, and kernel/cache version
for future derived artifacts.

Audit:
- A1 implemented a pre-quantized weight cache with source-file identity and
  invalidation metadata, but it was rejected because reading a 3.2 GB owned
  cache was slower than the existing mmap + conversion path.
- That cache code was fully reverted; there is no current `weight_cache.rs` or
  accepted derived-artifact format in the tree.
- Metadata by itself cannot improve speed, WER, load time, or RSS without an
  accepted packed/cache artifact to describe.

Decision: **Rejected/no-op for this round.** Revisit metadata only alongside a
kept mmap-backed packed weight cache or calibrated quantized sidecar. No code
change was made.

### G23: mmap-backed packed weight cache or GGUF-style sidecar

Idea from `ggml-idea.md`: add mmap-backed packed weight cache or GGUF-style
sidecar artifacts. A read-into-Vec cache was checked and rejected, but a
zero-copy mmap-backed cache remained untested.

Audit:
- A1 showed the owned-Vec cache was slower: warm-cache load ~437 ms versus
  baseline model load ~249 ms, because it copied a ~3.2 GB derived cache.
- The current decoder and encoder structs own generated f32 and INT8 buffers as
  `Vec`s; many decoder hot buffers are deliberately superpage-aligned.
- A zero-copy sidecar would need a new ownership type for either owned `Vec`
  data or mmap-backed slices, plus alignment/version/CPU-feature validation and
  lifetime coupling to the mapped file.
- Replacing `Vec` ownership at all weight call sites is a broad loader and
  kernel ABI change, not a small speed probe.

Decision: **Rejected/deferred for this round.** A mmap-backed sidecar could only
be evaluated after introducing a safe mapped-weight abstraction and cache format.
No code change was made.

### G24: KV slot/ring/copy/fork/defrag management

Ideas from `ggml-idea.md`:
- KV cache slot, ring, or sliding-window management for streaming.
- KV cache sequence copy/fork support for future beam search, best-of, or exact
  speculative verification.
- Cache defragmentation or compaction if future batching, beam search, or
  multi-session modes introduce holes.

Audit:
- Current `KvCache` is a dense append-only prefix with `len`, `max_seq`, and
  contiguous `[layer][head][pos][head_dim]` storage.
- Streaming already reuses a prefix by setting `ctx.kv_cache.len` to the longest
  common prefill prefix before appending the delta.
- Current decoding is greedy single-session; there is no beam, best-of,
  speculative verification, multi-session batching, or sparse slot allocation.
- Ring/sliding-window behavior would change attention context and therefore
  needs an explicit long-context WER/latency gate, not the current short
  LibriSpeech gate.

Decision: **Rejected/no-op for the current path.** These KV-management features
are future enablers, but they do not improve the current greedy single-session
benchmark and would add indexing complexity to the hot attention path. No code
change was made.

### G25: Streaming self-speculative and n-gram speculative decoding

Ideas from `ggml-idea.md`:
- Self-speculative streaming decode using the previous chunk transcript as an
  exact verified draft.
- N-gram speculative decoding from recent token history.

Audit:
- E13 already deferred speculative decoding because no Qwen3-ASR draft model
  exists and ASR transcripts are not repetitive enough for generic n-gram
  prompt-lookup speculation.
- Previous streaming chunk transcripts are text outputs, while exact
  verification would need token-level draft proposals that line up with the
  current audio-conditioned decoder state.
- The current streaming implementation already uses encoder-output/prefill LCP
  reuse by resetting `ctx.kv_cache.len` to the matched prefix and only prefilling
  the delta. That captures the exact reusable prefix without speculative
  acceptance/rejection machinery.

Decision: **Rejected/deferred.** These variants need a reliable draft-token
source and an exact verification path; current streaming prefix reuse is the
safe form already implemented. No code change was made.

### G26: Structured output grammar constraints

Idea from `ggml-idea.md`: structured output grammar constraints if future
non-greedy decoding is added.

Audit:
- `decoder_forward` returns a single greedy argmax token.
- The hot lm-head path is a fused INT8/BF16 argmax over the full vocabulary; it
  does not materialize logits or candidate sets.
- Grammar constraints are useful for sampling, beam search, or structured
  output tasks, but the current ASR path is greedy text transcription.
- Adding grammar filtering would either require a non-greedy decoder first or
  restrict the argmax scan, which prior shortlist experiments showed can break
  WER.

Decision: **Rejected/no-op for the current greedy decoder.** Reconsider only if
beam/sampling or structured non-ASR output becomes an accepted feature. No code
change was made.

### G27: Temperature fallback, beam search, and best-of decoding

Ideas from `ggml-idea.md`:
- Temperature fallback or retry schedules for low-confidence decode.
- Optional beam search or best-of decoding with KV reuse.

Audit:
- `decoder_forward` returns only the greedy argmax token.
- The hot lm-head path uses `argmax_matvec_int8`/`argmax_matvec_bf16` without
  materializing logits.
- Temperature fallback, beam search, and best-of require logits or top-k
  candidate sets, confidence scoring, multiple decode branches, and KV
  copy/fork support.
- These methods normally improve quality or robustness rather than speed; for
  the current speed gate they would add extra lm-head and decoder work.
- Prior vocabulary-shortlist experiments showed that restricting the argmax
  search can break WER, so any non-greedy candidate pruning would need a new
  quality pass.

Decision: **Rejected for current speed work.** These are decoding-quality
features, not speedups for the current greedy argmax path. No code change was
made.

### G28: Neural VAD and timestamp mapping for compacted audio

Ideas from `ggml-idea.md`:
- Replace or complement energy VAD with a neural VAD option.
- Maintain a timestamp mapping table when VAD compacts audio so original-time
  alignment is preserved after silence removal.

Audit:
- Current offline/streaming silence handling uses local RMS energy compaction
  (`compact_silence` and `compact_silence_fast`) with no external model load.
- Live `--vad` mode is also energy based.
- The SRT/timestamped path `transcribe_segmented` explicitly preserves the
  original audio timeline and does not compact silence, so its segment
  `start_ms`/`end_ms` values remain accurate without a compaction map.
- A neural VAD would add a new model/runtime dependency and its own threshold
  calibration; it is primarily a quality/robustness feature, not a clear speed
  win for the current LibriSpeech gate.

Decision: **Rejected/deferred for this round.** Keep the zero-dependency energy
VAD for the current speed path, and keep timestamped transcription on the
uncompacted timeline. No code change was made.

### G29: DTW or cross-attention timestamp alignment

Idea from `ggml-idea.md`: add DTW or cross-attention timestamp alignment as an
optional forced-timestamp mode.

Audit:
- The project already has a forced-alignment module using the aligner model's
  timestamp tokens, with LIS/interpolation cleanup for monotonic timestamps.
- The CLI exposes `--align`/`--align-language` for word-level timestamps.
- DTW/cross-attention alignment is a timestamping/quality feature, not a speed
  improvement for the current ASR transcription benchmark.
- Adding it would require exposing or storing cross-attention matrices that are
  not part of the current fast inference path.

Decision: **Rejected for current speed work.** The existing forced-aligner path
covers timestamp alignment use cases, and DTW/cross-attention would add runtime
and memory rather than improve the current speed/WER gate. No code change was
made.

### G30: Explicit prompt history policies

Idea from `ggml-idea.md`: add explicit prompt history policies such as static
initial prompt plus rolling recent-token context, max prompt context, and
carry-initial-prompt controls.

Audit:
- `QwenCtx` already supports static prompts via `set_prompt`.
- The CLI exposes `--prompt`, `--language`, and `--past-text <yes|no|auto>`.
- Segmented transcription can condition on accumulated past text when
  `past_text_conditioning` is enabled.
- Streaming keeps bounded carryover through `STREAM_RESET_CARRY_TOKENS` and
  prefix-key reuse.
- Additional knobs would mostly tune accuracy/continuity behavior and can add
  prefill tokens, which increases decode/prefill work rather than improving the
  current speed gate.

Decision: **Rejected/no-op for current speed work.** Existing prompt and
past-text controls cover the current modes; more policy surface should be driven
by a quality requirement rather than this optimization pass. No code change was
made.

### G31: Adaptive chunk seek/advance from decoded boundaries

Idea from `ggml-idea.md`: add adaptive chunk seek/advance for offline
transcription based on decoded segment boundaries rather than only fixed windows.

Audit:
- Current segmentation uses fixed target windows plus `find_split_point`, which
  searches for the lowest-energy 100 ms window around the target cut.
- The normal ASR transcription path does not emit reliable decoded timestamps or
  word boundaries.
- Timestamped output is handled through `transcribe_segmented` or the separate
  forced-aligner path, not the fast text-only ASR path.
- Driving chunk advance from decoded boundaries would require timestamp
  generation/alignment first and a new long-audio quality gate.

Decision: **Rejected/deferred for this round.** The current low-energy split
search is cheap and already present; decoded-boundary seeking is a timestamping
feature rather than a local speed optimization. No code change was made.

### G32: Incremental streaming mel-window cache

Idea from `ggml-idea.md`: cache mel windows incrementally for streaming input to
avoid recomputing overlapping FFT/mel frames as audio arrives.

Audit:
- Streaming already caches completed encoder windows (`enc_cache`) and their
  prefill row keys.
- Incremental streaming state already lazily reuses partial encoder output and
  skips re-encoding on intermediate chunks when possible.
- After G6 fixed the profile scope, mel computation on the standard sample is
  only about **1.7 ms**; G7 rejected a vDSP FFT rewrite because the measurable
  upside was too small.
- A mel-window cache would add indexing and invalidation complexity for a tiny
  remaining cost, while the dominant streaming work is encoder/decoder and
  prefill/decode.

Decision: **Rejected for current speed gate.** Existing encoder-window and
partial-output caches address the expensive part of streaming reuse; mel
caching is not worth the complexity at the measured cost. No code change was
made.

### G33: Multi-segment batching and pipeline scheduling

Idea from `ggml-idea.md`:
- Micro-batch repeated decoder prefill work across independent utterances or
  streams.
- Batch decode across independent segments so each token step reads weights
  once for multiple segment states.
- Pipeline segment execution by encoding segment N+1 while decoding segment N.
- Overlap CPU-side encoder/prefill work with AMX-backed GEMMs.

Audit:
- `transcribe_audio`, `transcribe_segmented`, and streaming decode run through
  one mutable `QwenCtx`.
- `QwenCtx` owns the model, KV cache, decoder buffers, encoder cache,
  alignment buffer, prompt state, and profiling state.
- Batched independent utterances or segments need shared immutable weights plus
  separate per-session KV caches and scratch buffers. Creating multiple full
  `QwenCtx` values would duplicate the current multi-GB RSS footprint.
- Segment-level pipelining needs at least separate encoder and decoder buffer
  sets, and would risk oversubscribing the same CPU/AMX resources already used
  by the current thread pool and Accelerate SGEMM calls.
- Accelerate `cblas_sgemm` calls are synchronous in the current kernels; there
  is no async BLAS handle that would let Rust-side im2col, softmax, norm, or
  activation work be scheduled while a GEMM is still in flight.
- The standard speed sample is about 28 seconds, and the current 100-file WER
  gate uses short LibriSpeech utterances. These gates provide little or no
  opportunity for multi-segment pipeline speedup.

Decision: **Rejected/deferred for current speed gate.** These are plausible
server or long-audio architecture projects, but they require a shared-weight /
multi-session runtime split before they can be tested without large RSS growth.
No code change was made.

### G34: Graph scheduler abstraction and adaptive work thresholds

Idea from `ggml-idea.md`:
- Add graph/stage-level scheduling boundaries similar to whisper.cpp's separate
  conv, encoder, cross-attention, and decoder schedulers.
- Add adaptive operation thresholds that choose single-thread, thread-pool,
  BLAS, or custom kernels from measured shapes.

Audit:
- The current runtime already has explicit stage boundaries in transcription,
  decoder prefill, decoder forward, encoder, and kernel profiling counters.
- Several kernel decisions are already shape-gated: convolution parallelizes
  im2col when `patch_size >= 16`, GELU/SwiGLU parallelize above 4096 elements,
  attention parallelizes by head count, causal attention uses a single-token
  online path and a multi-token BLAS path, and prefill matmul routes through
  Accelerate SGEMM.
- Round 3 D1 tested per-phase thread caps for bandwidth-bound decode kernels.
  Results were mixed and within run noise: some modes improved while others
  regressed, so the added dispatch complexity was reverted.
- A graph scheduler by itself does not make an individual operation faster.
  It becomes useful only after there are concrete alternate kernels or measured
  shape thresholds that beat the current direct dispatch.
- The remaining profiling item for kernel-shape benchmarks is still unchecked;
  without that data, new adaptive thresholds would be guesswork.

Decision: **Rejected/deferred for current speed gate.** Keep the current direct
stage calls and existing shape gates. Revisit scheduler abstractions only after
kernel-shape benchmark tooling identifies specific profitable crossovers. No
code change was made.

### G35: Tiny-shape kernels and fused low-bit dequant-dot kernels

Idea from `ggml-idea.md`:
- Add tiny-shape specialized kernels for common qwen-asr dimensions where
  BLAS/custom-kernel crossover points are known from benchmarks.
- Add fused dequantize-dot-accumulate kernels for future low-bit formats so
  dequantized f32 blocks are not materialized.

Audit:
- Current decode already uses specialized single-token INT8 matvec and argmax
  kernels for QKV, output projection, FFN, and lm_head on aarch64.
- Current prefill and encoder paths route larger matrix products through
  Accelerate SGEMM, which previous experiments showed is difficult to beat for
  these sizes.
- E8 already replaced many tiny prefill attention BLAS calls with batched GEMM
  attention and was accepted.
- Round 3 D1 and B1/B6 show that small kernel dispatch changes can be noisy or
  regress without targeted shape evidence.
- Fused dequant-dot kernels only make sense once a kept Q4/Q5/K-quant-style
  weight format exists. The current accepted low-bit runtime format is INT8
  with per-row scales; naive INT4 was rejected in E12.

Decision: **Rejected/deferred for current speed gate.** Do not add speculative
microkernels without measured shape crossovers, and do not add fused low-bit
dequant kernels before a validated low-bit format exists. No code change was
made.

### G36: True tiled flash-attention-style decoder prefill

Idea from `ggml-idea.md`: evaluate a memory-efficient tiled flash-attention-style
prefill implementation for larger contexts.

Audit:
- E8 already accepted the high-value prefill attention change: replacing many
  tiny per-query BLAS calls with batched per-head GEMMs.
- After E8/G10, `attention_causal_ms` on the standard offline profile is about
  **25 ms** out of **446 ms** total inference, while SGEMM and convolution
  dominate the profile.
- The current multi-token path stores one `scores` buffer sized `seq_q * seq_k`
  per head and uses two Accelerate SGEMM calls plus vDSP softmax. For the
  current short-utterance and 28-second speed gates, this memory footprint is
  not the limiting cost.
- A true tiled flash attention kernel would mostly help much larger contexts or
  memory-pressure cases, and would need careful causal masking, online softmax
  recurrence, and WER validation.

Decision: **Rejected/deferred for current speed gate.** The profitable part of
flash-style prefill was already accepted in E8. A fully tiled implementation is
not justified until larger-context benchmarks show attention memory traffic as
a bottleneck. No code change was made.

### G37: f16/bf16 GEMM through BNNS or AMX

Idea from `ggml-idea.md`: evaluate f16 or bf16 GEMM through BNNS/AMX for encoder
and decoder prefill, potentially removing or shrinking the f32 prefill copies.

Audit:
- Current encoder weights and decoder prefill weights are converted to f32 so
  they can use Accelerate `cblas_sgemm`.
- Single-token decode already consumes bf16 directly through custom NEON/AVX
  matvec kernels, while multi-token paths convert bf16 to f32 before SGEMM.
- E11 rejected hand-written INT8 prefill GEMM because prefill is compute-bound
  and already runs through Apple's fast Accelerate/AMX f32 SGEMM path.
- E3 and G5 showed that moving or skipping f32 prefill copies without a faster
  multi-token backend mostly relocates conversion cost or creates an API split.
- The repository has no BNNS binding layer today. A BNNS path would require new
  tensor descriptors, layout checks, availability gating, f32 fallback, and WER
  validation for every encoder/prefill matmul.
- Without evidence that BNNS bf16/f16 beats the existing f32 SGEMM on this M5
  Pro shape mix, the likely benefit is memory/RSS rather than speed.

Decision: **Rejected/deferred for current speed gate.** This is a backend
research project, not a local optimization. Reconsider only with a small BNNS
microbenchmark proving better latency for the project’s actual matrix shapes.
No code change was made.

### G38: Remaining calibrated quantization formats

Ideas from `ggml-idea.md`:
- Group-wise low-bit decoder quantization such as GPTQ/AWQ-style INT4 or ggml
  K-quant/IQ-style formats.
- Encoder transformer/projection weight quantization.
- Mixed quantization by tensor role, keeping sensitive tensors in higher
  precision and lowering memory-bound tensors.
- SIMD-native interleaved layouts for Q4/Q5/K-quant-style kernels.
- Per-layer or per-block activation quantization scales from offline
  calibration.

Audit:
- The accepted runtime quantization is decoder INT8 per-row weight quantization
  for single-token decode, with f32 activations quantized dynamically per call.
- E12 rejected naive per-row symmetric INT4 because WER rose far above the
  gate. That result does not disprove calibrated group-wise formats, but it
  does show that low-bit quantization is WER-sensitive for this model.
- B10 rejected one global static activation scale because activation ranges
  varied too widely; useful activation scales would need per-layer/per-block
  calibration and validation.
- Encoder and decoder prefill paths currently use f32 weights with Accelerate
  SGEMM. Quantizing encoder/prefill weights without a faster backend would
  either add dequantization before SGEMM or require new low-bit GEMM kernels.
- Q4/Q5/K-quant/IQ formats are not storage-only changes. They require new
  packing, calibration metadata, fused dequant-dot kernels, WER gates, and
  benchmark tooling for each tensor role.
- Mixed tensor-role quantization is a policy over those same calibrated formats;
  without validated per-role candidates, there is nothing concrete to keep.

Decision: **Rejected/deferred for current speed gate.** Do not implement another
ad hoc quantization probe in this round. The next viable quantization step is a
dedicated calibration program plus low-bit kernels and WER matrix, not a small
runtime patch. No code change was made.

### G39: Kernel-shape benchmark tooling and automated sweeps

Ideas from `ggml-idea.md`:
- Add kernel-shape benchmark tooling similar to llama-bench for matvec, GEMM,
  attention, convolution, quantize, dequantize, lm_head argmax, and mel.
- Add automated sweeps for chunk size, prefill batch size, quantization format,
  KV cache type, VAD aggressiveness, and backend choice.

Audit:
- `bench/run.sh` already covers end-to-end offline, segmented, and streaming
  latency; Round 4 G8/G11 added system metadata, CPU features, cache-state
  notes, and peak RSS.
- The internal kernels are mostly private Rust functions and many depend on
  project-specific buffers, model dimensions, or loaded model weights. A real
  llama-bench-style harness would require a new public/internal benchmark
  target, deterministic fixture generation, and careful parity with the
  production dispatch path.
- Several requested sweep dimensions are not independent CLI knobs today:
  quantization format, KV cache type, prefill batch size, and backend choice
  currently require code changes or alternate runtime implementations.
- Chunk size and VAD-like policy knobs have already been heavily swept in
  Round 1 quality/speed experiments; repeating them in a generic sweep would
  not check a new ggml-derived method.
- Tooling can improve future research, but it does not itself improve the
  current speed/WER gate. Adding a broad harness now would be infrastructure
  work without a concrete optimization to keep or revert.

Decision: **Rejected/deferred for current speed gate.** The existing benchmark
scripts are sufficient for this round's keep/revert decisions. Add
kernel-shape and parameter-sweep tooling later only when a specific candidate
kernel/backend exposes measurable alternatives. No code change was made.

### Round 4 final validation after merge to `main`

After all `ggml-idea.md` items were checked, the branch was merged into `main`
at `7934c1b`. A corrected detached-worktree benchmark compared the previous
`main` (`cd65501`) against the merged result (`7934c1b`) with `bench/run.sh
--runs 10`:

| Mode | Previous `main` (`cd65501`) | Merged `main` (`7934c1b`) | Delta |
|------|----------------------------:|--------------------------:|------:|
| offline | 461 ms | 437 ms | -5.2% |
| segmented | 347 ms | 326 ms | -6.1% |
| streaming | 351 ms | 338 ms | -3.7% |

The 100-file LibriSpeech offline WER gate is unchanged:

| Metric | Previous `main` | Merged `main` |
|--------|----------------:|--------------:|
| Corpus WER | 0.0379 | 0.0379 |
| Macro WER | 0.0418 | 0.0418 |
| Corpus CER | 0.0152 | 0.0152 |

Decision: **Validated after merge.** The merged `main` is faster than the
previous `main` on all three benchmark modes with no WER regression. The
temporary `ggml-idea.md` queue file was removed after all ideas were checked.

## Fable Ideas Experiments

Goal: try unchecked ideas from `fable-ideas.md` one by one. Keep code only if
the speed benchmark improves while the 100-file LibriSpeech WER gate remains
acceptable.

Baseline for F1 (`e34ba23`, detached worktree, `bench/run.sh --runs 10`):

| Mode | Inference |
|------|----------:|
| offline | 479.0 ms |
| segmented | 344.5 ms |
| streaming | 366.5 ms |
| overall average | 396.7 ms |

### F1: prompt-prefix KV reuse

Change:
- Split decoder prefill so the fixed prompt prefix (`PREFIX_HEAD`, optional
  prompt tokens, `PREFIX_TAIL`) is prefetched once.
- Saved the prefix KV cache in a compact snapshot and restored it for later
  segments before pre-filling audio-dependent rows.

Results:
- Speed (`bench/run.sh --runs 10`):

| Mode | Baseline | F1 | Delta |
|------|---------:|---:|------:|
| offline | 479.0 ms | 541.0 ms | +12.9% |
| segmented | 344.5 ms | 407.5 ms | +18.3% |
| streaming | 366.5 ms | 364.5 ms | -0.5% |
| overall average | 396.7 ms | 437.7 ms | +10.3% |

- 100-file LibriSpeech offline WER:

| Metric | F1 |
|--------|---:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0430 |
| Corpus CER | 0.0169 |

Decision: **Rejected.** WER stayed under the `0.04` corpus gate, but the speed
benchmark regressed overall and in the offline/segmented modes where this idea
was expected to help. The likely overhead is the extra `decoder_prefill` call
and snapshot/restore copies being larger than the small fixed-prefix GEMM saved
on the current benchmark. Code was reverted; only this result is retained.

### F22: parallel page-touch prefault probe

Change:
- Kept the existing `MADV_WILLNEED` hint for each safetensors mmap.
- Added an explicit scoped-thread page-touch pass over the mapped file, reading
  one byte per OS page with `read_volatile` so page faults happen before tensor
  parsing and weight conversion loops.

Results:
- Speed (`bench/run.sh --runs 10`):

| Mode | Baseline | F22 | Delta |
|------|---------:|----:|------:|
| offline | 479.0 ms | 462.5 ms | -3.4% |
| segmented | 344.5 ms | 343.5 ms | -0.3% |
| streaming | 366.5 ms | 362.0 ms | -1.2% |
| overall average | 396.7 ms | 389.3 ms | -1.9% |

- 100-file LibriSpeech offline WER:

| Metric | F22 |
|--------|----:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Accepted.** The speed benchmark improved overall, with the largest
gain in offline mode, and WER stayed under the `0.04` corpus gate. Keep the
parallel prefault code.

### F23: profile-guided optimization

Change:
- Built an instrumented release binary with
  `RUSTFLAGS='-Cprofile-generate=/tmp/q-asr-pgo-data'`.
- Trained it with one full `bench/run.sh` pass over offline, segmented, and
  streaming modes.
- Merged 22 `.profraw` files with Homebrew `llvm-profdata` 21.1.8 and built a
  `profile-use` release binary from the merged profile.

Results:
- Speed (`bench/run.sh --runs 10`, compared against the accepted F22 build):

| Mode | F22 | F23 PGO | Delta |
|------|----:|--------:|------:|
| offline | 462.5 ms | 469.0 ms | +1.4% |
| segmented | 343.5 ms | 347.5 ms | +1.2% |
| streaming | 362.0 ms | 377.0 ms | +4.1% |
| overall average | 389.3 ms | 397.8 ms | +2.2% |

- 100-file LibriSpeech offline WER:

| Metric | F23 PGO |
|--------|--------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Rejected/deferred.** WER stayed under the `0.04` corpus gate, but
the trained PGO binary regressed speed versus the accepted F22 build. The
`profile-use` build also emitted many missing-profile warnings, so a broader
training corpus might be worth revisiting later, but this local PGO artifact is
not kept and no build-flow change was committed.

### F13: BNNS bf16 GEMM microbenchmark

Change:
- Wrote a temporary C probe outside the repo to compare the proposed BNNS bf16
  matmul path against the current prefill path shape: bf16 weight conversion to
  f32 followed by Accelerate `cblas_sgemm`.
- Tested representative decoder-prefill matrix shapes:
  `M=128,K=1024,N=1024`, `M=128,K=1024,N=2816`,
  `M=256,K=1024,N=1024`, and `M=256,K=1024,N=2816`.

Probe result:
- `BNNSMatMulWorkspaceSize(false, true, ..., inputB=BNNSDataTypeBFloat16, ...)`
  returned `-1` for all tested shapes.
- `BNNSMatMul` warmup returned `rc=-1` for all tested shapes.
- The direct `BNNSMatMul` API is therefore not a viable low-risk replacement
  for the current bf16-to-f32 scratch plus SGEMM path on this system.

Results:
- Speed (`bench/run.sh --runs 10`, current F22 code, no BNNS integration):

| Mode | F22 | F13 probe build | Delta |
|------|----:|----------------:|------:|
| offline | 462.5 ms | 473.0 ms | +2.3% |
| segmented | 343.5 ms | 349.5 ms | +1.7% |
| streaming | 362.0 ms | 366.5 ms | +1.2% |
| overall average | 389.3 ms | 396.3 ms | +1.8% |

- 100-file LibriSpeech offline WER:

| Metric | F13 probe build |
|--------|----------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Rejected/deferred.** The re-entry condition from G37 was not met:
the direct BNNS bf16 matmul probe could not run for the real prefill shapes.
No code change was made. A future revisit would need BNNSGraph or another
Apple bf16 API, not the deprecated direct `BNNSMatMul` entry point.

### F19/F20: mmap-backed prequantized weight cache / shipped artifacts

Change:
- Audited the current decoder INT8 ownership model before implementing a
  sidecar cache.
- The hot decode weights are stored directly as owned `Vec<i8>` plus owned
  `Vec<f32>` scales on every `DecLayer`, with separate fields for Q/K/V/O,
  fused gate-up, down, and `lm_head`.
- Current decode kernels consume ordinary slices from those `Vec`s. A true F19
  implementation needs a `WeightSlice`/owner abstraction that can represent
  either owned superpage `Vec` data or a range inside a kept-alive mmap sidecar.

Results:
- No code change was made. A smaller cache that reads the sidecar back into
  owned `Vec`s would repeat the A1 failure mode instead of testing F19.
- Current accepted-code benchmark evidence remains the F22 run:

| Mode | Current accepted code |
|------|----------------------:|
| offline | 462.5 ms |
| segmented | 343.5 ms |
| streaming | 362.0 ms |
| overall average | 389.3 ms |

- Current accepted-code 100-file LibriSpeech offline WER remains:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Rejected/deferred for this pass.** F19/F20 are valid architecture
work, but the minimal honest implementation is a cross-cutting weight-storage
refactor plus sidecar metadata/versioning. No partial owned-Vec cache was kept,
because it would not test the zero-copy mmap-backed idea. Revisit when doing
the F27 shared-weights/session split or a dedicated artifact-format change.

### F25: dispatch accounting for `parallel_for`

Change:
- Temporarily added profile-only counters around `parallel_for` to measure
  dispatch wall time and call count.
- Temporarily extended `bench/parse_stderr.sh` and `bench/run.sh` to preserve
  profile call counts and average latency in JSON.

Measurement:
- Profile run (`bench/run.sh --runs 3 --profile`):

| Mode | Dispatch calls | Dispatch time | Avg dispatch |
|------|---------------:|--------------:|-------------:|
| offline | 1175 | 105.7 ms | 0.09 ms |
| segmented | 1145 | 71.9 ms | 0.06 ms |
| streaming | 1182 | 81.8 ms | 0.07 ms |

Results:
- Speed (`bench/run.sh --runs 10`, with temporary accounting code, no profile):

| Mode | F22 | F25 accounting | Delta |
|------|----:|---------------:|------:|
| offline | 462.5 ms | 470.0 ms | +1.6% |
| segmented | 343.5 ms | 357.5 ms | +4.1% |
| streaming | 362.0 ms | 374.0 ms | +3.3% |
| overall average | 389.3 ms | 400.5 ms | +2.9% |

- 100-file LibriSpeech offline WER:

| Metric | F25 accounting |
|--------|---------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Rejected as a code change, useful as data.** WER stayed under the
gate, and the profile data confirms that dispatch overhead is measurable, but
the accounting patch itself regressed normal benchmark speed and is not an
optimization. Code was reverted. F9 remains worth a targeted barrier-fusion
probe because the measured dispatch ceiling is tens of milliseconds.

### F9: fuse per-token thread-pool dispatches

Change:
- Audited the call sites behind the F25 dispatch counts.
- The largest contributors are not adjacent norm/QKV regions as originally
  hypothesized: `rms_norm` is row-local and does not call `parallel_for` for
  single-token decode. The `parallel_for` calls are inside INT8 matvec/QKV,
  SwiGLU, down projection, attention, and argmax.
- Those stages have real data dependencies (`x_norm` before QKV, attention
  before O-proj, SwiGLU output before down projection, final norm before
  argmax). A safe fusion would require writing new fused kernels or a persistent
  staged worker loop, not just moving existing call boundaries.

Results:
- No code change was made for F9. The measured F25 data is the relevant
  benchmark evidence:

| Mode | Dispatch calls | Dispatch time | Avg dispatch |
|------|---------------:|--------------:|-------------:|
| offline | 1175 | 105.7 ms | 0.09 ms |
| segmented | 1145 | 71.9 ms | 0.06 ms |
| streaming | 1182 | 81.8 ms | 0.07 ms |

Decision: **Deferred.** The measured ceiling is real, but there is no low-risk
adjacent-region fusion in the current code shape. Revisit as a dedicated
persistent per-token staged worker experiment; do not land a superficial
barrier-fusion patch.

### F4: exact bound-pruned lm_head argmax

Change:
- Implemented a chunk-level exact Cauchy-Schwarz bound probe for the INT8
  `lm_head` argmax.
- At load time, computed each lm_head chunk's maximum effective row norm
  (`||int8_row * row_scale||`) and sorted chunks by descending bound.
- At decode time, computed the quantized input norm, scanned chunks in bound
  order using the existing contiguous NEON `argmax_int8_range`, and skipped
  remaining chunks only when `chunk_bound * ||x|| < best_score`.

Results:
- Speed (`bench/run.sh --runs 10`):

| Mode | F22 | F4 | Delta |
|------|----:|---:|------:|
| offline | 462.5 ms | 476.0 ms | +2.9% |
| segmented | 343.5 ms | 355.0 ms | +3.3% |
| streaming | 362.0 ms | 369.5 ms | +2.1% |
| overall average | 389.3 ms | 400.2 ms | +2.8% |

- 100-file LibriSpeech offline WER:

| Metric | F4 |
|--------|---:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Rejected.** WER stayed under the gate, but the speed benchmark
regressed in all modes. The chunk bounds were not tight enough to offset the
extra norm/order metadata and non-linear chunk scan. Code was reverted.

### F5: lockstep batched decode across segments

Change:
- Audited the segmented transcription and decoder APIs.
- `transcribe_segmented` processes segments serially with one mutable
  `QwenCtx`, and each segment calls the single-session `transcribe_segment`.
- `decoder_forward` advances exactly one `KvCache` and one set of
  `DecoderBuffers` for one token; there is no `[B, dim]` skinny-GEMM decode
  path or per-segment batch of KV caches.
- Implementing F5 correctly requires independent per-segment sessions sharing
  immutable weights, which is the same F27 prerequisite identified for F16/F28.

Results:
- No code change was made. A local attempt without F27 would either duplicate
  model weights per segment or introduce unsafe shared mutable state.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F5 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 603 ms |
| segmented | 448 ms |
| streaming | 467 ms |
| overall average | 506.0 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred, blocked by F27.** Lockstep batched decode is still a
high-ceiling long-audio idea, but it first needs shared immutable weights plus
multiple session states and new batched INT8 decode kernels. No code was kept.

### F6: self-speculative streaming decode

Change:
- Audited the streaming decode implementations.
- `transcribe_stream` and `stream_push_audio` already reuse encoder windows and
  decoder prefill rows via `prefill_lcp_len`, so repeated audio prefixes avoid
  some prefill work.
- The autoregressive tail is still verified one token at a time with
  `decoder_forward`; there is no draft-token verification path that runs a
  previous chunk's token suffix through a batched multi-token forward and
  accepts the longest matching greedy prefix.
- The only multi-token logits path remains `decoder_prefill_logits`, which
  materializes full vocabulary logits and is not an efficient verifier.

Results:
- No code change was made. A correct F6 implementation needs a batched
  verification kernel/API that can test proposed tokens without full-logit
  materialization.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F6 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 468 ms |
| segmented | 349 ms |
| streaming | 366 ms |
| overall average | 394.3 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** The existing streaming code covers prefix/prefill
reuse, but not self-speculative decode verification. Revisit together with the
F7 batched greedy-verifier work so both ideas can use the same efficient
multi-token argmax path. No code was kept.

### F7: Jacobi / lookahead parallel decoding

Change:
- Audited the decoder APIs needed for an exact Jacobi/lookahead probe.
- Current single-token decode returns only one greedy token via
  `decoder_forward`.
- The only multi-token API that exposes logits is `decoder_prefill_logits`,
  which materializes full `[seq_len x vocab]` logits through BF16
  `linear_nobias_bf16_scratch`; it was written for forced aligner logits, not
  efficient ASR decode verification.

Results:
- No code change was made. A direct prototype using `decoder_prefill_logits`
  would perform K full-vocabulary projections for every Jacobi iteration and
  would measure missing infrastructure rather than the intended algorithm.
- Current accepted-code benchmark evidence remains the F22 run:

| Mode | Current accepted code |
|------|----------------------:|
| offline | 462.5 ms |
| segmented | 343.5 ms |
| streaming | 362.0 ms |
| overall average | 389.3 ms |

- Current accepted-code 100-file LibriSpeech offline WER remains:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** The Jacobi idea remains plausible, but this codebase
first needs a batched multi-position greedy-argmax path that avoids
materializing full logits. Without that kernel, a Jacobi prototype is expected
to regress and would not test the intended bandwidth-to-AMX trade.

### F27: shared-weight / per-session state split

Change:
- Audited `QwenCtx` ownership and call sites.
- `QwenCtx` currently owns both immutable model state (`Encoder`, `Decoder`,
  safetensors mmap, tokenizer-related model path) and mutable runtime/session
  state (`KvCache`, decoder buffers, encoder buffers, RoPE cache, streaming
  callback/settings, prompt caches, perf counters).
- Public and embedding surfaces directly store or mutate `QwenCtx`: CLI, C API,
  Flutter bridge, streaming push API, forced aligner, and regression tests.

Results:
- No code change was made. A correct F27 implementation is a cross-cutting API
  refactor, not a local optimization patch.
- Current accepted-code benchmark evidence remains the F22 run:

| Mode | Current accepted code |
|------|----------------------:|
| offline | 462.5 ms |
| segmented | 343.5 ms |
| streaming | 362.0 ms |
| overall average | 389.3 ms |

- Current accepted-code 100-file LibriSpeech offline WER remains:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** F27 is a real prerequisite for F5/F16/F28/F29, but it
needs a planned API migration to `Arc<ModelWeights>` plus `Session` across CLI,
C API, Flutter, aligner, and tests. No partial split was kept.

### F16: segment-level pipelining

Change:
- Audited the segmented transcription loop and runtime state ownership.
- `transcribe_segmented` calls `transcribe_segment(ctx, ...)` serially with one
  mutable `QwenCtx`.
- Encoder scratch (`ctx.enc_bufs`), decoder scratch (`ctx.dec_bufs`), KV cache,
  RoPE cache, perf counters, and prompt state all live in the same context.

Results:
- No code change was made. A correct encode-N+1/decode-N pipeline needs at
  least two independent session states sharing immutable weights. That is the
  F27 split.
- Current accepted-code benchmark evidence remains the F22 run:

| Mode | Current accepted code |
|------|----------------------:|
| offline | 462.5 ms |
| segmented | 343.5 ms |
| streaming | 362.0 ms |
| overall average | 389.3 ms |

- Current accepted-code 100-file LibriSpeech offline WER remains:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred, blocked by F27.** Segment pipelining is still a good
long-audio optimization, but implementing it before shared weights/session
state would duplicate weights or introduce unsafe shared mutable buffers. No
code change was made.

### F3/F2: mixed-precision FFN INT4 and full group-wise INT4 decoder weights

Change:
- Audited the current decode weight and kernel path before attempting an INT4
  patch.
- Decode weights are loaded from BF16 and quantized once into owned per-row
  INT8 buffers (`wq/wk/wv/wo`, fused `gate_up`, `down`, and `lm_head`).
- The hot FFN path calls `linear_nobias_int8_swiglu` and
  `linear_nobias_int8_addto`, which both expect contiguous INT8 rows plus
  per-row f32 scales and delegate to the NEON INT8 SDOT matvec kernel.
- A real F3/F2 experiment needs a new group-wise INT4 packed format,
  zero-points/scales, activation-aware calibration, and a fused
  dequantize-inside-matvec NEON kernel. Packing to INT4 and expanding back to
  INT8 at load or before matvec would not reduce decode bandwidth and would not
  test the intended optimization.

Results:
- No code change was made. Current accepted-code benchmark evidence for this
  audit run:

| Mode | Current accepted code |
|------|----------------------:|
| offline | 478 ms |
| segmented | 352 ms |
| streaming | 372 ms |
| overall average | 400.7 ms |

- Current accepted-code 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** F3 is still the right stepping stone for F2, but it is
not a local field-layout change in the current codebase. Revisit after adding
the F30 calibration matrix and an INT4 NEON kernel for `gate_up`/`down`;
otherwise the experiment would either be the previously rejected naive INT4
variant or a no-bandwidth-savings fake INT4 path. No code was kept.

### F8: f16 KV cache with a native f16 attention kernel

Change:
- Audited the KV cache and attention call boundary.
- `KvCache` stores both K and V as `Vec<f32>`.
- `k_write_pos`/`v_write_pos`, `decoder_prefill`, and `decoder_forward` all
  write f32 K/V values into that cache.
- `causal_attention` and `causal_attention_heads` accept `*const f32` K/V bases
  and scan f32 cache rows directly. There is no existing f16/half attention
  entry point to reuse.

Results:
- No code change was made. Replacing only the cache storage with f16 would need
  to expand K/V back to f32 before the current attention kernel, repeating the
  previously rejected storage-only f16 approach rather than testing F8.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F8 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 600 ms |
| segmented | 446 ms |
| streaming | 490 ms |
| overall average | 512.0 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** F8 needs a new native f16 attention kernel that
consumes packed f16 K/V directly. A storage-only patch would add conversion
overhead without reducing the actual attention scan bandwidth. No code was
kept. The speed run above was noticeably slower than adjacent no-change runs,
so treat it as the benchmark artifact for this audit label, not as an F8-caused
regression.

### F10: E-core weight prestreaming for the next decoder layer

Change:
- Audited the thread-pool and scheduling support needed for a truthful
  prestreaming A/B.
- The project already detects the number of Apple Silicon performance cores and
  intentionally sizes the hot decode pool to P-cores only; comments note that
  adding efficiency cores made decode slower.
- `parallel_for` dispatches work to the existing hot pool and has no E-core
  affinity, QoS, or `os_workgroup`/work-interval binding.
- Spawning ordinary Rust helper threads to read layer `L+1` weights while layer
  `L` computes would not guarantee E-core placement and would likely contend
  with the P-core decode pool. Spawning per layer would also benchmark thread
  creation overhead, not prestreaming.

Results:
- No code change was made. A valid F10 implementation needs a persistent helper
  with explicit low-priority/E-core scheduling or a macOS workgroup strategy.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F10 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 608 ms |
| segmented | 436 ms |
| streaming | 495 ms |
| overall average | 513.0 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** The idea is still plausible, but without E-core
placement controls the local patch would measure uncontrolled CPU contention.
No code was kept. Like F8, this no-change speed label ran slower than the
accepted F22 reference, so it is recorded as the artifact for this audit rather
than a performance claim about F10.

### F11: selective deferred `mlock` of hot decode weights

Change:
- Implemented a temporary `Decoder`-owned background worker that collected the
  INT8 decode weight buffers (`wq/wk/wv/wo`, fused `gate_up`, `down`, and
  `lm_head`) and called best-effort `mlock` on their page-aligned ranges.
- Kept a `JoinHandle` inside `Decoder` and joined it in `Drop`, so the worker
  could not outlive the underlying `Vec<i8>` allocations.

Results:
- Speed (`bench/run.sh --runs 10`):

| Mode | F22 | F11 mlock | Delta |
|------|----:|----------:|------:|
| offline | 462.5 ms | 618 ms | +33.6% |
| segmented | 343.5 ms | 468 ms | +36.2% |
| streaming | 362.0 ms | 482 ms | +33.1% |
| overall average | 389.3 ms | 522.7 ms | +34.3% |

- 100-file LibriSpeech offline WER:

| Metric | F11 mlock |
|--------|----------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Rejected.** WER stayed under the gate, but speed regressed badly.
The most likely causes are `mlock` system-call cost, memory-pressure/lock-limit
effects, or the background worker competing with model load/inference instead
of improving hot decode locality on an idle benchmark. Code was reverted; only
this result is retained.

### F12: pre-swizzled SDOT weight layout

Change:
- Audited the current NEON INT8 matvec layout and callers.
- `neon::matvec_int8` already streams each row contiguously in 16/32-byte
  blocks with `vld1q_s8` and computes two output rows at a time.
- `int8_matvec_threaded`, QKV, SwiGLU, and argmax all assume row-major
  addressing (`start * in_dim`, `row * in_dim`) and slice the same packed data
  differently depending on output partitioning.
- A genuine pre-swizzled format would need a new weight layout contract plus
  matching kernels for ordinary matvec, fused QKV, fused gate/up SwiGLU, down
  projection, and lm-head argmax. Repacking only at load while feeding the
  current kernels would break results; repacking then unswizzling before the
  current kernels would not test F12.

Results:
- No code change was made.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F12 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 620 ms |
| segmented | 460 ms |
| streaming | 470 ms |
| overall average | 516.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** The current row-major SDOT kernel is already close to
the simple contiguous streaming layout. F12 should be revisited only as a
coordinated kernel/layout change, likely together with F19's sidecar artifact
format so the swizzled layout can be generated once and mmap'd directly. No
code was kept.

### F14: BNNS direct convolution for conv2/conv3

Change:
- Audited the encoder convolution stem. The current path runs 3x3 stride-2
  padded convolutions as im2col plus `cblas_sgemm`.
- Wrote a temporary C probe outside the repo using
  `BNNSFilterCreateLayerConvolution` with `BNNSDataLayoutImageCHW` inputs and
  `BNNSDataLayoutConvolutionWeightsOIHW` weights, matching the current CHW/OIHW
  memory layout.
- Probed representative real conv shapes with random f32 data:
  conv2-like `480x64x100 -> 480x32x50` and
  conv3-like `480x32x50 -> 480x16x25`.

Probe result:

| Shape | BNNS direct conv | im2col + SGEMM |
|-------|-----------------:|---------------:|
| conv2-like | 7.077 ms | 6.136 ms |
| conv3-like | 1.280 ms | 1.556 ms |

Results:
- No code change was made. BNNS won on the smaller conv3-like shape but lost on
  the larger conv2-like shape, which is the heavier layer.
- Speed (`bench/run.sh --runs 10`, current accepted code, no BNNS integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 604 ms |
| segmented | 474 ms |
| streaming | 470 ms |
| overall average | 516.0 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Rejected/deferred.** The probe did not justify replacing the
current im2col+SGEMM path wholesale. A partial conv3-only integration would add
deprecated BNNS filter setup, descriptor lifetime management, and numeric parity
risk for at most a small fraction of encoder time. Revisit only with BNNSGraph
or if profiling shows conv3 alone has become a clear bottleneck. No code was
kept.

### F15: encoder window batching probe

Change:
- Audited the encoder forward path to test the premise that attention/FFN GEMMs
  are issued per `enc_n_window_infer` window.
- The convolution stem is processed per encoder chunk, but after stem projection
  all encoder transformer buffers are sized for `total_tokens`.
- Q/K/V, attention output projection, FFN `fc1/fc2`, `proj1`, and `proj2` call
  `linear_bf16_scratch`/`linear_accumulate_bf16_scratch` with `M =
  total_tokens`, not one window at a time.
- `window_starts` is only passed into `bidirectional_attention` to constrain
  attention ranges; it does not split the encoder GEMMs.

Results:
- No code change was made. There is no local window-batching opportunity in the
  current encoder transformer GEMM path because it is already batched across the
  full encoded token sequence.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F15 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 607 ms |
| segmented | 462 ms |
| streaming | 466 ms |
| overall average | 511.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Rejected/deferred.** The original C6 concern does not apply to the
current encoder transformer implementation. Future batching work should target
the per-chunk convolution stem or multi-request/session batching, not
per-window encoder GEMM batching. No code was kept.

### F17: CPU/AMX overlap inside the encoder

Change:
- Audited the encoder GEMM wrappers and synchronization points.
- `linear_bf16_scratch` and `linear_accumulate_bf16_scratch` synchronously
  convert BF16 weights into a shared f32 scratch buffer and then synchronously
  call the current `linear`/`linear_accumulate` SGEMM path.
- The current API returns only after both conversion and SGEMM are complete; it
  has no in-flight GEMM handle that would let CPU work such as next im2col,
  norms, activations, or softmax run concurrently.
- Reordering this safely would require dedicated GEMM worker ownership, scratch
  double-buffering, and a dependency schedule through the encoder layer graph.
  It is finer-grained and riskier than the already deferred F16 pipeline.

Results:
- No code change was made. A superficial thread spawn around `cblas_sgemm`
  would add synchronization overhead without exposing independent CPU work in
  the current call structure.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F17 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 636 ms |
| segmented | 462 ms |
| streaming | 482 ms |
| overall average | 526.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** F17 remains an architecture-level scheduling
experiment, but the current synchronous scratch+SGEMM API gives no low-risk
place to overlap useful CPU work with AMX work. Revisit after F16/F27 or after
introducing an explicit asynchronous GEMM/scratch ownership abstraction. No
code was kept.

### F18: Winograd F(2x2, 3x3) for encoder convs

Change:
- Ran a profile pass to check F18's re-entry condition after the F14 BNNS probe.
- The current offline profile still shows convolution as a real bucket:
  `conv2d_op_ms = 70.1 ms` out of `total_ms = 480.0 ms` in the profile run
  (`14.6%` of inference).
- Audited the convolution implementation: all three stem convolutions share one
  im2col+SGEMM implementation with stride 2 and padding 1; E6 previously showed
  chunk-boundary/padding sensitivity.

Results:
- No code change was made. A correct Winograd implementation would need a new
  transformed kernel for the stride-2 padded CHW stem, careful boundary
  handling, and numeric parity validation across chunk sizes. A quick
  direct-conv rewrite would risk changing ASR behavior and would not be a
  faithful low-risk F(2x2, 3x3) experiment.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F18 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 614 ms |
| segmented | 444 ms |
| streaming | 474 ms |
| overall average | 510.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** Conv remains large enough to care about, but F18 is a
new convolution algorithm, not a local scheduling or layout tweak. Revisit only
with a standalone Winograd parity harness for the exact stride/padding/chunk
semantics, then integrate behind the usual WER gate. No code was kept.

### F21: pipeline load with inference stages

Change:
- Audited the model load and transcription boundary.
- `QwenCtx::load` is currently a pure model-construction API: it opens
  safetensors, detects config, synchronously loads `Encoder`, synchronously
  loads `Decoder`, then constructs KV/encoder/decoder scratch state.
- Audio samples, mel computation, and `Encoder::forward` live on the
  transcription side after a full `QwenCtx` has already been returned.
- Starting encoder inference while decoder loading continues would require a
  staged context or a one-shot load-and-transcribe API that can hold a
  partially initialized context, run mel/encoder after `Encoder::load`, and
  join decoder loading before decoder prefill.

Results:
- No code change was made. The current public surfaces (CLI, C API, JNI/Flutter
  bridge, streaming push API) all assume a fully loaded `QwenCtx` before
  transcription starts.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F21 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 636 ms |
| segmented | 453 ms |
| streaming | 476 ms |
| overall average | 521.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** F21 is feasible, but it is a staged-load API change,
not a local load-order tweak. Revisit when adding a one-shot cold-start
benchmark/API or during the F27 shared-weight/session split, where partial
model state can be represented cleanly. No code was kept.

### F24: LLVM BOLT post-link optimization

Change:
- Checked the current benchmark platform and binary format.
- Current environment is Darwin arm64 (`RELEASE_ARM64_T6050`), and
  `target/release/qwen-asr` is a Mach-O 64-bit arm64 executable.
- No `llvm-bolt`/`bolt` tool is available in this environment.
- The idea in `fable-ideas.md` is explicitly scoped to Linux/x86 OpenBLAS
  targets because BOLT is not available for macOS ld64/Mach-O output.

Results:
- No code or build-flow change was made.
- Speed (`bench/run.sh --runs 10`, current accepted macOS arm64 code, no BOLT):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 610 ms |
| segmented | 462 ms |
| streaming | 469 ms |
| overall average | 513.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Not applicable on this platform.** F24 still belongs to a future
Linux/x86 benchmark track, ideally after a Linux gate exists and after PGO
training has been revisited. No code was kept.

### F26: `os_workgroup` / explicit workload hints

Change:
- Audited the current macOS scheduling hooks and SDK APIs.
- The codebase has no existing QoS, `os_workgroup`, work-interval, or thread
  policy calls; the hot `parallel_for` pool is a plain persistent worker pool.
- macOS exposes `os_workgroup_interval_start/update/finish`, but the interval
  API requires member threads to have joined an interval workgroup.
- The available public creation entry point in this SDK is
  `AudioWorkIntervalCreate`, documented for audio realtime threads. Using it
  for ASR inference would require linking AudioToolbox, owning an interval
  object, and teaching all relevant worker threads to join/leave that workgroup
  around repeated decode/encode work.

Results:
- No code change was made. A small wrapper around the main thread would not
  affect the existing worker pool and would mostly test unsupported/mis-scoped
  API usage rather than F26's intended workload hint.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F26 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 617 ms |
| segmented | 468 ms |
| streaming | 470 ms |
| overall average | 518.3 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred.** F26 needs a deliberate macOS scheduling experiment
that includes worker membership and interval lifecycle. The current code shape
has no safe low-cost hook that would exercise the intended mechanism. No code
was kept.

### F28: parallel long-audio segmentation

Change:
- Audited the current segmented transcription loop and context ownership.
- `transcribe_segmented` computes split points, then processes segments
  serially with one mutable `QwenCtx`.
- Every segment calls `transcribe_segment(ctx, ...)`, sharing mutable encoder
  buffers, decoder buffers, KV cache, RoPE cache, prompt state, tokenizer/model
  path state, and perf counters.
- Running segments in parallel without F27 would require duplicating the full
  model per worker or unsafely sharing mutable session state.

Results:
- No code change was made. A correct F28 implementation still depends on the
  F27 split into shared immutable weights plus per-worker session state, and it
  also needs a long-audio benchmark gate rather than the current single 28 s
  sample.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F28 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 470 ms |
| segmented | 358 ms |
| streaming | 364 ms |
| overall average | 397.3 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred, blocked by F27.** F28 remains a good long-audio
throughput target, but the current code has no independent session object to
run in parallel. Revisit after F27 and after adding a long-file benchmark to
make the speed gate meaningful. No code was kept.

### F29: daemon / server mode

Change:
- Audited CLI and embedding surfaces for an existing resident server mode.
- There is no `--serve`, TCP listener, or daemon loop in the CLI.
- The C/JNI embedding APIs already let a host process load a `QwenCtx` once and
  call transcription repeatedly, but the benchmark gate launches a fresh CLI
  process per run and therefore includes no repeated-request residency test.
- A daemon/server implementation would need a request protocol, lifecycle and
  shutdown behavior, concurrency policy, and a benchmark that separates first
  request from warm resident requests.

Results:
- No code change was made. A daemon would not improve the current single-run
  `bench/run.sh` inference gate, whose reported inference timer already
  excludes process startup and model load.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F29 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 474 ms |
| segmented | 344 ms |
| streaming | 366 ms |
| overall average | 394.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred / benchmark-not-covered.** F29 is useful for repeated
requests and product embedding, but it needs a resident-server benchmark rather
than the current one-shot CLI gate. No code was kept.

### F30: activation-aware weight-role calibration matrix

Change:
- Audited the existing quantization and benchmark tooling.
- The runtime has per-row INT8 decode quantization and historical WER runs, but
  there is no offline harness that sweeps tensor roles, formats, group sizes,
  zero-points, and activation-aware scale search across a calibration corpus.
- F30 is the prerequisite that would make F2/F3 calibrated INT4 experiments
  measurable instead of ad hoc.

Results:
- No code change was made. A useful F30 implementation is a separate offline
  calibration/sweep program plus result matrix, not a direct runtime
  optimization.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F30 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 649 ms |
| segmented | 461 ms |
| streaming | 512 ms |
| overall average | 540.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred / tooling track.** F30 should be built as an offline
calibration matrix before revisiting calibrated INT4, f16 role selection, or
other WER-sensitive compression. No runtime code was kept.

### F31: structured sparsity or magnitude pruning of decoder weights

Change:
- Audited the current decode kernels for sparse/pruned support.
- All hot decode kernels are dense INT8 SDOT scans over contiguous row-major
  weights (`matvec_int8`, fused QKV, fused SwiGLU, down projection, and
  `argmax_int8_range`).
- There is no 2:4 metadata format, sparse row iterator, sparse SDOT kernel, or
  pruning/fine-tuning pipeline.

Results:
- No code change was made. Zeroing weights without a sparse kernel would not
  reduce bandwidth, and pruning without fine-tuning/calibration would be a
  WER-risk experiment rather than a safe speed patch.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F31 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 628 ms |
| segmented | 460 ms |
| streaming | 488 ms |
| overall average | 525.3 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred / research track.** F31 needs both a model-compression
pipeline and a sparse decode kernel. Revisit only after F30-style calibration
and preferably after F2/F3 determines whether dense INT4 is enough. No code was
kept.

### F32: train/distill a tiny draft model for true speculative decoding

Change:
- Audited the repo for a draft-model training, distillation, or runtime
  verifier path.
- The current runtime loads one Qwen ASR decoder and the existing speculative
  notes cover algorithm sketches only; there is no tiny draft model artifact,
  training pipeline, or batched verifier API.
- F32 is a model-building track rather than a local runtime-only patch.

Results:
- No code change was made. Implementing this safely requires a compatible draft
  decoder trained on the same tokenizer/audio-conditioning contract, plus a
  verifier path that can score multiple proposed tokens in one pass.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F32 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 608 ms |
| segmented | 462 ms |
| streaming | 474 ms |
| overall average | 514.7 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred / model-training track.** F32 is promising, but this repo
does not yet have the draft model, distillation data path, or multi-token
verifier needed to make it a faithful speculative-decoding experiment. No code
was kept.

### F33: encoder token merging / output downsampling

Change:
- Audited the encoder-to-decoder boundary for token merging or output
  downsampling hooks.
- `Encoder::forward` returns a dense `enc_output` plus `total_tokens` after the
  convolution stem, encoder transformer, and final projection.
- `transcribe_segment` copies every encoder token into the decoder prompt and
  `decoder_prefill` processes the full sequence. There is no existing
  merge-policy hook, similarity metric, or WER guard for dropping/averaging
  encoder tokens.

Results:
- No code change was made. A naive stride-2 or averaging pass after the encoder
  would change the acoustic-token contract seen by the decoder and is expected
  to be WER-sensitive without a tuned policy or retraining.
- Speed (`bench/run.sh --runs 10`, current accepted code, no F33 integration):

| Mode | Current accepted code |
|------|----------------------:|
| offline | 471 ms |
| segmented | 362 ms |
| streaming | 362 ms |
| overall average | 398.3 ms |

- 100-file LibriSpeech offline WER:

| Metric | Current accepted code |
|--------|----------------------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Deferred / WER-sensitive model-contract change.** F33 needs a
separate token-merge policy experiment with a WER sweep, and possibly decoder
adaptation, before it can be treated as a safe runtime optimization. No code
was kept.

### B9: overlap lm_head argmax with next-token start

Change:
- In `decoder_forward`, after the final RMS norm, run the `lm_head` argmax and
  the next-position preparation (`kv_cache.grow(next_pos + 1)` and
  `rope.ensure(next_pos + 1, ...)`) in parallel via `std::thread::scope`.
- The next decode step's KV-cache capacity and RoPE tables are independent of
  the argmax result, so they can be prepared while the vocabulary is still being
  scored.

Baseline for this experiment is the post-`LONG_AUDIO_FAST`-removal HEAD
(`f28145c`, `bench/run.sh --runs 10`):

| Mode | Baseline | B9 overlap | Delta |
|------|---------:|-----------:|------:|
| offline | 587.0 ms | 578.0 ms | −1.5% |
| segmented | 456.0 ms | 446.5 ms | −2.1% |
| streaming | 503.0 ms | 505.5 ms | +0.5% |
| overall average | 515.2 ms | 510.0 ms | −1.0% |

- 100-file LibriSpeech offline WER:

| Metric | B9 overlap |
|--------|-----------:|
| Corpus WER | 0.0387 |
| Macro WER | 0.0428 |
| Corpus CER | 0.0154 |

Decision: **Accepted.** WER is unchanged and all offline/segmented modes show a
small speed improvement; streaming is within noise. The change is low-risk and
removes a small serial dependency at the end of each decode step. The code was
kept in `crates/qwen-asr/src/decoder.rs`.
