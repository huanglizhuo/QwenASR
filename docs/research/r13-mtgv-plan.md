# R13: Multi-token greedy verifier (MTGV) — implementation plan

Date: 2026-07-16. Planner: Fable 5. Implementation: Opus 4.8 subagents.

## Background

Prior art: E13 / G25 / F6 / F7 / F32 all deferred speculative decoding for two
reasons: (1) no multi-position greedy-argmax path that avoids materializing
full `[K × vocab]` logits; (2) no draft source. Reason (1) was closed by
R12-E2/E3: `decoder_forward_batched` + `int8_*_range_batched` kernels +
`argmax_matvec_int8_batched` stream each weight row once for up to
`MAX_BATCH = 8` batch elements, and R12-F1 profiling measured lockstep step
time ~constant in batch size (24 ms/step @ B=8, 27 @ B=5) — i.e. verifying
K ≤ 8 positions costs ≈ one single-token step.

Correctness is free by construction: INT8 dots are order-exact in i32, glue
(norm/RoPE/quantize) runs identical serial per-position code, and a greedy
verifier only ever commits tokens equal to the sequential greedy argmax → the
final transcript is byte-identical to HEAD. Any transcript diff = bug.

Cost model: single-token step streams ~575 MB (420 MB layers + 155 MB lm_head)
at ~115 GB/s ≈ 5 ms floor. Verify step for K ≤ 8 positions ≈ 1.0–1.2× a
single step; tokens committed per step = accepted_drafts + 1 (the argmax at
the last accepted position is the next true token, free). Worst case (all
drafts wrong) still commits 1 token per step at ~1.1× cost.

Expected gains: streaming end-to-end 1.3–1.8× (draft = previous chunk's
regenerated tail, d≈0, high acceptance); offline 1.15–1.5× IF the layer-skip
acceptance probe (R13-C) passes; segmented long audio ≈0 (lockstep already
amortizes the weight stream ~7×; B×K would exceed MAX_BATCH) — out of scope.

## Phase R13-A: verifier core (`decoder_forward_verify`)

New `#[cfg(target_arch = "aarch64")]` function in `decoder.rs`, a variant of
`decoder_forward_batched` where the K batch elements are K *consecutive
positions of one session* sharing one `KvCache`/`RopeCache`:

```rust
pub fn decoder_forward_verify(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut [DecoderBuffers],   // >= k independent buffer sets
    input_embeds: &[f32],          // k × dim: embeds of [t0, d1, .., d_{k-1}]
    k: usize,                      // 1..=kernels::MAX_BATCH
    out_argmax: &mut [i32],        // k results
)
```

Semantics (defines the off-by-one exactly): the caller holds the current
pending token `t0` (not yet fed) and draft tokens `d1..d_{k-1}`. The function
advances the session by k positions `base..base+k` (`base = kv_cache.len` on
entry), writes K/V for all k, sets `kv_cache.len = base + k`, and returns
`out_argmax[i]` = greedy argmax after position `base+i`. The **caller** then:

- `accepted` = longest `a` such that `d_{i} == out_argmax[i-1]` for all
  `1 <= i <= a` (a ∈ 0..k-1);
- commits `t0, d1..d_a` plus the free next token `out_argmax[a]`;
- rolls back `kv_cache.len = base + a + 1` (stale K/V rows beyond are
  overwritten later; no cleanup);
- `out_argmax[a]` becomes the next step's `t0`.

Implementation notes (all load-bearing):
- `k == 1` delegates to `decoder_forward` (identical math, keeps the tuned
  single-token epilogue).
- Grow/ensure ONCE up front: `kv_cache.grow(base + k)` and
  `rope.ensure(base + k, ..)` BEFORE capturing raw pointers (the batched code
  grows per-session at setup; here a single shared cache must not reallocate
  after pointers are taken).
- `pos[bi] = base + bi`; per-position RoPE rows `rope.cos_at(base+bi)`;
  attention `total_seq = pos[bi] + 1`, all `kv_k_ptr[bi]` aliases of the same
  buffer. The existing stage order is already correct: the KV-write stage
  barrier precedes the attention stage, so position i reads rows written by
  positions j < i in the same layer (same causal structure as prefill).
- KV writes go to distinct `pos` rows → no write conflicts across batch lanes.
- lm_head: `argmax_matvec_int8_batched` over all k final hidden states (one
  155 MB stream per verify step instead of per token).
- Reuse the R12-E2 batched kernels untouched; no new kernels expected.
- Buffer pool: k independent `DecoderBuffers` (decode-only fields are small,
  ~100 KB each; prefill vecs stay empty). Add a lazily-grown
  `Vec<DecoderBuffers>` pool where the caller needs it (QwenCtx for R13-B).

Tests (`tests/verify.rs`, model-gated + skip pattern and `TEST_MUTEX`
serialization as in `tests/lockstep.rs`):
1. **Exactness vs sequential** (no audio needed): with the real model, prefill
   a fixed arbitrary token sequence into a fresh `KvCache`, then feed a fixed
   arbitrary continuation list one token at a time through `decoder_forward`,
   recording every argmax. Reset (fresh cache, same prefill), replay the same
   continuation through `decoder_forward_verify` in chunks of k ∈ {2, 5, 8}:
   every `out_argmax` must equal the sequential argmax at the same position.
2. **Rollback + continue**: run verify with deliberately wrong drafts,
   roll back per the acceptance rule, continue sequentially; the committed
   token stream must be identical to the pure-sequential stream.
3. Full suite `cargo test --release` green, zero warnings
   (`RUSTFLAGS="-C target-cpu=native"`), default features and
   `--no-default-features` both compile.

No callsite changes in this phase. Gate: tests + zero warnings only (no bench
movement expected).

## Phase R13-B: streaming overlap-draft integration (after A lands)

Draft source: the previous chunk's regenerated tail. In `transcribe_stream`
(`transcribe.rs:1851`) and `stream_push_audio` (~`transcribe.rs:2391`), the
previous iteration's `chunk_tokens` — saved BEFORE
`raw_tokens.truncate(n_prefix_tokens)` — predicts this chunk's regenerated
tokens with high fidelity (same audio re-decoded, diverging only near the new
audio tail). `prefill_lcp_len` already reuses the identical *prefix* rows;
the verifier accepts the re-converged *suffix* the LCP cut off.

Plan:
1. Instrument first (verbose-gated counters): per chunk log
   `chunk_tokens.len()`, LCP reuse, and match-prefix length between
   consecutive chunks' regenerated tails → measured acceptance ceiling.
2. Factor the shared decode-tail loop of both streaming paths into a helper;
   rewrite it: maintain a draft cursor into the previous tail (aligned by
   position past `n_prefix_tokens`); while drafts remain, take up to
   `MAX_BATCH-1` next drafts, call `decoder_forward_verify`, commit accepted
   run; on first mismatch drop ALL remaining drafts (no realignment
   heuristics in v1) and continue single-token.
3. EOS (`TOKEN_ENDOFTEXT` / `TOKEN_IM_END`) may appear anywhere in a committed
   run: truncate at the first terminal and set `kv_cache.len` accordingly;
   `max_new_tokens` counts every committed token.
4. Kill switch: `QWEN_ASR_VERIFY=0` env → pure single-token path.

Gates: transcripts byte-identical to HEAD on all bench modes (required by
construction — any diff is a bug); full test suite; streaming speed improved
on back-to-back interleaved pairs (`bench/run.sh --runs 10`, judge on 3-mode
average, ±1.5% noise floor); offline/segmented flat.

## Phase R13-C: offline layer-skip acceptance probe (parallel, throwaway)

> **OUTCOME (2026-07-16): gate FAILED decisively — track closed.** Best
> agreement at L ≤ 10 was 1.94% (need ≥65%); every projected speedup < 1.0×.
> The decoder resolves the token only in its final ~8 layers. See the R13-C
> entry in experiments.md. R13-D below is cancelled; offline mode has no
> viable draft source.

Question: at early layer L, does `argmax(lm_head(final_norm(x_L)))` agree with
the final (layer-28) argmax often enough (≥65% at some L ≤ 10) to fund a
layer-skip self-draft for offline mode?

Method (temporary patch, never merged):
- In `decoder_forward`'s fused region, after each probed layer's last barrier,
  tid 0 copies `x` (dim floats) into a probe snapshot buffer
  (env `QWEN_ASR_PROBE_LAYERS="4,6,8,10,14"`).
- After the region: for each probed layer, `rms_norm(snapshot, decoder.norm)`
  + `lm_head_argmax`, compare to the step's real token. ~25 ms/token overhead
  is fine for a probe.
- Run offline mode over bench samples + ~50 LibriSpeech dev-clean files
  (`librispeech-wer-bench/dev-clean-full`), ≥3k decode steps total.
- Report per-layer: top-1 agreement %, and simulated verify economics: from
  the per-token agreement bitstream, expected accepted-run length for K=4 and
  K=8 windows, and projected decode speedup
  `(a+1)/(c + K·d)` with c=1.15, d=(L/28·420+155)/575 (full head) and
  d=(L/28·420+8)/575 (8k shortlist head).

Decision gate for R13-D (layer-skip draft + shortlist head): some L ≤ 10 with
agreement ≥65% AND projected offline decode speedup ≥1.25×. Otherwise record
the numbers in experiments.md and close the offline track.

## Phase R13-D: offline layer-skip draft — CANCELLED (R13-C gate failed)

Sketch: early-exit draft loop (first L layers + final norm + shortlist head of
~8k candidate rows: token frequency table from LibriSpeech + all tokens seen
in the current utterance), K=4–6 drafts per verify step, drafts write their
first-L-layer K/V which verify recomputes identically (or reuses — decide
during implementation). Full-set R12-B3 WER gate NOT needed for correctness
(output is byte-identical by construction) but run the 100-file continuity
check once as an implementation-bug tripwire.

## Benchmark discipline (all phases)

Back-to-back interleaved A/B pairs, `bench/run.sh --runs 10`, judge on 3-mode
average, ±1.5% noise floor, thermally warm machine. Record every phase as an
R13-* entry in `docs/research/experiments.md` (keep/reject + numbers), per
repo convention.
