//! Exactness + rollback acceptance tests for the R13-A multi-token greedy
//! verifier (`decoder_forward_verify`).
//!
//! The verifier advances one decode session by up to `MAX_BATCH` consecutive
//! positions in a single fused region, streaming each weight row once for all
//! positions (R12-E2 batched INT8 kernels). By construction every lane's math
//! is bit-identical to the sequential single-token [`decoder_forward`], so:
//!   1. each verify `out_argmax[i]` must equal the sequential argmax at the same
//!      position (regardless of what draft tokens fill the later lanes), and
//!   2. the greedy acceptance rule + rollback must reproduce a byte-identical
//!      committed token stream for any drafts (correct on accept, safe on
//!      mismatch).
//!
//! `decoder_forward_verify` is aarch64-only; the whole file compiles to nothing
//! elsewhere.
#![cfg(target_arch = "aarch64")]

use qwen_asr::context::QwenCtx;
use qwen_asr::decoder::{
    decoder_forward, decoder_forward_verify, decoder_prefill, tok_embed_bf16_to_f32,
    verify_accepted_len, DecoderBuffers, KvCache, RopeCache, VerifyBufferPool,
};
use qwen_asr::kernels;

use std::sync::Mutex;

// The kernel thread pool is a global singleton; serialize against the other
// model-loading integration tests (each test file owns its own static mutex).
static TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Resolve a path that exists either relative to the workspace root or to the
/// crate dir (cargo runs test binaries with cwd = package dir).
fn resolve(rel: &str) -> Option<String> {
    for prefix in ["", "../../"] {
        let p = format!("{prefix}{rel}");
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    None
}

/// Build a fresh KV cache + RoPE cache and prefill a fixed token sequence.
/// Returns `(kv_cache, rope, prefill_len)` positioned right after the prefill.
fn fresh_prefill(ctx: &QwenCtx, prefill_tokens: &[i32]) -> (KvCache, RopeCache, usize) {
    let cfg = &ctx.config;
    let decoder = &ctx.model.decoder;
    let dim = cfg.dec_hidden;
    let tok_emb = decoder.tok_embeddings_bf16;

    let seq = prefill_tokens.len();
    let mut embeds = vec![0f32; seq * dim];
    for (i, &t) in prefill_tokens.iter().enumerate() {
        unsafe {
            tok_embed_bf16_to_f32(&mut embeds[i * dim..(i + 1) * dim], tok_emb, t, dim);
        }
    }

    let mut kv = KvCache::new(cfg.dec_layers, 2048, cfg.dec_kv_heads, cfg.dec_head_dim);
    let mut rope = RopeCache::new();
    let mut bufs = DecoderBuffers::new(cfg);
    bufs.ensure_prefill(seq, cfg);
    decoder_prefill(decoder, cfg, &mut kv, &mut rope, &mut bufs, &embeds, seq);
    assert_eq!(kv.len, seq);
    (kv, rope, seq)
}

/// Sequential single-token greedy argmaxes for a fixed continuation fed one at a
/// time after the prefill. `seq_argmax[i]` is the greedy token predicted AFTER
/// feeding `continuation[i]` (at position `prefill_len + i`).
fn sequential_argmaxes(ctx: &QwenCtx, prefill_tokens: &[i32], continuation: &[i32]) -> Vec<i32> {
    let cfg = &ctx.config;
    let decoder = &ctx.model.decoder;
    let dim = cfg.dec_hidden;
    let tok_emb = decoder.tok_embeddings_bf16;

    let (mut kv, mut rope, _) = fresh_prefill(ctx, prefill_tokens);
    let mut bufs = DecoderBuffers::new(cfg);
    let mut emb = vec![0f32; dim];
    let mut out = Vec::with_capacity(continuation.len());
    for &t in continuation {
        unsafe {
            tok_embed_bf16_to_f32(&mut emb, tok_emb, t, dim);
        }
        let a = decoder_forward(decoder, cfg, &mut kv, &mut rope, &mut bufs, &emb);
        out.push(a);
    }
    out
}

#[test]
fn verify_exactness_vs_sequential() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let model_dir = match resolve("qwen3-asr-0.6b") {
        Some(d) if std::path::Path::new(&d).join("model.safetensors").exists() => d,
        _ => {
            eprintln!("Skipping verify test: model not downloaded");
            return;
        }
    };

    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());
    let ctx = QwenCtx::load(&model_dir).expect("load ctx");
    let cfg = &ctx.config;
    let decoder = &ctx.model.decoder;
    let dim = cfg.dec_hidden;
    let tok_emb = decoder.tok_embeddings_bf16;

    // Fixed arbitrary prefill + continuation (valid token ids). Semantics are
    // irrelevant — the test only checks numeric equivalence of the two paths.
    let prefill: Vec<i32> = vec![10, 250, 500, 12, 900, 33, 4, 777];
    let continuation: Vec<i32> = vec![
        7, 42, 100, 256, 1, 333, 88, 640, 500, 3, 71, 900, 12, 45, 600, 9,
    ];

    let seq_argmax = sequential_argmaxes(&ctx, &prefill, &continuation);
    assert_eq!(seq_argmax.len(), continuation.len());

    let mut pool = VerifyBufferPool::new();
    let mut compared = 0usize;

    for &k in &[2usize, 5, 8] {
        let (mut kv, mut rope, base0) = fresh_prefill(&ctx, &prefill);
        let mut pos = 0usize;
        let mut embeds = vec![0f32; k * dim];
        let mut out = vec![0i32; k];

        for chunk in continuation.chunks(k) {
            let kk = chunk.len();
            for (i, &t) in chunk.iter().enumerate() {
                unsafe {
                    tok_embed_bf16_to_f32(&mut embeds[i * dim..(i + 1) * dim], tok_emb, t, dim);
                }
            }
            let expect_base = base0 + pos;
            assert_eq!(kv.len, expect_base, "verify entry position drift (k={k})");
            let bufs = pool.ensure(kk, cfg);
            decoder_forward_verify(
                decoder,
                cfg,
                &mut kv,
                &mut rope,
                bufs,
                &embeds[..kk * dim],
                kk,
                &mut out[..kk],
            );
            assert_eq!(kv.len, expect_base + kk, "kv.len must be base + k (k={k})");
            for i in 0..kk {
                assert_eq!(
                    out[i],
                    seq_argmax[pos + i],
                    "verify argmax mismatch at position {} (chunk k={k})",
                    pos + i
                );
                compared += 1;
            }
            pos += kk;
        }
        assert_eq!(pos, continuation.len());
    }

    // 3 sweeps over the whole continuation.
    assert_eq!(compared, 3 * continuation.len());
    eprintln!(
        "verify_exactness_vs_sequential: {} positions compared exactly (k in 2,5,8)",
        compared
    );

    kernels::set_threads(kernels::get_num_cpus());
}

/// Drive greedy generation through the verifier + acceptance rule. `drafts_for`
/// proposes up to `k-1` draft tokens given the already-committed stream
/// (position `prefill_len` onward, seed included as element 0). Returns the
/// committed tokens EXCLUDING the seed (i.e. the generated argmax stream).
#[allow(clippy::too_many_arguments)]
fn generate_via_verify(
    ctx: &QwenCtx,
    prefill: &[i32],
    seed: i32,
    n_target: usize,
    k: usize,
    pool: &mut VerifyBufferPool,
    mut drafts_for: impl FnMut(&[i32]) -> Vec<i32>,
) -> Vec<i32> {
    let cfg = &ctx.config;
    let decoder = &ctx.model.decoder;
    let dim = cfg.dec_hidden;
    let tok_emb = decoder.tok_embeddings_bf16;

    let (mut kv, mut rope, base0) = fresh_prefill(ctx, prefill);
    // committed[0] = seed (occupies position base0, KV written on the first
    // verify step); committed[1..] are the generated tokens.
    let mut committed = vec![seed];
    let mut embeds = vec![0f32; k * dim];
    let mut out = vec![0i32; k];

    while committed.len() - 1 < n_target {
        let base = kv.len;
        assert_eq!(base, base0 + (committed.len() - 1), "position drift");
        let t0 = *committed.last().unwrap();

        let mut drafts = drafts_for(&committed);
        drafts.truncate(k - 1);
        let kk = 1 + drafts.len();

        // window = [t0, d1, .., d_{kk-1}]
        unsafe {
            tok_embed_bf16_to_f32(&mut embeds[..dim], tok_emb, t0, dim);
        }
        for (i, &d) in drafts.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(&mut embeds[(i + 1) * dim..(i + 2) * dim], tok_emb, d, dim);
            }
        }

        let bufs = pool.ensure(kk, cfg);
        decoder_forward_verify(
            decoder,
            cfg,
            &mut kv,
            &mut rope,
            bufs,
            &embeds[..kk * dim],
            kk,
            &mut out[..kk],
        );
        assert_eq!(kv.len, base + kk);

        let a = verify_accepted_len(&drafts, &out[..kk]);
        assert!(a <= drafts.len());
        // Commit accepted drafts d1..d_a, then the free next token out[a].
        for &d in &drafts[..a] {
            committed.push(d);
        }
        committed.push(out[a]);
        // Roll the shared cache back to the committed length.
        kv.len = base + a + 1;
    }

    // A step may commit several tokens at once, so the stream can overshoot
    // n_target; trim to the requested length (every committed token is correct
    // by the acceptance rule).
    let mut gen = committed[1..].to_vec();
    gen.truncate(n_target);
    gen
}

#[test]
fn verify_rollback_matches_sequential() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let model_dir = match resolve("qwen3-asr-0.6b") {
        Some(d) if std::path::Path::new(&d).join("model.safetensors").exists() => d,
        _ => {
            eprintln!("Skipping verify rollback test: model not downloaded");
            return;
        }
    };

    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());
    let ctx = QwenCtx::load(&model_dir).expect("load ctx");

    let prefill: Vec<i32> = vec![5, 99, 260, 41, 7, 800, 13];
    let seed: i32 = 128;
    let n_target = 20usize;

    // Pure sequential greedy reference: feed seed, then feed each argmax.
    let cfg = &ctx.config;
    let decoder = &ctx.model.decoder;
    let dim = cfg.dec_hidden;
    let tok_emb = decoder.tok_embeddings_bf16;
    let seq_stream = {
        let (mut kv, mut rope, _) = fresh_prefill(&ctx, &prefill);
        let mut bufs = DecoderBuffers::new(cfg);
        let mut emb = vec![0f32; dim];
        let mut stream = Vec::with_capacity(n_target);
        let mut cur = seed;
        for _ in 0..n_target {
            unsafe {
                tok_embed_bf16_to_f32(&mut emb, tok_emb, cur, dim);
            }
            let a = decoder_forward(decoder, cfg, &mut kv, &mut rope, &mut bufs, &emb);
            stream.push(a);
            cur = a;
        }
        stream
    };

    let mut pool = VerifyBufferPool::new();

    for &k in &[2usize, 5, 8] {
        // (a) Deliberately wrong drafts: acceptance rule should reject them
        // (a == 0 most steps), roll back k-1 positions, and still reproduce the
        // sequential stream one token per step.
        // Fixed "wrong" token; if one happens to match the true token the
        // acceptance rule still commits it correctly, so the assertion holds
        // regardless.
        let wrong =
            generate_via_verify(&ctx, &prefill, seed, n_target, k, &mut pool, |_committed| {
                vec![39997i32; k - 1]
            });
        assert_eq!(
            &wrong[..],
            &seq_stream[..],
            "wrong-draft verify stream must equal sequential (k={k})"
        );

        // (b) Oracle drafts: feed the TRUE upcoming tokens so every draft is
        // accepted (commit k tokens per step); the committed stream must be
        // identical too. Drafts for the current step are the sequential tokens
        // at the positions the verifier is about to produce.
        let oracle =
            generate_via_verify(&ctx, &prefill, seed, n_target, k, &mut pool, |committed| {
                let generated = committed.len() - 1; // index of next token in seq_stream
                let mut d = Vec::new();
                for j in 0..(k - 1) {
                    match seq_stream.get(generated + j) {
                        Some(&t) => d.push(t),
                        None => break,
                    }
                }
                d
            });
        assert_eq!(
            &oracle[..],
            &seq_stream[..],
            "oracle-draft verify stream must equal sequential (k={k})"
        );
    }

    eprintln!(
        "verify_rollback_matches_sequential: {} committed tokens per strategy match sequential (k in 2,5,8)",
        seq_stream.len()
    );

    kernels::set_threads(kernels::get_num_cpus());
}
