//! Microbenchmarks for issue #47: every site where N pool threads call
//! `cblas_sgemm` concurrently.
//!
//! There are three, and they are covered here in the order they were found:
//!
//! 1. [`conv_stem_gemm_bench`] — the encoder conv stem (`QASR_CONV_POOLED`).
//! 2. [`linear_gemm_bench`] — `linear` via `sgemm_nt_pooled` (`QASR_LINEAR_POOLED`).
//! 3. [`causal_attention_prefill_bench`] — head-parallel prefill attention
//!    (`QASR_ATTN_POOLED`).
//!
//! Each reproduces its kernel's real shapes without the model: only the shapes
//! and the BLAS call pattern drive the effect, so random weights are fine.
//!
//! The policy flags are cached in a `OnceLock`, so one process can only measure
//! one side. CI runs this binary once per configuration and diffs the printed
//! `BENCH_TOTAL_MS` lines — see `.github/workflows/ci.yml`.
//!
//! `#[ignore]`d so they never run as part of the normal suite.

use qwen_asr::config::CONV_HIDDEN;
use qwen_asr::kernels;
use std::time::Instant;

/// Mel bins in, and the encoder's `enc_chunk_size` (see `config.rs`). A 28.2 s
/// clip is 2820 mel frames, so 29 chunks — the 87 `conv2d_op` calls in the
/// issue's profile output.
const MEL_BINS: usize = 128;
const CHUNK_W: usize = 100;
const N_CHUNKS: usize = 29;

/// Deterministic filler. Values only need to be finite and non-degenerate;
/// a plain LCG keeps this dependency-free and reproducible across runs.
fn fill_pseudo_random(buf: &mut [f32], seed: u64) {
    let mut s = seed | 1;
    for v in buf.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map the high bits into roughly [-0.5, 0.5].
        *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
    }
}

struct Layer {
    name: &'static str,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
}

impl Layer {
    fn out_dims(&self) -> (usize, usize) {
        // kernel 3, stride 2, padding 1 for all three stem convs.
        ((self.h_in + 2 - 3) / 2 + 1, (self.w_in + 2 - 3) / 2 + 1)
    }
}

#[test]
#[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
fn conv_stem_gemm_bench() {
    let threads = kernels::get_default_threads();
    kernels::set_threads(threads);

    // The stem's three convs, chained: each layer's output dims feed the next.
    let (h1, w1) = ((MEL_BINS + 2 - 3) / 2 + 1, (CHUNK_W + 2 - 3) / 2 + 1);
    let (h2, w2) = ((h1 + 2 - 3) / 2 + 1, (w1 + 2 - 3) / 2 + 1);
    let layers = [
        Layer {
            name: "conv1",
            c_in: 1,
            c_out: CONV_HIDDEN,
            h_in: MEL_BINS,
            w_in: CHUNK_W,
        },
        Layer {
            name: "conv2",
            c_in: CONV_HIDDEN,
            c_out: CONV_HIDDEN,
            h_in: h1,
            w_in: w1,
        },
        Layer {
            name: "conv3",
            c_in: CONV_HIDDEN,
            c_out: CONV_HIDDEN,
            h_in: h2,
            w_in: w2,
        },
    ];

    let pooled = std::env::var("QASR_CONV_POOLED").unwrap_or_else(|_| "<default>".into());
    println!("threads={threads} QASR_CONV_POOLED={pooled} chunks={N_CHUNKS}");

    let mut grand_total_ms = 0.0f64;
    let mut grand_total_macs = 0u64;

    for layer in &layers {
        let (h_out, w_out) = layer.out_dims();
        let patch = layer.c_in * 3 * 3;
        let spatial = h_out * w_out;

        let mut input = vec![0.0f32; layer.c_in * layer.h_in * layer.w_in];
        let mut weight = vec![0.0f32; layer.c_out * patch];
        let bias = vec![0.0f32; layer.c_out];
        fill_pseudo_random(&mut input, 0x5EED);
        fill_pseudo_random(&mut weight, 0xC0FFEE);

        let mut out = vec![0.0f32; layer.c_out * spatial];
        let mut cols: Vec<f32> = Vec::new();

        // One untimed pass so the `cols` scratch is grown and pages are faulted
        // in; the real encoder reuses these buffers across chunks.
        kernels::conv2d_with_cols(
            &mut out,
            &input,
            &weight,
            Some(&bias),
            &mut cols,
            layer.c_in,
            layer.c_out,
            layer.h_in,
            layer.w_in,
            3,
            3,
            2,
            1,
        );

        let start = Instant::now();
        for _ in 0..N_CHUNKS {
            kernels::conv2d_with_cols(
                &mut out,
                &input,
                &weight,
                Some(&bias),
                &mut cols,
                layer.c_in,
                layer.c_out,
                layer.h_in,
                layer.w_in,
                3,
                3,
                2,
                1,
            );
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;

        let macs = (layer.c_out * patch * spatial) as u64 * N_CHUNKS as u64;
        grand_total_ms += ms;
        grand_total_macs += macs;

        println!(
            "{:<6} M={:<4} N={:<5} K={:<5} cols={:>6.1}MB  {:>9.1} ms  ({:>6.1} GFLOP/s)",
            layer.name,
            layer.c_out,
            spatial,
            patch,
            (patch * spatial * 4) as f64 / (1024.0 * 1024.0),
            ms,
            (2.0 * macs as f64) / (ms / 1000.0) / 1e9,
        );

        // Guard against the compiler deciding none of this is observable.
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{} produced non-finite output",
            layer.name
        );
    }

    println!(
        "TOTAL  {:.1} ms  ({:.1} GFLOP/s)",
        grand_total_ms,
        (2.0 * grand_total_macs as f64) / (grand_total_ms / 1000.0) / 1e9,
    );
    // Machine-readable line for the CI comparison step.
    println!("BENCH_TOTAL_MS {grand_total_ms:.1}");
}

/// The other site that issues concurrent `cblas_sgemm` calls from the pool:
/// `linear`, via `sgemm_nt_pooled`. The conv stem hangs on stock OpenBLAS when
/// sliced; this covers whether `linear`'s smaller operands and lower slice
/// count (`out_dim = 896` → 7) avoid it or merely make it rarer. Driven by
/// `QASR_LINEAR_POOLED`.
#[test]
#[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
fn linear_gemm_bench() {
    let threads = kernels::get_default_threads();
    kernels::set_threads(threads);

    // Encoder transformer shapes: d_model 896, ffn 3584, ~750 tokens for a
    // 28 s clip. `sgemm_nt_pooled` needs seq_len >= 2 and out_dim >= 256.
    const SEQ: usize = 750;
    const D: usize = 896;
    const FFN: usize = 3584;
    let shapes = [("qkvo", D, D), ("fc1", D, FFN), ("fc2", FFN, D)];

    let pooled = std::env::var("QASR_LINEAR_POOLED").unwrap_or_else(|_| "<default>".into());
    println!("threads={threads} QASR_LINEAR_POOLED={pooled} seq={SEQ}");

    let mut total_ms = 0.0f64;
    let mut total_macs = 0u64;
    // 18 encoder layers; one representative call per shape per layer.
    const REPS: usize = 18;

    for (name, in_dim, out_dim) in shapes {
        let mut x = vec![0.0f32; SEQ * in_dim];
        let mut w = vec![0.0f32; out_dim * in_dim];
        let b = vec![0.0f32; out_dim];
        fill_pseudo_random(&mut x, 0xA11CE);
        fill_pseudo_random(&mut w, 0xB0B);
        let mut y = vec![0.0f32; SEQ * out_dim];

        kernels::linear(&mut y, &x, &w, Some(&b), SEQ, in_dim, out_dim);

        let start = Instant::now();
        for _ in 0..REPS {
            kernels::linear(&mut y, &x, &w, Some(&b), SEQ, in_dim, out_dim);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;

        let macs = (SEQ * in_dim * out_dim) as u64 * REPS as u64;
        total_ms += ms;
        total_macs += macs;
        println!(
            "{name:<5} M={SEQ:<4} N={out_dim:<5} K={in_dim:<5} {ms:>9.1} ms  ({:>6.1} GFLOP/s)",
            (2.0 * macs as f64) / (ms / 1000.0) / 1e9,
        );
        assert!(
            y.iter().all(|v| v.is_finite()),
            "{name} produced non-finite output"
        );
    }

    println!(
        "TOTAL  {:.1} ms  ({:.1} GFLOP/s)",
        total_ms,
        (2.0 * total_macs as f64) / (total_ms / 1000.0) / 1e9,
    );
    println!("BENCH_TOTAL_MS {total_ms:.1}");
}

/// The third site that issues concurrent `cblas_sgemm` calls from the pool, and
/// the one #47's gating does *not* cover: `causal_attention` fans out over
/// query heads with `parallel_for`, and for `seq_q > 1` each worker's
/// `causal_attention_heads` issues two `cblas_sgemm` calls per head. That is the
/// decoder prefill path (`decoder.rs:926` passes `seq_len`, not 1); single-token
/// decode takes the BLAS-free online-softmax branch and is unaffected.
///
/// Gated by `attn_parallel_heads` (Apple-only default); `QASR_ATTN_POOLED=1`
/// forces the fan-out on so the off-vendor cost stays visible in CI.
#[test]
#[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
fn causal_attention_prefill_bench() {
    // Decoder geometry: 16 query heads, 8 KV heads (GQA), head_dim 64, 28
    // layers. seq_q = seq_k = a representative prefill length.
    const N_HEADS: usize = 16;
    const N_KV: usize = 8;
    const HEAD_DIM: usize = 64;
    const LAYERS: usize = 28;
    const SEQ: usize = 512;

    // Drive the real `attn_parallel_heads` gate rather than a thread-count
    // proxy, so this measures the shipped decision. The pool keeps all its
    // threads either way; only the head fan-out changes.
    let threads = kernels::get_default_threads();
    kernels::set_threads(threads);
    let pooled = std::env::var("QASR_ATTN_POOLED").unwrap_or_else(|_| "<default>".into());
    println!("threads={threads} QASR_ATTN_POOLED={pooled} seq_q={SEQ} seq_k={SEQ} heads={N_HEADS}/{N_KV}");

    let q_hidden = N_HEADS * HEAD_DIM;
    // KV cache layout is [head][pos][head_dim]; head_stride spans one KV head.
    let head_stride = SEQ * HEAD_DIM;
    let mut q = vec![0.0f32; SEQ * q_hidden];
    let mut k = vec![0.0f32; N_KV * head_stride];
    let mut v = vec![0.0f32; N_KV * head_stride];
    fill_pseudo_random(&mut q, 0xDEC0DE);
    fill_pseudo_random(&mut k, 0x1234_5678);
    fill_pseudo_random(&mut v, 0x8765_4321);
    let mut out = vec![0.0f32; SEQ * q_hidden];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let call = |out: &mut Vec<f32>| {
        kernels::causal_attention(
            out,
            &q,
            k.as_ptr(),
            v.as_ptr(),
            head_stride,
            SEQ,
            SEQ,
            N_HEADS,
            N_KV,
            HEAD_DIM,
            scale,
            0,
        );
    };
    call(&mut out);

    let start = Instant::now();
    for _ in 0..LAYERS {
        call(&mut out);
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0;

    assert!(
        out.iter().all(|x| x.is_finite()),
        "non-finite attention output"
    );
    println!(
        "attn   {LAYERS} layers  {ms:.1} ms  ({:.2} ms/layer)",
        ms / LAYERS as f64
    );
    println!("BENCH_TOTAL_MS {ms:.1}");
}
