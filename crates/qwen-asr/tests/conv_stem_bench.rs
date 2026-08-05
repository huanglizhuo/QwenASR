//! Conv-stem GEMM microbenchmark for issue #47.
//!
//! Reproduces the encoder stem's three `conv2d_with_cols` calls at their real
//! shapes without needing the model: only the shapes and the BLAS call pattern
//! matter for the effect under test, so random weights are fine.
//!
//! The pool-slicing policy (`QASR_CONV_POOLED`) is cached in a `OnceLock`, so a
//! single process can only measure one side. CI runs this binary twice, once
//! per setting, and diffs the printed totals — see `.github/workflows/ci.yml`.
//!
//! `#[ignore]`d so it never runs as part of the normal suite.

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
