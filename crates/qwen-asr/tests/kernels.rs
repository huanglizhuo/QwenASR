use qwen_asr::kernels;
use qwen_asr::kernels::generic;

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn test_bf16_roundtrip() {
    let values: Vec<f32> = vec![
        0.0,
        1.0,
        -1.0,
        std::f32::consts::PI,
        -2.71,
        100.5,
        0.001,
        -0.001,
    ];
    for &v in &values {
        let bf16 = ((v.to_bits() + 0x8000) >> 16) as u16;
        let back = kernels::bf16_to_f32(bf16);
        assert!(
            (v - back).abs() < 0.02 * v.abs().max(1.0),
            "BF16 roundtrip failed: {} -> {} -> {}",
            v,
            bf16,
            back
        );
    }
}

#[test]
fn test_bf16_matvec_vs_generic() {
    let in_dim = 128;
    let out_dim = 16;
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01 - 0.64).collect();
    let mut w_bf16 = vec![0u16; out_dim * in_dim];
    for r in 0..out_dim {
        for c in 0..in_dim {
            let v = ((r * in_dim + c) as f32) * 0.001 - 0.5;
            w_bf16[r * in_dim + c] = (v.to_bits() >> 16) as u16;
        }
    }
    let bias: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.1).collect();

    let mut y_generic = vec![0.0f32; out_dim];
    unsafe {
        generic::bf16_matvec_fused(
            &mut y_generic,
            &x,
            w_bf16.as_ptr(),
            Some(&bias),
            in_dim,
            out_dim,
        );
    }

    let mut y_dispatch = vec![0.0f32; out_dim];
    kernels::linear_nobias_bf16(&mut y_dispatch, &x, w_bf16.as_ptr(), 1, in_dim, out_dim);
    for i in 0..out_dim {
        y_dispatch[i] += bias[i];
    }

    let err = max_abs_err(&y_generic, &y_dispatch);
    assert!(
        err < 0.01,
        "bf16 matvec dispatch vs generic mismatch: max_err={}",
        err
    );
}

#[test]
fn test_dot_f32_vs_generic() {
    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32) * 0.002 - 1.0).collect();

    let result_generic = generic::dot_f32(&a, &b, n);
    let result_dispatch = kernels::dot_f32(&a, &b, n);

    let err = (result_generic - result_dispatch).abs();
    assert!(
        err < 0.01 * result_generic.abs().max(1.0),
        "dot_f32 mismatch: generic={}, dispatch={}, err={}",
        result_generic,
        result_dispatch,
        err
    );
}

#[test]
fn test_rms_norm() {
    let dim = 128;
    let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.02 - 1.28).collect();
    let w: Vec<f32> = vec![1.0; dim];
    let eps = 1e-6;

    let mut out = vec![0.0f32; dim];
    kernels::rms_norm(&mut out, &x, &w, 1, dim, eps);

    let rms: f32 = (out.iter().map(|v| v * v).sum::<f32>() / dim as f32).sqrt();
    assert!(
        (rms - 1.0).abs() < 0.01,
        "RMS norm output should have RMS ~1.0, got {}",
        rms
    );
}

#[test]
fn test_layer_norm() {
    let dim = 128;
    let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.05 - 3.2).collect();
    let w: Vec<f32> = vec![1.0; dim];
    let b: Vec<f32> = vec![0.0; dim];
    let eps = 1e-5;

    let mut out = vec![0.0f32; dim];
    kernels::layer_norm(&mut out, &x, &w, &b, 1, dim, eps);

    let mean: f32 = out.iter().sum::<f32>() / dim as f32;
    let var: f32 = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
    assert!(
        mean.abs() < 0.01,
        "LayerNorm mean should be ~0, got {}",
        mean
    );
    assert!(
        (var - 1.0).abs() < 0.02,
        "LayerNorm variance should be ~1, got {}",
        var
    );
}

#[test]
fn test_softmax() {
    let n = 10;
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    kernels::softmax(&mut x, 1, n);

    let sum: f32 = x.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax should sum to 1.0, got {}",
        sum
    );
    for i in 1..n {
        assert!(
            x[i] >= x[i - 1],
            "Softmax should be monotonically increasing"
        );
    }
    assert!(
        x[0] > 0.0 && x[9] > 0.0,
        "All softmax values should be positive"
    );
}

#[test]
fn test_gelu() {
    let mut x = vec![0.0f32, 1.0, -1.0, 2.0, -0.5];
    let n = x.len();
    kernels::gelu(&mut x, n);
    assert!(x[0].abs() < 1e-5, "GELU(0) should be ~0");
    assert!(x[1] > 0.5, "GELU(1) should be > 0.5");
    assert!(x[2] < 0.0, "GELU(-1) should be negative");
}

#[test]
fn test_silu() {
    let orig = vec![0.0f32, 1.0, -1.0, 5.0];
    let expected: Vec<f32> = orig.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    let mut x = orig;
    let n = x.len();
    kernels::silu(&mut x, n);
    let err = max_abs_err(&x, &expected);
    assert!(err < 1e-5, "SiLU mismatch, max_err={}", err);
}

// ========================================================================
// No-BLAS NEON GEMM fallback tests (R13 Android track).
//
// These only compile/run under `--no-default-features`, where `matmul_nn`,
// `matmul_t`, `linear`, and `linear_accumulate` route through the new NEON
// pool-parallel fallback (on aarch64) or the scalar loops (other arches).
// Each is checked against a naive f32 reference on random matrices across edge
// cases (m/n/k = 1, odd sizes, non-multiple-of-4 k, with/without bias).
// ========================================================================
#[cfg(not(feature = "blas"))]
mod noblas_gemm {
    use qwen_asr::kernels;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn f32_pm1(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f32_pm1()).collect()
        }
    }

    // Relative max error: max|a-b| / max(1, max|ref|).
    fn rel_err(got: &[f32], reference: &[f32]) -> f32 {
        let max_ref = reference
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()))
            .max(1.0);
        got.iter()
            .zip(reference)
            .map(|(&g, &r)| (g - r).abs())
            .fold(0.0f32, f32::max)
            / max_ref
    }

    // Naive references (match the non-aarch64 fallback bodies exactly).
    fn ref_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut s = 0.0f32;
                for ki in 0..k {
                    s += a[mi * k + ki] * b[ki * n + ni];
                }
                c[mi * n + ni] = s;
            }
        }
        c
    }
    fn ref_nt(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        seq: usize,
        k: usize,
        out: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; seq * out];
        for s in 0..seq {
            for o in 0..out {
                let mut sum = bias.map_or(0.0, |b| b[o]);
                for ki in 0..k {
                    sum += x[s * k + ki] * w[o * k + ki];
                }
                y[s * out + o] = sum;
            }
        }
        y
    }

    // Cover both the small (inline single-thread) and large (pool-parallel)
    // dispatch paths, odd sizes, and non-multiple-of-4 k.
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (1, 1, 130),
        (3, 1, 129),
        (1, 17, 1),
        (7, 5, 3),
        (2, 4, 200),
        (5, 33, 133),
        (200, 130, 140),
        (1, 1024, 300),
        (130, 1024, 300),
    ];

    #[test]
    fn matmul_nn_matches_reference() {
        let mut rng = Rng::new(0xA5A5_1234);
        for &(m, k, n) in SHAPES {
            let a = rng.vec(m * k);
            let b = rng.vec(k * n);
            let mut c = vec![0.0f32; m * n];
            kernels::matmul_nn(&mut c, &a, &b, m, k, n);
            let want = ref_nn(&a, &b, m, k, n);
            let e = rel_err(&c, &want);
            assert!(e < 1e-4, "matmul_nn m={m} k={k} n={n} rel_err={e}");
        }
    }

    #[test]
    fn matmul_t_matches_reference() {
        let mut rng = Rng::new(0x000B_EEF9);
        for &(m, k, n) in SHAPES {
            // matmul_t: A[m,k] @ B[n,k]^T -> C[m,n]; treat as ref_nt(x=A, w=B, no bias).
            let a = rng.vec(m * k);
            let b = rng.vec(n * k);
            let mut c = vec![0.0f32; m * n];
            kernels::matmul_t(&mut c, &a, &b, m, k, n);
            let want = ref_nt(&a, &b, None, m, k, n);
            let e = rel_err(&c, &want);
            assert!(e < 1e-4, "matmul_t m={m} k={k} n={n} rel_err={e}");
        }
    }

    #[test]
    fn linear_matches_reference() {
        let mut rng = Rng::new(0x00F0_0D77);
        for &(seq, k, out) in SHAPES {
            for use_bias in [false, true] {
                let x = rng.vec(seq * k);
                let w = rng.vec(out * k);
                let bias = if use_bias { Some(rng.vec(out)) } else { None };
                let mut y = vec![0.0f32; seq * out];
                kernels::linear(&mut y, &x, &w, bias.as_deref(), seq, k, out);
                let want = ref_nt(&x, &w, bias.as_deref(), seq, k, out);
                let e = rel_err(&y, &want);
                assert!(
                    e < 1e-4,
                    "linear seq={seq} k={k} out={out} bias={use_bias} rel_err={e}"
                );
            }
        }
    }

    #[test]
    fn linear_accumulate_matches_reference() {
        let mut rng = Rng::new(0x1357_9BDF);
        for &(seq, k, out) in SHAPES {
            for use_bias in [false, true] {
                let x = rng.vec(seq * k);
                let w = rng.vec(out * k);
                let bias = if use_bias { Some(rng.vec(out)) } else { None };
                let y0 = rng.vec(seq * out); // pre-existing residual
                let mut y = y0.clone();
                kernels::linear_accumulate(&mut y, &x, &w, bias.as_deref(), seq, k, out);
                let delta = ref_nt(&x, &w, bias.as_deref(), seq, k, out);
                let want: Vec<f32> = y0.iter().zip(&delta).map(|(a, b)| a + b).collect();
                let e = rel_err(&y, &want);
                assert!(
                    e < 1e-4,
                    "linear_accumulate seq={seq} k={k} out={out} bias={use_bias} rel_err={e}"
                );
            }
        }
    }
}

#[test]
fn test_vec_ops() {
    let n = 256;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();

    // Test scale
    let mut a_copy = a.clone();
    kernels::vec_scale_inplace(&mut a_copy, 2.0, n);
    for i in 0..n {
        assert!((a_copy[i] - a[i] * 2.0).abs() < 1e-5);
    }

    // Test axpy: a += 0.5 * b
    let mut a_copy = a.clone();
    kernels::vec_axpy_inplace(&mut a_copy, &b, 0.5, n);
    for i in 0..n {
        assert!((a_copy[i] - (a[i] + 0.5 * b[i])).abs() < 1e-5);
    }

    // Test scale_add: a = a * 0.9 + b
    let mut a_copy = a.clone();
    let expected: Vec<f32> = (0..n).map(|i| a[i] * 0.9 + b[i]).collect();
    kernels::vec_scale_add(&mut a_copy, &b, 0.9, n);
    let err = max_abs_err(&a_copy, &expected);
    assert!(err < 1e-4, "vec_scale_add mismatch, max_err={}", err);
}
