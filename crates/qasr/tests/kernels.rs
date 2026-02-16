use qasr::kernels;
use qasr::kernels::generic;

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn test_bf16_roundtrip() {
    let values: Vec<f32> = vec![0.0, 1.0, -1.0, 3.14, -2.71, 100.5, 0.001, -0.001];
    for &v in &values {
        let bf16 = ((v.to_bits() + 0x8000) >> 16) as u16;
        let back = kernels::bf16_to_f32(bf16);
        assert!((v - back).abs() < 0.02 * v.abs().max(1.0),
            "BF16 roundtrip failed: {} -> {} -> {}", v, bf16, back);
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
    generic::bf16_matvec_fused(&mut y_generic, &x, w_bf16.as_ptr(), Some(&bias), in_dim, out_dim);

    let mut y_dispatch = vec![0.0f32; out_dim];
    kernels::linear_nobias_bf16(&mut y_dispatch, &x, w_bf16.as_ptr(), 1, in_dim, out_dim);
    for i in 0..out_dim {
        y_dispatch[i] += bias[i];
    }

    let err = max_abs_err(&y_generic, &y_dispatch);
    assert!(err < 0.01, "bf16 matvec dispatch vs generic mismatch: max_err={}", err);
}

#[test]
fn test_dot_f32_vs_generic() {
    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32) * 0.002 - 1.0).collect();

    let result_generic = generic::dot_f32(&a, &b, n);
    let result_dispatch = kernels::dot_f32(&a, &b, n);

    let err = (result_generic - result_dispatch).abs();
    assert!(err < 0.01 * result_generic.abs().max(1.0),
        "dot_f32 mismatch: generic={}, dispatch={}, err={}", result_generic, result_dispatch, err);
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
    assert!((rms - 1.0).abs() < 0.01, "RMS norm output should have RMS ~1.0, got {}", rms);
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
    assert!(mean.abs() < 0.01, "LayerNorm mean should be ~0, got {}", mean);
    assert!((var - 1.0).abs() < 0.02, "LayerNorm variance should be ~1, got {}", var);
}

#[test]
fn test_softmax() {
    let n = 10;
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    kernels::softmax(&mut x, 1, n);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0, got {}", sum);
    for i in 1..n {
        assert!(x[i] >= x[i - 1], "Softmax should be monotonically increasing");
    }
    assert!(x[0] > 0.0 && x[9] > 0.0, "All softmax values should be positive");
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
