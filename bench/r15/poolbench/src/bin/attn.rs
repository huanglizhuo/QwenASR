// Single-token GQA attention: current kernel (kernels::causal_attention, 1 thread) vs a 3-pass
// restructure built from the SAME public helpers (dot_f32 / vec_scale_add / vec_axpy_inplace /
// vec_scale_inplace) in the same per-head operation order -> must be bit-identical.
use qwen_asr::kernels;
use std::time::Instant;
extern "C" {
    fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32;
}
const HD: usize = 128;
const NH: usize = 16;
const NKV: usize = 8;
const G: usize = NH / NKV;
fn rng(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}
fn three_pass(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hs: usize,
    n: usize,
    scale: f32,
    scores: &mut [f32],
    coef: &mut [f32],
    kind: &mut [u8],
) {
    for kv in 0..NKV {
        let kh = &k[kv * hs..];
        let vh = &v[kv * hs..];
        for g in 0..G {
            let h = kv * G + g;
            let qh = &q[h * HD..(h + 1) * HD];
            // pass 1: all scores (independent -> ILP across positions)
            for j in 0..n {
                scores[j] = kernels::dot_f32(qh, &kh[j * HD..(j + 1) * HD], HD) * scale;
            }
            // pass 2: online-softmax scalar recurrence, identical order
            let mut m = -1e30f32;
            let mut s = 0.0f32;
            for j in 0..n {
                let sc = scores[j];
                if sc > m {
                    let c = (m - sc).exp();
                    s = s * c + 1.0;
                    coef[j] = c;
                    kind[j] = 1;
                    m = sc;
                } else {
                    let w = (sc - m).exp();
                    s += w;
                    coef[j] = w;
                    kind[j] = 0;
                }
            }
            // pass 3: V accumulation, identical op sequence
            let o = &mut out[h * HD..(h + 1) * HD];
            for x in o.iter_mut() {
                *x = 0.0;
            }
            for j in 0..n {
                let vr = &vh[j * HD..(j + 1) * HD];
                if kind[j] == 1 {
                    kernels::vec_scale_add(o, vr, coef[j], HD);
                } else {
                    kernels::vec_axpy_inplace(o, vr, coef[j], HD);
                }
            }
            if s > 0.0 {
                kernels::vec_scale_inplace(o, 1.0 / s, HD);
            }
        }
    }
}
fn main() {
    let qos: u32 = match std::env::args().nth(1).as_deref() {
        Some("e") => 0x09,
        _ => 0x21,
    }; // BACKGROUND (E-core) / USER_INTERACTIVE
    unsafe {
        pthread_set_qos_class_self_np(qos, 0);
    }
    kernels::set_threads(1);
    let max_seq = 8192;
    let hs = max_seq * HD;
    let mut seed = 0x1234_5678_9abc_def1u64;
    let k: Vec<f32> = (0..NKV * hs).map(|_| rng(&mut seed)).collect();
    let v: Vec<f32> = (0..NKV * hs).map(|_| rng(&mut seed)).collect();
    let q: Vec<f32> = (0..NH * HD).map(|_| rng(&mut seed) * 4.0).collect();
    let scale = 1.0 / (HD as f32).sqrt();
    let (mut sc, mut cf, mut kd) = (vec![0f32; max_seq], vec![0f32; max_seq], vec![0u8; max_seq]);
    println!(
        "core class: {}",
        if qos == 0x09 {
            "E (QOS_BACKGROUND)"
        } else {
            "P (QOS_USER_INTERACTIVE)"
        }
    );
    for &n in &[75usize, 420, 900, 1770, 4000] {
        let mut a = vec![0f32; NH * HD];
        let mut b = vec![0f32; NH * HD];
        let reps = (200_000 / n).max(20);
        let mut ta = f64::MAX;
        let mut tb = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..reps {
                kernels::causal_attention(
                    &mut a,
                    &q,
                    k.as_ptr(),
                    v.as_ptr(),
                    hs,
                    1,
                    n,
                    NH,
                    NKV,
                    HD,
                    scale,
                    n - 1,
                );
            }
            ta = ta.min(t.elapsed().as_secs_f64() / reps as f64);
            let t = Instant::now();
            for _ in 0..reps {
                three_pass(&mut b, &q, &k, &v, hs, n, scale, &mut sc, &mut cf, &mut kd);
            }
            tb = tb.min(t.elapsed().as_secs_f64() / reps as f64);
        }
        let same = a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "three-pass attention changed output at context {n}");
        // per decode token on ONE thread owning one KV group: 28 layers x (time/8 groups)
        println!("ctx {n:>5}: current {:>7.1} us  3-pass {:>7.1} us  speedup {:.2}x  bit-identical={same}  | per-token(1 group x 28 layers): {:.2} -> {:.2} ms",
            ta * 1e6, tb * 1e6, ta / tb, ta / 8.0 * 28.0 * 1e3, tb / 8.0 * 28.0 * 1e3);
    }
}
