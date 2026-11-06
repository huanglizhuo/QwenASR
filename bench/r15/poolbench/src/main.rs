use qwen_asr::kernels;
use std::time::Instant;
fn spin_us(us: f64) {
    let t = Instant::now();
    while t.elapsed().as_secs_f64() * 1e6 < us {}
}
fn main() {
    let nt: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    kernels::set_threads(nt);
    let n = 1usize << 18; // smallest size that takes the pooled path (8 items of 32K)
    let src = vec![0x3f80u16; n];
    let mut dst = vec![0f32; n];
    // serial reference cost
    let t = Instant::now();
    for _ in 0..200 {
        kernels::bf16_to_f32_buf(&mut dst, &src);
    }
    let serial = t.elapsed().as_secs_f64() * 1e6 / 200.0;
    for _ in 0..200 {
        kernels::bf16_to_f32_buf_parallel(&mut dst, &src);
    } // warm
    println!("threads={nt}  serial convert of {n} elems = {serial:.1} us");
    for gap in [0.0, 5.0, 20.0, 50.0, 100.0, 300.0, 1000.0] {
        let mut v = Vec::new();
        for _ in 0..300 {
            spin_us(gap);
            let t = Instant::now();
            kernels::bf16_to_f32_buf_parallel(&mut dst, &src);
            v.push(t.elapsed().as_secs_f64() * 1e6);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  idle gap {gap:>6.0} us before dispatch -> pooled call p50 {:.1} us, p90 {:.1} us",
            v[150], v[270]
        );
    }
    assert!(dst.iter().all(|&v| v == 1.0));
    std::hint::black_box(&dst);
}
