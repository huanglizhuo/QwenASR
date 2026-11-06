// Would a per-chunk (100-frame) conv-stem cache hit in the streaming push path?
// Compare mel chunks of the partial window [ws, cursor) vs a later, longer slice.
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let s = qwen_asr::audio::load_wav(&path).unwrap();
    let sr = 16000usize;
    let mut tot = 0;
    let mut hit = 0;
    // window starts every 8 s; partial encodes at +2,+4,+6 s, full window at +8 s (2 s chunks, throttle 2 -> every 4 s)
    for ws in (0..s.len().saturating_sub(8 * sr)).step_by(8 * sr) {
        for (a, b) in [(4usize, 8usize), (2, 6), (6, 8), (4, 6)] {
            let (m1, f1) = qwen_asr::audio::mel_spectrogram(&s[ws..ws + a * sr]).unwrap();
            let (m2, f2) = qwen_asr::audio::mel_spectrogram(&s[ws..ws + b * sr]).unwrap();
            let full_chunks = f1 / 100; // chunks fully inside the shorter slice
            for c in 0..full_chunks {
                tot += 1;
                let same = (0..128).all(|m| {
                    (c * 100..(c + 1) * 100)
                        .all(|t| m1[m * f1 + t].to_bits() == m2[m * f2 + t].to_bits())
                });
                if same {
                    hit += 1;
                }
            }
        }
    }
    println!("chunk mel bit-identical: {hit}/{tot}");
}
