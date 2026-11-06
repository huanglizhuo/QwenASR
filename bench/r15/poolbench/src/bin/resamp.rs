use std::time::Instant;
fn bessel_i0(x: f64) -> f64 {
    let mut s = 1.0;
    let mut t = 1.0;
    let xx = x * x;
    for k in 1..=20 {
        t *= xx / (4.0 * k as f64 * k as f64);
        s += t;
    }
    s
}
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
/// Polyphase version of qwen_asr::audio::resample: same Kaiser-sinc formula, coefficients tabulated per phase.
fn resample_poly(samples: &[f32], from: i32, to: i32) -> Vec<f32> {
    let n = samples.len();
    let new_n = (n as i64 * to as i64 / from as i64) as usize;
    let g = gcd(from as i64, to as i64);
    let up = (to as i64 / g) as usize;
    let down = (from as i64 / g) as usize;
    let half = 16i64;
    let beta = 6.0f64;
    let ratio = to as f64 / from as f64;
    let cutoff = if ratio < 1.0 { ratio } else { 1.0 };
    let inv = 1.0 / bessel_i0(beta);
    // table[phase][tap], tap t <-> j = center - half + 1 + t
    let taps = (2 * half) as usize;
    let mut tab = vec![0f64; up * taps];
    let mut wsum = vec![0f64; up];
    for p in 0..up {
        let frac = p as f64 / up as f64;
        for t in 0..taps {
            let k = t as i64 - half + 1;
            let d = k as f64 - frac;
            let x = d * cutoff;
            let s = if x.abs() < 1e-9 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let np = d / half as f64;
            let w = if np <= -1.0 || np >= 1.0 {
                0.0
            } else {
                bessel_i0(beta * (1.0 - np * np).sqrt()) * inv
            };
            let c = s * w * cutoff;
            tab[p * taps + t] = c;
            wsum[p] += c;
        }
    }
    let mut out = vec![0f32; new_n];
    for (i, o) in out.iter_mut().enumerate() {
        let num = i as i64 * down as i64;
        let center = num / up as i64;
        let p = (num % up as i64) as usize;
        let mut acc = 0f64;
        let base = center - half + 1;
        for t in 0..taps {
            let j = base + t as i64;
            if j >= 0 && (j as usize) < n {
                acc += samples[j as usize] as f64 * tab[p * taps + t];
            }
        }
        *o = if wsum[p] > 1e-9 {
            (acc / wsum[p]) as f32
        } else {
            0.0
        };
    }
    out
}
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(&path).unwrap();
    // shortcut: canonical 44.1 kHz mono PCM16 WAV only; use a WAV parser for other fixtures.
    assert!(data.len() >= 44);
    assert_eq!(&data[..4], b"RIFF");
    assert_eq!(&data[8..16], b"WAVEfmt ");
    assert_eq!(u32::from_le_bytes(data[16..20].try_into().unwrap()), 16);
    assert_eq!(&data[20..24], &[1, 0, 1, 0]);
    assert_eq!(u32::from_le_bytes(data[24..28].try_into().unwrap()), 44100);
    assert_eq!(u16::from_le_bytes(data[34..36].try_into().unwrap()), 16);
    assert_eq!(&data[36..40], b"data");
    assert_eq!(
        u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize,
        data.len() - 44
    );
    let pcm = &data[44..];
    let s: Vec<f32> = pcm
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    let t = Instant::now();
    let a = qwen_asr::audio::resample(&s, 44100, 16000);
    let ta = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let b = resample_poly(&s, 44100, 16000);
    let tb = t.elapsed().as_secs_f64() * 1e3;
    let mut ndiff = 0;
    let mut maxd = 0f32;
    let mut maxulp = 0i64;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            ndiff += 1;
            maxd = maxd.max((x - y).abs());
            maxulp = maxulp.max((x.to_bits() as i64 - y.to_bits() as i64).abs());
        }
    }
    println!("current resample: {ta:.1} ms   polyphase: {tb:.1} ms   ({} out samples; differing: {ndiff}, max |diff| {maxd:e}, max ulp {maxulp})", a.len());
}
