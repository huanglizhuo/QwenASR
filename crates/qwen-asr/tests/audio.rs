use qwen_asr::audio;

mod common;

#[test]
fn test_load_wav() {
    let path = match common::sample("jfk.wav") {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: jfk.wav sample not found");
            return;
        }
    };
    let samples = audio::load_wav(&path);
    assert!(samples.is_some(), "Should load JFK WAV successfully");
    let s = samples.unwrap();
    // JFK clip is ~11s at 16kHz
    assert!(
        s.len() > 160000,
        "Expected >10s of audio, got {} samples",
        s.len()
    );
    assert!(
        s.len() < 200000,
        "Expected <12.5s of audio, got {} samples",
        s.len()
    );
    // Check values are in reasonable range
    let max_val = s.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_val > 0.01, "Audio should not be silent");
    assert!(
        max_val <= 1.0,
        "Audio values should be normalized to [-1, 1]"
    );
}

#[test]
fn test_mel_spectrogram_shape() {
    let path = match common::sample("jfk.wav") {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: jfk.wav sample not found");
            return;
        }
    };
    let samples = audio::load_wav(&path).unwrap();
    let result = audio::mel_spectrogram(&samples);
    assert!(result.is_some(), "mel_spectrogram should succeed");
    let (mel, n_frames) = result.unwrap();

    // 128 mel bins * n_frames
    assert_eq!(
        mel.len(),
        128 * n_frames,
        "Mel shape should be [128, n_frames]"
    );
    // ~11s of audio at 100fps = ~1100 frames (minus 1 dropped frame)
    assert!(
        n_frames > 1000,
        "Expected >1000 mel frames for 11s audio, got {}",
        n_frames
    );
    assert!(
        n_frames < 1200,
        "Expected <1200 mel frames for 11s audio, got {}",
        n_frames
    );

    // Check values are in reasonable range after normalization: (log10 + 4) / 4
    let min_val = mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_val >= -1.0,
        "Normalized mel values should be >= -1.0, got {}",
        min_val
    );
    assert!(
        max_val <= 2.0,
        "Normalized mel values should be <= 2.0, got {}",
        max_val
    );
}

#[test]
fn test_mel_spectrogram_short_audio() {
    // Generate 0.5s of 440Hz sine at 16kHz
    let n = 8000;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();

    let result = audio::mel_spectrogram(&samples);
    assert!(
        result.is_some(),
        "mel_spectrogram should handle short audio"
    );
    let (_mel, n_frames) = result.unwrap();
    // 0.5s at 100fps = ~50 frames
    assert!(n_frames > 20, "Expected >20 frames for 0.5s audio");
    assert!(n_frames < 80, "Expected <80 frames for 0.5s audio");
}

#[test]
fn test_compact_silence() {
    // Create audio with silence gap: 0.5s tone, 2s silence, 0.5s tone
    let sr = 16000;
    let tone_samples = sr / 2; // 0.5s
    let silence_samples = sr * 2; // 2s
    let total = 2 * tone_samples + silence_samples;
    let mut samples = vec![0.0f32; total];

    // Fill tone regions
    for i in 0..tone_samples {
        let v = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.3;
        samples[i] = v;
        samples[tone_samples + silence_samples + i] = v;
    }

    let compacted = audio::compact_silence(&samples);
    // Compacted should be shorter than original (silence removed)
    assert!(
        compacted.len() < samples.len(),
        "Compacted audio should be shorter: {} vs {}",
        compacted.len(),
        samples.len()
    );
    // But not too short (should keep some silence and both tone regions)
    assert!(
        compacted.len() > tone_samples,
        "Compacted audio should keep at least the tone regions"
    );
}

/// Serial reference: the original single-threaded resampler loop, verbatim.
fn resample_reference(samples: &[f32], from_rate: i32, to_rate: i32) -> Vec<f32> {
    fn bessel_i0(x: f64) -> f64 {
        let (mut sum, mut term, xx) = (1.0, 1.0, x * x);
        for k in 1..=20 {
            term *= xx / (4.0 * k as f64 * k as f64);
            sum += term;
        }
        sum
    }
    let n_frames = samples.len();
    let new_n = (n_frames as i64 * to_rate as i64 / from_rate as i64) as usize;
    let (sinc_half, kaiser_beta) = (16, 6.0f64);
    let ratio = to_rate as f64 / from_rate as f64;
    let cutoff = if ratio < 1.0 { ratio } else { 1.0 };
    let inv_i0_beta = 1.0 / bessel_i0(kaiser_beta);
    (0..new_n)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let center = src_pos as i32;
            let (mut acc, mut wsum) = (0.0f64, 0.0f64);
            for j in (center - sinc_half + 1)..=(center + sinc_half) {
                let d = j as f64 - src_pos;
                let x = d * cutoff;
                let s = if x.abs() < 1e-9 {
                    1.0
                } else {
                    (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                let npos = d / sinc_half as f64;
                let w = if npos <= -1.0 || npos >= 1.0 {
                    0.0
                } else {
                    bessel_i0(kaiser_beta * (1.0 - npos * npos).sqrt()) * inv_i0_beta
                };
                let coeff = s * w * cutoff;
                if j >= 0 && (j as usize) < n_frames {
                    acc += samples[j as usize] as f64 * coeff;
                }
                wsum += coeff;
            }
            if wsum > 1e-9 {
                (acc / wsum) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// The parallel resampler path (long inputs) must be bit-identical to the
/// original serial loop.
#[test]
fn test_resample_parallel_bit_identical() {
    let n = 44_100 * 3; // 3 s @ 44.1 kHz -> 48k outputs, takes the parallel path
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5 + ((i as f32) * 0.0007).cos() * 0.25)
        .collect();
    for (from, to) in [(44_100, 16_000), (48_000, 16_000), (8_000, 16_000)] {
        let got = audio::resample(&x, from, to);
        let want = resample_reference(&x, from, to);
        assert_eq!(got.len(), want.len());
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "resample {from}->{to} diverged from the serial reference"
        );
    }
}
