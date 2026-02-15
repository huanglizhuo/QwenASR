use q_asr::audio;

#[test]
fn test_load_wav() {
    let path = "/tmp/qwen-asr-ref/samples/jfk.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping test: {} not found", path);
        return;
    }
    let samples = audio::load_wav(path);
    assert!(samples.is_some(), "Should load JFK WAV successfully");
    let s = samples.unwrap();
    // JFK clip is ~11s at 16kHz
    assert!(s.len() > 160000, "Expected >10s of audio, got {} samples", s.len());
    assert!(s.len() < 200000, "Expected <12.5s of audio, got {} samples", s.len());
    // Check values are in reasonable range
    let max_val = s.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_val > 0.01, "Audio should not be silent");
    assert!(max_val <= 1.0, "Audio values should be normalized to [-1, 1]");
}

#[test]
fn test_mel_spectrogram_shape() {
    let path = "/tmp/qwen-asr-ref/samples/jfk.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping test: {} not found", path);
        return;
    }
    let samples = audio::load_wav(path).unwrap();
    let result = audio::mel_spectrogram(&samples);
    assert!(result.is_some(), "mel_spectrogram should succeed");
    let (mel, n_frames) = result.unwrap();

    // 128 mel bins * n_frames
    assert_eq!(mel.len(), 128 * n_frames, "Mel shape should be [128, n_frames]");
    // ~11s of audio at 100fps = ~1100 frames (minus 1 dropped frame)
    assert!(n_frames > 1000, "Expected >1000 mel frames for 11s audio, got {}", n_frames);
    assert!(n_frames < 1200, "Expected <1200 mel frames for 11s audio, got {}", n_frames);

    // Check values are in reasonable range after normalization: (log10 + 4) / 4
    let min_val = mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(min_val >= -1.0, "Normalized mel values should be >= -1.0, got {}", min_val);
    assert!(max_val <= 2.0, "Normalized mel values should be <= 2.0, got {}", max_val);
}

#[test]
fn test_mel_spectrogram_short_audio() {
    // Generate 0.5s of 440Hz sine at 16kHz
    let n = 8000;
    let samples: Vec<f32> = (0..n).map(|i| {
        (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5
    }).collect();

    let result = audio::mel_spectrogram(&samples);
    assert!(result.is_some(), "mel_spectrogram should handle short audio");
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
    assert!(compacted.len() < samples.len(),
        "Compacted audio should be shorter: {} vs {}", compacted.len(), samples.len());
    // But not too short (should keep some silence and both tone regions)
    assert!(compacted.len() > tone_samples,
        "Compacted audio should keep at least the tone regions");
}
