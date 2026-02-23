use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe;
use qwen_asr::align;
use qwen_asr::kernels;

use std::sync::Mutex;

// Global mutex to serialize regression tests — the thread pool is a global singleton
// and doesn't support concurrent callers from different threads.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn setup_model() -> Option<QwenCtx> {
    let model_dir = "qwen3-asr-0.6b";
    if !std::path::Path::new(model_dir).join("model.safetensors").exists() {
        eprintln!("Skipping regression test: model not downloaded at {}", model_dir);
        return None;
    }
    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());
    QwenCtx::load(model_dir)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 0..=a.len() { dp[i][0] = i; }
    for j in 0..=b.len() { dp[0][j] = j; }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[a.len()][b.len()]
}

#[test]
fn test_offline_jfk() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut ctx = match setup_model() {
        Some(c) => c,
        None => return,
    };
    let wav = "/tmp/qwen-asr-ref/samples/jfk.wav";
    if !std::path::Path::new(wav).exists() {
        eprintln!("Skipping: {} not found", wav);
        return;
    }

    let result = transcribe::transcribe(&mut ctx, wav);
    assert!(result.is_some(), "Offline transcription should succeed");
    let text = result.unwrap();

    let expected = "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.";
    let dist = levenshtein(&text.to_lowercase(), &expected.to_lowercase());
    assert!(dist <= 5,
        "JFK offline: Levenshtein distance {} > 5\nExpected: {}\nGot: {}", dist, expected, text);
}

#[test]
fn test_offline_test_speech() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut ctx = match setup_model() {
        Some(c) => c,
        None => return,
    };
    let wav = "/tmp/qwen-asr-ref/samples/test_speech.wav";
    if !std::path::Path::new(wav).exists() {
        eprintln!("Skipping: {} not found", wav);
        return;
    }

    let result = transcribe::transcribe(&mut ctx, wav);
    assert!(result.is_some(), "Offline transcription should succeed");
    let text = result.unwrap();

    // Allow some tolerance for ASR output
    assert!(text.to_lowercase().contains("hello"),
        "Should contain 'hello', got: {}", text);
    assert!(text.to_lowercase().contains("speech"),
        "Should contain 'speech', got: {}", text);
}

#[test]
fn test_segmented_mode() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut ctx = match setup_model() {
        Some(c) => c,
        None => return,
    };
    let wav = "/tmp/qwen-asr-ref/samples/night_of_the_living_dead_1968/45s_dont_be_afraid_of_me.wav";
    if !std::path::Path::new(wav).exists() {
        eprintln!("Skipping: {} not found", wav);
        return;
    }

    ctx.segment_sec = 30.0;
    let result = transcribe::transcribe(&mut ctx, wav);
    assert!(result.is_some(), "Segmented transcription should succeed");
    let text = result.unwrap();

    // Check key phrases are present
    let lower = text.to_lowercase();
    assert!(lower.contains("afraid"), "Should contain 'afraid', got: {}", text);
    assert!(lower.contains("helen"), "Should contain 'helen', got: {}", text);
}

#[test]
fn test_streaming_mode() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut ctx = match setup_model() {
        Some(c) => c,
        None => return,
    };

    let wav = "/tmp/qwen-asr-ref/samples/jfk.wav";
    if !std::path::Path::new(wav).exists() {
        eprintln!("Skipping: {} not found", wav);
        return;
    }

    let samples = qwen_asr::audio::load_wav(wav);
    assert!(samples.is_some());
    let samples = samples.unwrap();

    let result = transcribe::transcribe_stream(&mut ctx, &samples);
    assert!(result.is_some(), "Streaming transcription should succeed");
    let text = result.unwrap();

    let expected = "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.";
    let dist = levenshtein(&text.to_lowercase(), &expected.to_lowercase());
    assert!(dist <= 10,
        "JFK streaming: Levenshtein distance {} > 10\nExpected: {}\nGot: {}", dist, expected, text);
}

fn load_audio_reference() -> String {
    let path = "bench/samples/audio.txt";
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[test]
fn test_offline_audio_wav() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut ctx = match setup_model() {
        Some(c) => c,
        None => return,
    };
    let wav = "bench/samples/audio.wav";
    if !std::path::Path::new(wav).exists() {
        eprintln!("Skipping: {} not found", wav);
        return;
    }
    let reference = load_audio_reference();
    if reference.is_empty() {
        eprintln!("Skipping: bench/samples/audio.txt not found or empty");
        return;
    }

    let result = transcribe::transcribe(&mut ctx, wav);
    assert!(result.is_some(), "Offline transcription should succeed");
    let text = result.unwrap();

    let dist = levenshtein(&text.to_lowercase(), &reference.to_lowercase());
    assert!(dist <= 5,
        "audio.wav offline: Levenshtein distance {} > 5\nExpected: {}\nGot: {}", dist, reference, text);
}

#[test]
fn test_segmented_audio_wav() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut ctx = match setup_model() {
        Some(c) => c,
        None => return,
    };
    let wav = "bench/samples/audio.wav";
    if !std::path::Path::new(wav).exists() {
        eprintln!("Skipping: {} not found", wav);
        return;
    }
    let reference = load_audio_reference();
    if reference.is_empty() {
        eprintln!("Skipping: bench/samples/audio.txt not found or empty");
        return;
    }

    ctx.segment_sec = 30.0;
    let result = transcribe::transcribe(&mut ctx, wav);
    assert!(result.is_some(), "Segmented transcription should succeed");
    let text = result.unwrap();

    let dist = levenshtein(&text.to_lowercase(), &reference.to_lowercase());
    assert!(dist <= 10,
        "audio.wav segmented: Levenshtein distance {} > 10\nExpected: {}\nGot: {}", dist, reference, text);
}

#[test]
fn test_streaming_audio_wav() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut ctx = match setup_model() {
        Some(c) => c,
        None => return,
    };
    let wav = "bench/samples/audio.wav";
    if !std::path::Path::new(wav).exists() {
        eprintln!("Skipping: {} not found", wav);
        return;
    }
    let reference = load_audio_reference();
    if reference.is_empty() {
        eprintln!("Skipping: bench/samples/audio.txt not found or empty");
        return;
    }

    let samples = qwen_asr::audio::load_wav(wav);
    assert!(samples.is_some(), "Should load bench/samples/audio.wav");
    let samples = samples.unwrap();

    let result = transcribe::transcribe_stream(&mut ctx, &samples);
    assert!(result.is_some(), "Streaming transcription should succeed");
    let text = result.unwrap();

    let dist = levenshtein(&text.to_lowercase(), &reference.to_lowercase());
    assert!(dist <= 10,
        "audio.wav streaming: Levenshtein distance {} > 10\nExpected: {}\nGot: {}", dist, reference, text);
}

fn setup_aligner_model() -> Option<QwenCtx> {
    let model_dir = "qwen3-aligner-0.6b";
    if !std::path::Path::new(model_dir).join("model.safetensors").exists() {
        eprintln!("Skipping alignment test: aligner model not downloaded at {}", model_dir);
        return None;
    }
    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());
    QwenCtx::load(model_dir)
}

#[test]
fn test_forced_align() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut ctx = match setup_aligner_model() {
        Some(c) => c,
        None => return,
    };

    let wav = "audio.wav";
    if !std::path::Path::new(wav).exists() {
        eprintln!("Skipping: {} not found", wav);
        return;
    }

    let samples = qwen_asr::audio::load_wav(wav);
    assert!(samples.is_some(), "Should load audio.wav");
    let samples = samples.unwrap();

    let text = "Shenyang, a city with its own small secrets. Since you are going there, I expect you to keep your eyes open. Some things are worth bringing back, and you know disappointing me is rarely a wise decision.";
    let results = align::forced_align(&mut ctx, &samples, text, "English");
    assert!(results.is_some(), "Forced alignment should succeed");
    let results = results.unwrap();

    // Word count should match whitespace-split of input text
    let expected_words: Vec<&str> = text.split_whitespace().collect();
    assert_eq!(results.len(), expected_words.len(),
        "Word count mismatch: expected {}, got {}", expected_words.len(), results.len());

    // All timestamps should be non-negative
    for r in &results {
        assert!(r.start_ms >= 0.0, "Negative start_ms for '{}': {}", r.text, r.start_ms);
        assert!(r.end_ms >= 0.0, "Negative end_ms for '{}': {}", r.text, r.end_ms);
    }

    // Timestamps should be generally non-decreasing (each word starts >= previous word's start)
    for i in 1..results.len() {
        assert!(results[i].start_ms >= results[i - 1].start_ms,
            "Non-monotonic start_ms at word '{}' ({}): {} < {}",
            results[i].text, i, results[i].start_ms, results[i - 1].start_ms);
    }
}
