//! R14-B1 integration test: the offline prompt-lookup speculative decode path
//! must be byte-identical to the pure single-token decode path.
//!
//! The offline serial decode loop (`decode_segment_core`) speculates with
//! n-gram drafts from the segment's own committed-token history, verified by
//! `decoder_forward_verify`. Correctness is by construction (a greedy verifier
//! only ever commits the sequential greedy argmax), so any transcript
//! difference between `QWEN_ASR_VERIFY=1` and `=0` is a bug.
//!
//! Both the single-segment path (offline default) and the multi-segment
//! serial fallback (`-S`-style splitting forced onto one worker, which still
//! routes each segment through the same serial decode loop) are exercised.

use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe;

use std::sync::Mutex;

// The kernel thread pool is a global singleton; serialize against the other
// model-loading integration tests (each test file owns its own static mutex).
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn resolve(rel: &str) -> Option<String> {
    for prefix in ["", "../../"] {
        let p = format!("{prefix}{rel}");
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    None
}

fn load_ctx() -> Option<QwenCtx> {
    let model_dir = match resolve("qwen3-asr-0.6b") {
        Some(d) if std::path::Path::new(&d).join("model.safetensors").exists() => d,
        _ => {
            eprintln!("Skipping offline-verify test: model not downloaded");
            return None;
        }
    };
    qwen_asr::kernels::set_verbose(0);
    qwen_asr::kernels::set_threads(qwen_asr::kernels::get_num_cpus());
    QwenCtx::load(&model_dir)
}

fn load_audio() -> Option<Vec<f32>> {
    let wav = resolve("bench/samples/audio.wav")?;
    qwen_asr::audio::load_wav(&wav)
}

/// Offline transcription with the verifier forced on/off. `segment_sec = 0`
/// keeps the audio as a single segment (the serial decode loop); a positive
/// value splits it, with parallel scheduling disabled via env so every
/// segment still takes the serial verify-capable loop.
fn run_offline(samples: &[f32], verify: bool, segment_sec: f32) -> String {
    std::env::set_var("QWEN_ASR_VERIFY", if verify { "1" } else { "0" });
    if segment_sec > 0.0 {
        // Force the serial schedule: one worker, lockstep off.
        std::env::set_var("QWEN_ASR_SEG_LOCKSTEP", "0");
        std::env::set_var("QWEN_ASR_SEG_WORKERS", "1");
    }
    let mut ctx = load_ctx().expect("ctx");
    ctx.segment_sec = segment_sec;
    let out = transcribe::transcribe_audio(&mut ctx, samples).unwrap_or_default();
    if segment_sec > 0.0 {
        std::env::remove_var("QWEN_ASR_SEG_LOCKSTEP");
        std::env::remove_var("QWEN_ASR_SEG_WORKERS");
    }
    out
}

#[test]
fn offline_single_segment_verify_on_off_identical() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let samples = match (load_ctx(), load_audio()) {
        (Some(_), Some(s)) => s,
        _ => return, // skip (model or audio missing)
    };

    let on = run_offline(&samples, true, 0.0);
    let off = run_offline(&samples, false, 0.0);
    std::env::remove_var("QWEN_ASR_VERIFY");

    assert_eq!(
        on, off,
        "offline: QWEN_ASR_VERIFY on/off transcripts differ\nON:  {on:?}\nOFF: {off:?}"
    );
    assert!(!on.is_empty(), "offline produced empty transcript");
    eprintln!("offline verify on/off identical ({} bytes)", on.len());
}

#[test]
fn offline_segmented_serial_verify_on_off_identical() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let samples = match (load_ctx(), load_audio()) {
        (Some(_), Some(s)) => s,
        _ => return, // skip (model or audio missing)
    };

    // 10 s segments over the 28.2 s sample => 3 segments through the serial
    // loop, each exercising the verifier with a fresh prompt-lookup history.
    let on = run_offline(&samples, true, 10.0);
    let off = run_offline(&samples, false, 10.0);
    std::env::remove_var("QWEN_ASR_VERIFY");

    assert_eq!(
        on, off,
        "offline -S10 serial: QWEN_ASR_VERIFY on/off transcripts differ\nON:  {on:?}\nOFF: {off:?}"
    );
    assert!(!on.is_empty(), "offline -S10 produced empty transcript");
    eprintln!(
        "offline -S10 serial verify on/off identical ({} bytes)",
        on.len()
    );
}
