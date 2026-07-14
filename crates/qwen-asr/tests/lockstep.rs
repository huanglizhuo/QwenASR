//! Acceptance test for the R12-E3 lockstep batched segment decode scheduler.
//!
//! A multi-segment transcription through the lockstep path must be
//! byte-identical to the same audio decoded segment-by-segment through the
//! sequential serial loop: the batched decode step is exact per session (INT8
//! integer accumulation, R12-E2), and the per-segment mel/encode/prefill runs
//! in the same single-threaded worker environment as the L3 parallel path.

use qwen_asr::context::QwenCtx;
use qwen_asr::kernels;
use qwen_asr::{audio, transcribe};

use std::sync::Mutex;

// The kernel thread pool is a global singleton; serialize against the other
// model-loading integration tests.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Resolve a path that exists either relative to the workspace root or to the
/// crate dir (cargo runs test binaries with cwd = package dir).
fn resolve(rel: &str) -> Option<String> {
    for prefix in ["", "../../"] {
        let p = format!("{prefix}{rel}");
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    None
}

#[test]
fn lockstep_multi_segment_matches_serial() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let model_dir = match resolve("qwen3-asr-0.6b") {
        Some(d) if std::path::Path::new(&d).join("model.safetensors").exists() => d,
        _ => {
            eprintln!("Skipping lockstep test: model not downloaded");
            return;
        }
    };
    let wav = match resolve("bench/samples/audio.wav") {
        Some(w) => w,
        None => {
            eprintln!("Skipping lockstep test: bench audio not found");
            return;
        }
    };

    kernels::set_verbose(0);
    let samples = audio::load_wav(&wav).expect("failed to load wav");

    // Short segments force a multi-segment split of the bench clip.
    let run = |seg_workers_serial: bool, lockstep: bool, batch: Option<usize>| {
        if seg_workers_serial {
            std::env::set_var("QWEN_ASR_SEG_WORKERS", "1");
        } else {
            std::env::remove_var("QWEN_ASR_SEG_WORKERS");
        }
        if lockstep {
            std::env::remove_var("QWEN_ASR_SEG_LOCKSTEP");
        } else {
            std::env::set_var("QWEN_ASR_SEG_LOCKSTEP", "0");
        }
        match batch {
            Some(b) => std::env::set_var("QWEN_ASR_SEG_BATCH", b.to_string()),
            None => std::env::remove_var("QWEN_ASR_SEG_BATCH"),
        }
        let mut ctx = QwenCtx::load(&model_dir).expect("load ctx");
        ctx.segment_sec = 3.0;
        transcribe::transcribe_audio(&mut ctx, &samples).expect("transcribe")
    };

    // Reference: sequential segment-by-segment serial decode, single-threaded
    // kernels (the environment every lockstep segment's prefill runs in, and
    // the numeric baseline the batched decode step must reproduce exactly).
    kernels::set_threads(1);
    let reference = run(true, false, None);

    // Lockstep at nt = 1 (batched region runs inline on one thread).
    let lockstep_nt1 = run(false, true, None);
    assert_eq!(
        lockstep_nt1, reference,
        "lockstep (nt=1) must match sequential serial decode"
    );

    // Lockstep with the threaded pool: the batched INT8 decode step is exact
    // regardless of thread count or batch composition, and prefill workers are
    // pinned single-threaded, so the transcript must not change.
    kernels::set_threads(4);
    let lockstep_nt4 = run(false, true, None);
    assert_eq!(
        lockstep_nt4, reference,
        "lockstep (nt=4) must match sequential serial decode"
    );

    // A small forced batch exercises live-set shrink + dynamic session refill.
    let lockstep_b2 = run(false, true, Some(2));
    assert_eq!(
        lockstep_b2, reference,
        "lockstep (B=2, refill) must match sequential serial decode"
    );

    std::env::remove_var("QWEN_ASR_SEG_WORKERS");
    std::env::remove_var("QWEN_ASR_SEG_LOCKSTEP");
    std::env::remove_var("QWEN_ASR_SEG_BATCH");
    kernels::set_threads(kernels::get_num_cpus());
}
