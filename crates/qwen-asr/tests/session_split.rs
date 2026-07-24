//! Acceptance test for the R12-E1 shared-model / per-session split (F27).
//!
//! Demonstrates that two independent [`QwenCtx`] sessions created from one
//! shared [`QwenModel`] can transcribe different audio *concurrently* on two
//! threads and produce byte-identical transcripts to two sequential, fully
//! independent `QwenCtx::load` contexts. This is the substrate the stage-2
//! lockstep batched-segment decoder is built on.

use qwen_asr::audio;
use qwen_asr::context::{QwenCtx, QwenModel};
use qwen_asr::kernels;
use qwen_asr::transcribe;

use std::sync::Mutex;

// The kernel thread pool is a global singleton; serialize against the other
// model-loading integration tests.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

mod common;

fn load_wav(path: &str) -> Vec<f32> {
    audio::load_wav(path).expect("failed to load wav")
}

#[test]
fn two_sessions_one_model_match_sequential() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let model_dir = match common::model_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skipping session-split test: model not downloaded");
            return;
        }
    };
    let (wav_a, wav_b) = match (
        common::sample("audio.wav"),
        common::resolve("bench/long/samples/long-2min.wav"),
    ) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("Skipping session-split test: bench audio not found");
            return;
        }
    };

    kernels::set_verbose(0);
    // Force single-threaded kernels so the two concurrent sessions never touch
    // the shared global pool. Both the concurrent and the sequential reference
    // paths use the same setting, so the comparison is apples-to-apples.
    kernels::set_threads(1);

    let samples_a = load_wav(&wav_a);
    let samples_b = load_wav(&wav_b);

    // --- Reference: two fully independent contexts, run sequentially. ---
    let mut ref_a = QwenCtx::load(&model_dir).expect("load ref A");
    let ref_text_a = transcribe::transcribe_audio(&mut ref_a, &samples_a).expect("ref A decode");
    let mut ref_b = QwenCtx::load(&model_dir).expect("load ref B");
    let ref_text_b = transcribe::transcribe_audio(&mut ref_b, &samples_b).expect("ref B decode");

    // --- Under test: ONE shared model, TWO sessions, run concurrently. ---
    let model: std::sync::Arc<QwenModel> = QwenModel::load(&model_dir).expect("load shared model");
    let mut sess_a = model.new_session();
    let mut sess_b = model.new_session();

    let sa = &samples_a;
    let sb = &samples_b;
    let (got_a, got_b) = std::thread::scope(|scope| {
        let ha = scope.spawn(|| transcribe::transcribe_audio(&mut sess_a, sa));
        let hb = scope.spawn(|| transcribe::transcribe_audio(&mut sess_b, sb));
        (ha.join().unwrap(), hb.join().unwrap())
    });
    let got_a = got_a.expect("session A decode");
    let got_b = got_b.expect("session B decode");

    assert_eq!(
        got_a, ref_text_a,
        "concurrent session A must match sequential fresh context"
    );
    assert_eq!(
        got_b, ref_text_b,
        "concurrent session B must match sequential fresh context"
    );
    // Sanity: the two inputs really are different audio.
    assert_ne!(ref_text_a, ref_text_b, "test inputs should differ");

    // The shared model is still referenced by both live sessions.
    assert_eq!(std::sync::Arc::strong_count(sess_a.model()), 3);

    kernels::set_threads(kernels::get_num_cpus());
}
