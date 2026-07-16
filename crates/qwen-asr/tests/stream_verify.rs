//! R13-B integration test: the streaming overlap-draft multi-token verifier
//! must be byte-identical to the pure single-token decode path.
//!
//! Two streaming entry points are exercised:
//!   * [`transcribe_stream`] — the `--stream` CLI path. With the default
//!     `unfixed_chunks` it decodes only the final chunk, so the verifier is
//!     inert (no drafts); the test still asserts the kill switch changes
//!     nothing (guards the `ndraft == 0` branch and the single-token fallback).
//!   * [`stream_push_audio`] — the incremental/live path. It decodes every
//!     chunk and regenerates the running transcript, so the previous chunk's
//!     tail is a high-fidelity draft stream and the multi-token verifier is
//!     genuinely active. `QWEN_ASR_VERIFY=1` vs `QWEN_ASR_VERIFY=0` must
//!     produce byte-identical transcripts.
//!
//! Correctness is by construction (a greedy verifier only ever commits the
//! sequential greedy argmax), so any transcript difference is a bug.

use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe::{self, StreamState};

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
            eprintln!("Skipping stream-verify test: model not downloaded");
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

/// Run `transcribe_stream` end-to-end with the verifier forced on/off.
fn run_transcribe_stream(samples: &[f32], verify: bool) -> String {
    std::env::set_var("QWEN_ASR_VERIFY", if verify { "1" } else { "0" });
    let mut ctx = load_ctx().expect("ctx");
    ctx.past_text_conditioning = true;
    ctx.stream_chunk_sec = 3.0;
    // Force per-chunk rollback decode (default unfixed_chunks=99 decodes only the
    // final chunk, which leaves the verifier inert); this drives the multi-chunk
    // overlap-draft path in transcribe_stream itself.
    ctx.stream_unfixed_chunks = 1;
    // A no-op token callback keeps transcribe_stream on the streaming path
    // (it falls back to offline decode when no callback is set).
    ctx.token_cb = Some(Box::new(|_s: &str| {}));
    transcribe::transcribe_stream(&mut ctx, samples).unwrap_or_default()
}

/// Drive `stream_push_audio` incrementally (mimicking live capture): feed the
/// audio in 2 s pushes against a growing cumulative buffer, finalize at the
/// end, and accumulate the returned delta text.
fn run_push_stream(samples: &[f32], verify: bool) -> String {
    std::env::set_var("QWEN_ASR_VERIFY", if verify { "1" } else { "0" });
    let mut ctx = load_ctx().expect("ctx");
    ctx.past_text_conditioning = true;
    ctx.stream_chunk_sec = 3.0;
    ctx.token_cb = None;

    let mut state = StreamState::new();
    let mut cumulative: Vec<f32> = Vec::new();
    let step = 2 * 16000usize; // 2 s pushes
    let mut out = String::new();
    let mut pos = 0usize;
    while pos < samples.len() {
        let end = (pos + step).min(samples.len());
        cumulative.extend_from_slice(&samples[pos..end]);
        pos = end;
        let finalize = pos >= samples.len();
        if let Some(delta) = transcribe::stream_push_audio(&mut ctx, &cumulative, &mut state, finalize) {
            out.push_str(&delta);
        }
    }
    out
}

#[test]
fn transcribe_stream_verify_on_off_identical() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let samples = match (load_ctx(), load_audio()) {
        (Some(_), Some(s)) => s,
        _ => return, // skip (model or audio missing)
    };

    let on = run_transcribe_stream(&samples, true);
    let off = run_transcribe_stream(&samples, false);
    std::env::remove_var("QWEN_ASR_VERIFY");

    assert_eq!(
        on, off,
        "transcribe_stream: QWEN_ASR_VERIFY on/off transcripts differ\nON:  {on:?}\nOFF: {off:?}"
    );
    eprintln!("transcribe_stream verify on/off identical ({} bytes)", on.len());
}

#[test]
fn stream_push_audio_verify_on_off_identical() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let samples = match (load_ctx(), load_audio()) {
        (Some(_), Some(s)) => s,
        _ => return, // skip (model or audio missing)
    };

    let on = run_push_stream(&samples, true);
    let off = run_push_stream(&samples, false);
    std::env::remove_var("QWEN_ASR_VERIFY");

    assert_eq!(
        on, off,
        "stream_push_audio: QWEN_ASR_VERIFY on/off transcripts differ\nON:  {on:?}\nOFF: {off:?}"
    );
    assert!(!on.is_empty(), "stream_push_audio produced empty transcript");
    eprintln!(
        "stream_push_audio verify on/off identical ({} bytes)",
        on.len()
    );
}
