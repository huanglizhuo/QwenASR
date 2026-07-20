//! Task 3 — `vad_segment_reset` (VAD Live) discrete per-utterance segmentation.
//!
//! Verifies the VAD *detection* facility (100 ms-frame silence-boundary scan) is
//! DECOUPLED from the VAD *action*: with `multilingual = false` and the new
//! `vad_segment_reset = true`, the scan still runs and each detected utterance
//! boundary performs a hard segment reset (text carry + encoder window dropped),
//! WITHOUT engaging any language-re-detection logic.
//!
//! The fixture is the code-switch clip `bench/samples/mixed_zh_en.wav` (utterances
//! separated by ~1.8 s silence gaps — well over the 0.6 s boundary threshold), so
//! several boundaries fire over the run.
//!
//! Assertion strategy (model-gated, no golden text): observe the encoder-window
//! count after every push. A segment reset releases the encoder cache, so the
//! count DROPS. `vad_segment_reset = true` must produce strictly MORE such drops
//! than the default (both flags false) path — whose only resets are the periodic
//! reanchor / degen resets — proving the VAD boundaries are driving extra resets.
//!
//! Run with output:
//!   cargo test --release --test vad_segment_reset -- --nocapture
//!
//! Skips cleanly when the model or the fixture wav is absent.

use qwen_asr::context::QwenCtx;
use qwen_asr::kernels;
use qwen_asr::transcribe::{stream_push_audio, StreamState};
use std::sync::Mutex;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

const EXPECT_TOTAL: usize = 595_011;

fn load_fixture() -> Option<Vec<f32>> {
    let bytes = std::fs::read(workspace_path("bench/samples/mixed_zh_en.wav")).ok()?;
    qwen_asr::audio::parse_wav_buffer(&bytes)
}

/// Feed `samples` through `stream_push_audio` in 0.5 s mic-sized pushes and
/// record the encoder-window count after each push. Returns
/// `(window_counts, final_text, min_stable_after_first_boundary_seen)`.
fn run_stream(ctx: &mut QwenCtx, samples: &[f32]) -> (Vec<usize>, String, bool) {
    ctx.stream_chunk_sec = 1.5;
    ctx.stream_unfixed_chunks = 2;
    ctx.past_text_conditioning = true;

    let mut state = StreamState::new();
    let push = (0.5 * 16000.0) as usize;
    let mut cursor = 0usize;
    let mut counts: Vec<usize> = Vec::new();
    // Detect a mid-stream stable-token reset: stable count grows, then drops to a
    // small value (segment reset) while more audio still follows.
    let mut prev_stable = 0usize;
    let mut saw_stable_reset = false;

    while cursor < samples.len() {
        let end = (cursor + push).min(samples.len());
        let finalize = end >= samples.len();
        let _ = stream_push_audio(ctx, &samples[..end], &mut state, finalize);
        counts.push(state.enc_window_count());
        let stable = state.stable_token_count();
        if prev_stable >= 3 && stable < prev_stable && !finalize {
            saw_stable_reset = true;
        }
        prev_stable = stable;
        cursor = end;
    }
    let _ = stream_push_audio(ctx, samples, &mut state, true);
    (counts, state.text(), saw_stable_reset)
}

/// Count how many times the window-count sequence decreases (each decrease is a
/// distinct encoder-cache release / segment reset event).
fn count_drops(counts: &[usize]) -> usize {
    counts.windows(2).filter(|w| w[1] < w[0]).count()
}

#[test]
fn vad_segment_reset_triggers_extra_boundary_resets() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let model_dir = workspace_path("qwen3-asr-0.6b");
    if !model_dir.join("model.safetensors").exists() {
        eprintln!("SKIP: model not downloaded at {}", model_dir.display());
        return;
    }
    let model_dir = model_dir.to_str().unwrap();
    let samples = match load_fixture() {
        Some(s) => s,
        None => {
            eprintln!("SKIP: fixture bench/samples/mixed_zh_en.wav not found");
            return;
        }
    };
    if (samples.len() as i64 - EXPECT_TOTAL as i64).abs() > 800 {
        eprintln!(
            "SKIP: fixture sample count {} != expected {}",
            samples.len(),
            EXPECT_TOTAL
        );
        return;
    }

    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());

    // ---- OFF: default path (both flags false) — baseline reanchor/degen resets.
    let mut ctx = QwenCtx::load(model_dir).expect("load model");
    ctx.multilingual = false;
    ctx.vad_segment_reset = false;
    let (off_counts, _off_text, off_stable_reset) = run_stream(&mut ctx, &samples);
    let off_drops = count_drops(&off_counts);

    // ---- ON: vad_segment_reset only (multilingual stays OFF — no language
    // re-detection engaged, pure discrete segmentation).
    let mut ctx = QwenCtx::load(model_dir).expect("load model");
    ctx.multilingual = false;
    ctx.set_vad_segment_reset(true);
    assert!(ctx.vad_segment_reset, "setter must enable the flag");
    assert!(!ctx.multilingual, "vad_segment_reset must not touch multilingual");
    let (on_counts, on_text, on_stable_reset) = run_stream(&mut ctx, &samples);
    let on_drops = count_drops(&on_counts);

    eprintln!("\n=== Task 3: vad_segment_reset discrete segmentation ===");
    eprintln!("encoder-cache release events — OFF(default): {off_drops}   ON(vad): {on_drops}");
    eprintln!("mid-stream stable-token reset seen — OFF: {off_stable_reset}   ON: {on_stable_reset}");
    eprintln!("ON final transcript (first 200 chars): {:.200}", on_text.trim());

    // The VAD boundaries (silence gaps) must drive strictly more segment resets
    // than the default path's periodic reanchor/degen resets alone.
    assert!(
        on_drops > off_drops,
        "vad_segment_reset did not produce extra boundary resets \
         (ON drops={on_drops} <= OFF drops={off_drops}) — decoupled scan ineffective"
    );
    // And a stable-token carry reset must be observed mid-stream in ON mode (a
    // hard segment reset, not merely an encoder-cache trim).
    assert!(
        on_stable_reset,
        "vad_segment_reset never dropped the committed stable-token carry mid-stream"
    );
    // ON must still produce non-trivial text (segmentation didn't destroy output).
    assert!(
        on_text.trim().chars().count() > 10,
        "vad_segment_reset produced near-empty transcript: {on_text:?}"
    );
}
