//! Finite-file streaming must flush the whole final hypothesis through both
//! its return value and token callback, including at a re-anchor boundary.

use qwen_asr::{audio, context::QwenCtx, kernels, transcribe};
use std::sync::{Arc, Mutex};

mod common;

#[test]
fn file_stream_flushes_final_text() {
    let (Some(model), Some(wav)) = (common::model_dir(), common::sample("audio.wav")) else {
        eprintln!("SKIP: file-stream model/audio fixture unavailable");
        return;
    };
    kernels::set_verbose(0);
    kernels::set_threads(12);
    let samples = audio::load_wav(&wav).expect("audio fixture");
    let mut ctx = QwenCtx::load(&model).expect("model");
    ctx.past_text_conditioning = true;
    let emitted = Arc::new(Mutex::new(String::new()));
    let callback_text = Arc::clone(&emitted);
    ctx.token_cb = Some(Box::new(move |text| {
        callback_text.lock().unwrap().push_str(text);
    }));

    // The short fixture already needs more than 32 tokens. The second case
    // ends at chunk index 45, where a reset used to discard the final decode.
    let duration = samples.len() as f32 / 16_000.0;
    let mut expected = None;
    for (chunk_sec, unfixed_chunks, budget) in [
        (8.0, 99, 32),
        (duration / 45.5, 99, 32),
        // Decode before index 45 and continue after it: the old reset dropped
        // the rollback tail and skipped the ending after shrinking context.
        (duration / 46.5, 44, 64),
        // Completed 8s encoder windows leave several 1s steps unchanged;
        // those are not a reason to discard the accumulated transcript.
        (1.0, 1, 32),
        // An unchanged capped hypothesis may commit nothing; EOF must still
        // regenerate the full tail rather than reuse that truncated result.
        (1.0, 1, 1),
    ] {
        ctx.stream_chunk_sec = chunk_sec;
        ctx.stream_unfixed_chunks = unfixed_chunks;
        ctx.stream_max_new_tokens = budget;
        emitted.lock().unwrap().clear();
        let text = transcribe::transcribe_stream(&mut ctx, &samples).expect("transcription");
        assert!(
            text.ends_with("rarely a wise decision."),
            "file stream lost the ending (chunk_sec={chunk_sec}): {text:?}"
        );
        assert!(ctx.perf_text_tokens > 32, "final decode was capped at 32");
        for phrase in ["keep your eyes open", "worth bringing back"] {
            assert!(
                text.contains(phrase),
                "file stream lost middle text: {text:?}"
            );
        }
        assert_eq!(emitted.lock().unwrap().trim(), text);
        if let Some(ref expected) = expected {
            if unfixed_chunks == 99 {
                assert_eq!(&text, expected, "reset lost or duplicated committed text");
            }
        } else {
            expected = Some(text);
        }
    }
}
