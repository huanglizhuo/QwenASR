//! R15-5: `transcribe_batch_lockstep` (multi-recording lockstep batch) must
//! reproduce, per recording, a single-threaded sibling session's
//! `transcribe_audio` — the CLI multi-file worker path it replaces. Own test
//! binary: the kernel pool is a process-global singleton whose spawned workers
//! only grow, so this must not share a process with tests that shrink it.

#![cfg(target_arch = "aarch64")]

use qwen_asr::context::QwenCtx;
use qwen_asr::kernels;
use qwen_asr::{audio, transcribe};

mod common;

#[test]
fn batch_lockstep_matches_single_threaded_per_file() {
    let model_dir = match common::model_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skipping batch lockstep test: model not downloaded");
            return;
        }
    };
    let wav = match common::sample("audio.wav") {
        Some(w) => w,
        None => {
            eprintln!("Skipping batch lockstep test: bench audio not found");
            return;
        }
    };

    kernels::set_verbose(0);
    let samples = audio::load_wav(&wav).expect("failed to load wav");
    // Three recordings of different lengths (5 s, 9 s, 7 s) cut from the clip.
    let sr = 16_000;
    let inputs: Vec<Vec<f32>> = [(0, 5), (5, 14), (14, 21)]
        .iter()
        .map(|&(a, b)| samples[a * sr..b * sr].to_vec())
        .collect();

    kernels::set_threads(4);
    let model = qwen_asr::context::QwenModel::load(&model_dir).expect("load model");

    // Reference: one single-threaded sibling session per recording, run on a
    // separate thread exactly like the CLI's fallback workers.
    let reference: Vec<Option<String>> = std::thread::scope(|s| {
        inputs
            .iter()
            .map(|x| {
                let m = model.clone();
                s.spawn(move || {
                    kernels::set_thread_override_public(1);
                    let mut session = QwenCtx::from_model(m);
                    transcribe::transcribe_audio(&mut session, x)
                })
                .join()
                .expect("worker")
            })
            .collect()
    });

    let mut session = QwenCtx::from_model(model.clone());
    let got = transcribe::transcribe_batch_lockstep(&mut session, &inputs)
        .expect("lockstep batch path should be eligible on aarch64");
    assert_eq!(
        got, reference,
        "batch lockstep must match per-file single-threaded decode"
    );

    // Ineligible inputs fall back (None): a sub-0.5 s recording.
    let short = vec![samples[..sr].to_vec(), samples[..sr / 4].to_vec()];
    assert!(transcribe::transcribe_batch_lockstep(&mut session, &short).is_none());
}
