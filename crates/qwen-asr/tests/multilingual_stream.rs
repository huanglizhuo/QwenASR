//! Task 4 — mixed-language (code-switch) streaming validation.
//!
//! Drives the incremental `stream_push_audio` path over a synthesized
//! zh/en/zh/en… clip (`bench/samples/mixed_zh_en.wav`, built with macOS
//! `say -v Tingting` for Chinese + segments of `bench/samples/audio.wav` for
//! English, separated by 1.8 s silence gaps) and checks per-utterance
//! output-language correctness in two modes:
//!
//! * OFF  — the app's current shipping default (`want_language_detection =
//!          false`, `multilingual = false`): a fixed `language English` preamble
//!          is prefilled, so Chinese speech is *translated* to English and the
//!          whole conversation collapses to one language.
//! * ON   — multilingual mode (`multilingual = true`): at every silence
//!          utterance boundary the language carry is dropped so the next
//!          utterance re-detects its language from fresh audio, preserving the
//!          code-switch (Chinese stays Chinese, English stays English).
//!
//! Run with output:
//!   cargo test --release --test multilingual_stream -- --nocapture
//!
//! Skips cleanly when the model or the fixture wav is absent.

use qwen_asr::context::QwenCtx;
use qwen_asr::kernels;
use qwen_asr::transcribe::{stream_push_audio, StreamState};
use std::sync::Mutex;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

// Resolve fixtures relative to the workspace root (CARGO_MANIFEST_DIR points at
// crates/qwen-asr), independent of the test's CWD.
fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

// Total sample count of the committed fixture (16 kHz mono). If a regenerated
// fixture differs, the test skips rather than misreporting against stale
// offsets.
const EXPECT_TOTAL: usize = 595_011;

// Per-utterance checkpoint offsets (end of the silence gap following each
// utterance), derived from the concatenation part sample counts:
//   zh1 57297 | sil 28800 | en1 80000 | sil | zh2 38802 | sil | en2 80000 |
//   sil | zh3 43530 | sil | en3 80000 | sil | zh4 42582
// The committed text delta between consecutive checkpoints is one utterance.
const CHECKPOINTS: &[usize] = &[86_097, 194_897, 262_499, 371_299, 443_629, 552_429, 595_011];
// Expected spoken language of each utterance, in order.
const EXPECT_LANG: &[Lang] = &[
    Lang::Zh,
    Lang::En,
    Lang::Zh,
    Lang::En,
    Lang::Zh,
    Lang::En,
    Lang::Zh,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lang {
    Zh,
    En,
    Mixed,
    Empty,
}

/// Classify a transcript fragment by dominant script: Han characters → Chinese,
/// ASCII latin letters → English.
fn classify(text: &str) -> Lang {
    let mut han = 0usize;
    let mut latin = 0usize;
    for c in text.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&c) {
            han += 1;
        } else if c.is_ascii_alphabetic() {
            latin += 1;
        }
    }
    match (han, latin) {
        (0, 0) => Lang::Empty,
        (h, l) if h > 0 && l == 0 => Lang::Zh,
        (h, l) if l > 0 && h == 0 => Lang::En,
        (h, l) if h >= l * 3 => Lang::Zh,
        (h, l) if l >= h * 3 => Lang::En,
        _ => Lang::Mixed,
    }
}

fn load_fixture() -> Option<Vec<f32>> {
    let bytes = std::fs::read(workspace_path("bench/samples/mixed_zh_en.wav")).ok()?;
    qwen_asr::audio::parse_wav_buffer(&bytes)
}

/// Feed `samples` through `stream_push_audio` in 0.5 s mic-sized pushes (the
/// device cadence). Returns the committed stable transcript captured at each
/// checkpoint offset.
fn run_stream(ctx: &mut QwenCtx, samples: &[f32]) -> Vec<String> {
    ctx.stream_chunk_sec = 1.5;
    ctx.stream_unfixed_chunks = 2;
    ctx.past_text_conditioning = true;

    let mut state = StreamState::new();
    let push = (0.5 * 16000.0) as usize;
    let mut cursor = 0usize;
    let mut ci = 0usize;
    let mut snaps = vec![String::new(); CHECKPOINTS.len()];

    while cursor < samples.len() {
        let end = (cursor + push).min(samples.len());
        let finalize = end >= samples.len();
        let _ = stream_push_audio(ctx, &samples[..end], &mut state, finalize);
        cursor = end;
        // Snapshot committed text whenever we cross a checkpoint boundary.
        while ci < CHECKPOINTS.len() && cursor >= CHECKPOINTS[ci] {
            snaps[ci] = state.text();
            ci += 1;
        }
    }
    // Ensure a final flush + any trailing checkpoints filled.
    let _ = stream_push_audio(ctx, samples, &mut state, true);
    while ci < CHECKPOINTS.len() {
        snaps[ci] = state.text();
        ci += 1;
    }
    snaps
}

/// Convert cumulative committed snapshots into per-utterance deltas.
fn deltas(snaps: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(snaps.len());
    let mut prev_len = 0usize;
    for s in snaps {
        let bytes = s.as_bytes();
        // The committed transcript only grows; take the newly appended tail.
        if bytes.len() >= prev_len {
            out.push(String::from_utf8_lossy(&bytes[prev_len..]).into_owned());
        } else {
            out.push(s.clone());
        }
        prev_len = bytes.len();
    }
    out
}

fn score(deltas: &[String]) -> (usize, Vec<Lang>) {
    let mut correct = 0usize;
    let langs: Vec<Lang> = deltas.iter().map(|d| classify(d)).collect();
    for (got, want) in langs.iter().zip(EXPECT_LANG.iter()) {
        if got == want {
            correct += 1;
        }
    }
    (correct, langs)
}

/// Alignment-robust metric: reduce the full transcript to its ordered sequence
/// of language runs (each maximal run of Han → Zh, of ASCII-latin words → En;
/// consecutive same-language runs merged), independent of streaming commit lag /
/// duplication. A correct code-switch transcription yields the expected
/// alternating [Zh,En,Zh,En,Zh,En,Zh] run sequence.
fn lang_runs(text: &str) -> Vec<Lang> {
    let mut runs: Vec<Lang> = Vec::new();
    for c in text.chars() {
        let l = if ('\u{4e00}'..='\u{9fff}').contains(&c) {
            Lang::Zh
        } else if c.is_ascii_alphabetic() {
            Lang::En
        } else {
            continue; // punctuation / digits / spaces are language-neutral
        };
        if runs.last() != Some(&l) {
            runs.push(l);
        }
    }
    runs
}

/// Longest common subsequence length between a run sequence and the expected
/// language pattern — how many utterances landed in the right language, in order.
fn lcs(a: &[Lang], b: &[Lang]) -> usize {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    dp[a.len()][b.len()]
}

#[test]
fn multilingual_code_switch_language_correctness() {
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
            eprintln!("SKIP: fixture bench/samples/mixed_zh_en.wav not found (build it per docs/research/experiments.md R13-Android)");
            return;
        }
    };
    if (samples.len() as i64 - EXPECT_TOTAL as i64).abs() > 800 {
        eprintln!(
            "SKIP: fixture sample count {} != expected {} (regenerated differently)",
            samples.len(),
            EXPECT_TOTAL
        );
        return;
    }

    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());

    // ---- OFF: the app's current shipping default (fixed English preamble,
    // no per-utterance re-detection) — the real before-state users see today.
    let mut ctx = QwenCtx::load(model_dir).expect("load model");
    ctx.want_language_detection = false;
    ctx.multilingual = false;
    let off_snaps = run_stream(&mut ctx, &samples);
    let off_deltas = deltas(&off_snaps);
    let (off_correct, off_langs) = score(&off_deltas);

    // ---- ON: multilingual per-utterance re-detection (the fix) ----
    let mut ctx = QwenCtx::load(model_dir).expect("load model");
    ctx.set_multilingual(true);
    let on_snaps = run_stream(&mut ctx, &samples);
    let on_deltas = deltas(&on_snaps);
    let (on_correct, on_langs) = score(&on_deltas);

    let n = EXPECT_LANG.len();
    eprintln!("\n=== Task 4: mixed-language per-utterance language correctness ===");
    eprintln!("utt  expect  OFF(default)  ON(multilingual)");
    for i in 0..n {
        eprintln!(
            "{:>3}  {:>6?}  {:>11?}  {:>16?}   OFF='{}' ON='{}'",
            i + 1,
            EXPECT_LANG[i],
            off_langs.get(i).copied().unwrap_or(Lang::Empty),
            on_langs.get(i).copied().unwrap_or(Lang::Empty),
            off_deltas.get(i).map(|s| s.as_str()).unwrap_or("").trim(),
            on_deltas.get(i).map(|s| s.as_str()).unwrap_or("").trim(),
        );
    }
    eprintln!("per-checkpoint correct — OFF: {off_correct}/{n}   ON: {on_correct}/{n}");

    // Alignment-robust metric: language-run sequence of the full transcript vs
    // the expected alternating pattern (immune to streaming commit lag).
    let off_final = off_snaps.last().cloned().unwrap_or_default();
    let on_final = on_snaps.last().cloned().unwrap_or_default();
    let off_runs = lang_runs(&off_final);
    let on_runs = lang_runs(&on_final);
    let expect_runs: Vec<Lang> = EXPECT_LANG.to_vec();
    let off_lcs = lcs(&off_runs, &expect_runs);
    let on_lcs = lcs(&on_runs, &expect_runs);
    let off_en = off_runs.iter().filter(|&&l| l == Lang::En).count();
    let on_en = on_runs.iter().filter(|&&l| l == Lang::En).count();
    eprintln!("run-sequence  expect={expect_runs:?}");
    eprintln!("run-sequence  OFF={off_runs:?}  (English runs={off_en}, LCS={off_lcs}/{n})");
    eprintln!("run-sequence  ON ={on_runs:?}  (English runs={on_en}, LCS={on_lcs}/{n})");
    eprintln!(
        "English utterances rendered in English — OFF: {off_en}/3   ON: {on_en}/3\n"
    );

    // Headline metric: LCS of the transcript's language-run sequence against the
    // expected alternating pattern. The app's current default renders the whole
    // conversation in a single language (Chinese speech translated to English),
    // so its run sequence collapses to ~[En] (LCS≈1). Multilingual mode preserves
    // the code-switch, recovering a materially higher LCS. Robust to streaming
    // commit lag / duplication.
    assert!(
        on_lcs > off_lcs,
        "multilingual ON did not improve code-switch language correctness over the \
         default (run-seq LCS {on_lcs} vs {off_lcs}) — boundary reset ineffective"
    );
}
