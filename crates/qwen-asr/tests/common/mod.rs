//! Shared test-fixture resolution for the integration tests.
//!
//! Convention: tests locate the model and sample audio via the environment
//! variables `QWEN_ASR_TEST_MODEL_DIR` / `QWEN_ASR_TEST_SAMPLES_DIR` when set,
//! falling back to the repo-local `qwen3-asr-0.6b/` and `bench/samples/`
//! directories. Tests skip cleanly when a fixture is absent (e.g. in CI,
//! where the model is not downloaded).
//!
//! Not every test binary uses every helper.
#![allow(dead_code)]

/// Resolve a repo-relative path that exists either relative to the current
/// directory or to the workspace root (cargo runs test binaries with cwd =
/// package dir, i.e. `crates/qwen-asr`, so `../../` reaches the repo root).
pub fn resolve(rel: &str) -> Option<String> {
    for prefix in ["", "../../"] {
        let p = format!("{prefix}{rel}");
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    None
}

/// Directory containing the Qwen3-ASR model weights (`model.safetensors`).
pub fn model_dir() -> Option<String> {
    if let Ok(d) = std::env::var("QWEN_ASR_TEST_MODEL_DIR") {
        if std::path::Path::new(&d).join("model.safetensors").exists() {
            return Some(d);
        }
        eprintln!("QWEN_ASR_TEST_MODEL_DIR={d} has no model.safetensors; ignoring");
    }
    resolve("qwen3-asr-0.6b").filter(|d| std::path::Path::new(d).join("model.safetensors").exists())
}

/// Path to a sample file (wav/txt) inside the samples directory.
pub fn sample(name: &str) -> Option<String> {
    if let Ok(d) = std::env::var("QWEN_ASR_TEST_SAMPLES_DIR") {
        let p = format!("{d}/{name}");
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
        eprintln!("QWEN_ASR_TEST_SAMPLES_DIR={d} has no {name}; ignoring");
    }
    resolve(&format!("bench/samples/{name}"))
}
