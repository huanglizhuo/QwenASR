use std::sync::Mutex;
use flutter_rust_bridge::frb;
use qwen_asr::context::QwenCtx;
use qwen_asr::{audio, kernels, transcribe};

#[frb(opaque)]
pub struct QwenAsrEngine {
    inner: Mutex<QwenCtx>,
}

impl QwenAsrEngine {
    /// Load model from a directory path. Returns None if loading fails.
    pub fn load(model_dir: String, n_threads: i32, verbosity: i32) -> Option<QwenAsrEngine> {
        kernels::set_verbose(verbosity);
        let threads = if n_threads <= 0 {
            kernels::get_num_cpus()
        } else {
            n_threads as usize
        };
        kernels::set_threads(threads);
        QwenCtx::load(&model_dir).map(|ctx| QwenAsrEngine {
            inner: Mutex::new(ctx),
        })
    }

    /// Transcribe a WAV file at the given path.
    pub fn transcribe_file(&self, wav_path: String) -> Option<String> {
        let mut ctx = self.inner.lock().unwrap();
        transcribe::transcribe(&mut ctx, &wav_path)
    }

    /// Transcribe raw PCM f32 samples (16kHz mono).
    pub fn transcribe_pcm(&self, samples: Vec<f32>) -> Option<String> {
        let mut ctx = self.inner.lock().unwrap();
        transcribe::transcribe_audio(&mut ctx, &samples)
    }

    /// Transcribe from a WAV file buffer (bytes).
    pub fn transcribe_wav_buffer(&self, wav_data: Vec<u8>) -> Option<String> {
        let samples = audio::parse_wav_buffer(&wav_data)?;
        let mut ctx = self.inner.lock().unwrap();
        transcribe::transcribe_audio(&mut ctx, &samples)
    }

    /// Set the segment duration in seconds (0 = no segmentation).
    #[frb(sync)]
    pub fn set_segment_sec(&self, sec: f32) {
        let mut ctx = self.inner.lock().unwrap();
        ctx.segment_sec = sec;
    }

    /// Set the forced language. Returns false if the language is invalid.
    #[frb(sync)]
    pub fn set_language(&self, language: String) -> bool {
        let mut ctx = self.inner.lock().unwrap();
        ctx.set_force_language(&language).is_ok()
    }

    /// Get last transcription performance stats as a formatted string.
    #[frb(sync)]
    pub fn perf_stats(&self) -> String {
        let ctx = self.inner.lock().unwrap();
        format!(
            "audio={:.1}ms encode={:.1}ms decode={:.1}ms total={:.1}ms tokens={}",
            ctx.perf_audio_ms, ctx.perf_encode_ms, ctx.perf_decode_ms,
            ctx.perf_total_ms, ctx.perf_text_tokens
        )
    }
}

#[frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
}
