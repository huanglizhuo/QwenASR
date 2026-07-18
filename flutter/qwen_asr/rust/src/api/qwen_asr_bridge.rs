use flutter_rust_bridge::frb;
use qwen_asr::context::QwenCtx;
use qwen_asr::{audio, kernels, transcribe};
use std::sync::Mutex;

/// Engine internals guarded by a single mutex.
///
/// Holds the model context plus a persistent streaming session so that live
/// microphone audio can be pushed incrementally (mirrors the C-FFI
/// `QwenAsrStreamState` in `crates/qwen-asr/src/c_api.rs`).
struct EngineInner {
    ctx: QwenCtx,
    /// Persistent incremental-streaming state (encoder caches, token history).
    stream_state: transcribe::StreamState,
    /// Full accumulated audio buffer — `stream_push_audio` requires the whole
    /// buffer each call and advances an internal cursor over it.
    stream_audio: Vec<f32>,
}

#[frb(opaque)]
pub struct QwenAsrEngine {
    inner: Mutex<EngineInner>,
}

/// Result of a single [`QwenAsrEngine::stream_push`] call.
///
/// `text` is the committed **stable** transcript accumulated so far (same value
/// the push path returned previously). `provisional` is the newest unfixed tail
/// — a lower-confidence hypothesis held back from the stable commit by the
/// rollback / unfixed-chunks window. Render `provisional` distinctly (e.g. grey)
/// and expect it to be revised or promoted to `text` on later pushes. Both are
/// empty when nothing has been produced; `provisional` is empty after finalize.
pub struct StreamPartial {
    pub text: String,
    pub provisional: String,
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
            inner: Mutex::new(EngineInner {
                ctx,
                stream_state: transcribe::StreamState::new(),
                stream_audio: Vec::new(),
            }),
        })
    }

    /// Transcribe a WAV file at the given path.
    pub fn transcribe_file(&self, wav_path: String) -> Option<String> {
        let mut g = self.inner.lock().unwrap();
        transcribe::transcribe(&mut g.ctx, &wav_path)
    }

    /// Transcribe raw PCM f32 samples (16kHz mono).
    pub fn transcribe_pcm(&self, samples: Vec<f32>) -> Option<String> {
        let mut g = self.inner.lock().unwrap();
        transcribe::transcribe_audio(&mut g.ctx, &samples)
    }

    /// Transcribe from a WAV file buffer (bytes).
    pub fn transcribe_wav_buffer(&self, wav_data: Vec<u8>) -> Option<String> {
        let samples = audio::parse_wav_buffer(&wav_data)?;
        let mut g = self.inner.lock().unwrap();
        transcribe::transcribe_audio(&mut g.ctx, &samples)
    }

    /// Set the segment duration in seconds (0 = no segmentation).
    #[frb(sync)]
    pub fn set_segment_sec(&self, sec: f32) {
        let mut g = self.inner.lock().unwrap();
        g.ctx.segment_sec = sec;
    }

    /// Set the forced language. Returns false if the language is invalid.
    #[frb(sync)]
    pub fn set_language(&self, language: String) -> bool {
        let mut g = self.inner.lock().unwrap();
        g.ctx.set_force_language(&language).is_ok()
    }

    /// Get last transcription performance stats as a formatted string.
    #[frb(sync)]
    pub fn perf_stats(&self) -> String {
        let g = self.inner.lock().unwrap();
        format!(
            "audio={:.1}ms encode={:.1}ms decode={:.1}ms total={:.1}ms tokens={}",
            g.ctx.perf_audio_ms,
            g.ctx.perf_encode_ms,
            g.ctx.perf_decode_ms,
            g.ctx.perf_total_ms,
            g.ctx.perf_text_tokens
        )
    }

    // --------------------------------------------------------------------
    // Streaming API (live microphone / real-time incremental transcription)
    // --------------------------------------------------------------------

    /// Reset the streaming session for a new utterance. Clears the accumulated
    /// audio buffer and token history but keeps the loaded tokenizer/model.
    ///
    /// Call once before starting a fresh live-capture session.
    pub fn stream_reset(&self) {
        let mut g = self.inner.lock().unwrap();
        let inner = &mut *g;
        inner.stream_state.reset();
        inner.stream_audio.clear();
    }

    /// Push a new chunk of PCM audio into the live streaming session and return
    /// the current **stable** transcript plus the newest **provisional** tail.
    ///
    /// `samples`: new PCM chunk (f32, 16 kHz, mono, values in -1.0..1.0).
    /// `finalize`: set true on the final push (Stop) to flush remaining audio
    ///             and emit all rollback-buffered tokens.
    ///
    /// Returns a [`StreamPartial`]: `text` is the committed stable transcript
    /// (same as before), `provisional` is the unfixed tail (empty when none, and
    /// always empty after finalize). Runs on a flutter_rust_bridge worker thread,
    /// so the UI stays responsive while a push is in flight.
    pub fn stream_push(&self, samples: Vec<f32>, finalize: bool) -> StreamPartial {
        let mut g = self.inner.lock().unwrap();
        let inner = &mut *g;

        if !samples.is_empty() {
            inner.stream_audio.extend_from_slice(&samples);
        }

        // Disjoint field borrows: ctx (mut), stream_audio (shared),
        // stream_state (mut) are separate fields of EngineInner.
        let _ = transcribe::stream_push_audio(
            &mut inner.ctx,
            &inner.stream_audio,
            &mut inner.stream_state,
            finalize,
        );

        StreamPartial {
            text: inner.stream_state.text(),
            provisional: inner.stream_state.provisional_text(),
        }
    }

    /// Configure the engine's internal streaming chunk size in seconds
    /// (default 8.0). Smaller values emit partial transcripts more frequently
    /// (lower latency) at some throughput cost. This is the primary live-latency
    /// tuning knob and is independent of the mic push-buffer size on the Dart
    /// side.
    #[frb(sync)]
    pub fn set_stream_chunk_sec(&self, sec: f32) {
        if sec > 0.0 {
            let mut g = self.inner.lock().unwrap();
            g.ctx.stream_chunk_sec = sec;
        }
    }

    /// Configure the streaming token rollback window (default 5). Larger values
    /// re-decode more of the tail each chunk for higher stability.
    #[frb(sync)]
    pub fn set_stream_rollback(&self, tokens: i32) {
        if tokens >= 0 {
            let mut g = self.inner.lock().unwrap();
            g.ctx.stream_rollback = tokens;
        }
    }

    /// Configure max new tokens decoded per chunk (default 32).
    #[frb(sync)]
    pub fn set_stream_max_new_tokens(&self, tokens: i32) {
        if tokens > 0 {
            let mut g = self.inner.lock().unwrap();
            g.ctx.stream_max_new_tokens = tokens;
        }
    }

    /// Configure how many leading chunks stay "unfixed" before tokens are
    /// progressively committed to the stable transcript.
    ///
    /// The engine default (99) effectively holds all output until `finalize`,
    /// which for a fast-pushed live stream truncates at `max_new_tokens`. Set a
    /// small value (e.g. 2) for correct progressive incremental streaming.
    #[frb(sync)]
    pub fn set_stream_unfixed_chunks(&self, chunks: i32) {
        if chunks >= 0 {
            let mut g = self.inner.lock().unwrap();
            g.ctx.stream_unfixed_chunks = chunks;
        }
    }

    /// Enable reusing previously decoded text as decoder context across chunks
    /// (prefix rollback). Recommended `true` for live streaming quality; matches
    /// the CLI `--stream` path.
    #[frb(sync)]
    pub fn set_past_text_conditioning(&self, enabled: bool) {
        let mut g = self.inner.lock().unwrap();
        g.ctx.past_text_conditioning = enabled;
    }
}

#[frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
}
