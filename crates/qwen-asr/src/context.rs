//! Top-level engine state split into an immutable, shareable [`QwenModel`]
//! (loaded weights) and a per-session [`QwenCtx`] (KV cache, scratch, settings).
//!
//! A [`QwenModel`] is `Sync` and lives behind an `Arc`; any number of
//! independent [`QwenCtx`] sessions can be created from one model with
//! [`QwenModel::new_session`] and driven concurrently on separate threads,
//! each with its own mutable decode state. This is the substrate the stage-2
//! lockstep batched-segment decoder consumes.

use crate::config::*;
use crate::decoder::*;
use crate::encoder::EncoderBuffers;
use crate::encoder::*;
use crate::kernels;
use crate::safetensors::MultiSafetensors;
use crate::tokenizer::QwenTokenizer;
use std::sync::Arc;

pub type TokenCallback = Box<dyn Fn(&str) + Send>;

/// Immutable, thread-shareable model: memory-mapped safetensors plus every
/// weight pointer/quantized buffer, the detected [`QwenConfig`], and the model
/// directory. Nothing here is mutated after [`QwenModel::load`], so a single
/// `Arc<QwenModel>` can back many concurrent [`QwenCtx`] sessions.
///
/// The mmap (`_safetensors`) is held here so the raw BF16 pointers inside
/// [`Encoder`]/[`Decoder`] stay valid for the model's entire lifetime.
pub struct QwenModel {
    pub config: QwenConfig,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub _safetensors: MultiSafetensors, // kept alive for mmap'd BF16 pointers
    pub model_dir: String,
}

impl QwenModel {
    /// Load a Qwen3-ASR model from `model_dir` into a shareable `Arc`.
    ///
    /// The directory must contain `model*.safetensors` and `vocab.json`.
    /// Returns `None` if any required file is missing or malformed.
    pub fn load(model_dir: &str) -> Option<Arc<QwenModel>> {
        if kernels::verbose() >= 1 {
            eprintln!("Loading model from {}", model_dir);
        }

        let _pg = kernels::ProfileGuard::new(&kernels::PROF.model_load);
        let load_t0 = std::time::Instant::now();
        let ms = MultiSafetensors::open(model_dir)?;

        // Detect model variant from tensor shapes
        let info = crate::config::DetectInfo {
            has_enc_layer_18: ms
                .has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight"),
            lm_head_shape: ms
                .find("thinker.lm_head.weight")
                .map(|(_, t)| t.shape.as_slice()),
            embed_tokens_shape: ms
                .find("thinker.model.embed_tokens.weight")
                .map(|(_, t)| t.shape.as_slice()),
            gate_proj_shape: ms
                .find("thinker.model.layers.0.mlp.gate_proj.weight")
                .map(|(_, t)| t.shape.as_slice()),
        };
        let cfg = QwenConfig::detect(&info);

        if kernels::verbose() >= 1 {
            let variant = if cfg.dec_hidden >= 2048 {
                "1.7B"
            } else {
                "0.6B"
            };
            let model_type = if cfg.is_aligner() {
                "ForcedAligner"
            } else {
                "ASR"
            };
            eprintln!("Detected: Qwen3-{}-{}", model_type, variant);
            if cfg.is_aligner() {
                eprintln!(
                    "  classify_num={}, timestamp_segment_time={:.0}ms",
                    cfg.classify_num, cfg.timestamp_segment_time
                );
                eprintln!(
                    "  encoder: {}d {}L, decoder: {}d {}L",
                    cfg.enc_d_model, cfg.enc_layers, cfg.dec_hidden, cfg.dec_layers
                );
            }
        }

        // Load encoder
        if kernels::verbose() >= 1 {
            eprintln!("Loading encoder weights...");
        }
        let encoder = {
            let _pg = kernels::ProfileGuard::new(&kernels::PROF.encoder_load);
            Encoder::load(&ms, &cfg)?
        };

        // Load decoder
        if kernels::verbose() >= 1 {
            eprintln!("Loading decoder weights...");
        }
        let decoder = {
            let _pg = kernels::ProfileGuard::new(&kernels::PROF.decoder_load);
            Decoder::load(&ms, &cfg, model_dir)?
        };

        if kernels::verbose() >= 1 {
            eprintln!(
                "Model loaded in {:.0} ms",
                load_t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        Some(Arc::new(QwenModel {
            config: cfg,
            encoder,
            decoder,
            _safetensors: ms,
            model_dir: model_dir.to_string(),
        }))
    }

    /// Create a fresh, independent decode session that shares this model's
    /// immutable weights. Multiple sessions can run concurrently on different
    /// threads (`QwenModel` is `Sync`) — each owns its own KV cache, RoPE
    /// cache, decoder/encoder scratch and settings. This is the per-session
    /// factory the stage-2 lockstep scheduler builds on.
    pub fn new_session(self: &Arc<Self>) -> QwenCtx {
        QwenCtx::from_model(self.clone())
    }
}

/// Top-level ASR engine state owning model weights, KV cache, and scratch buffers.
///
/// Create with [`QwenCtx::load`], then pass to functions in the [`crate::transcribe`] module.
///
/// # Configurable fields
///
/// | Field | Default | Description |
/// |-------|---------|-------------|
/// | `segment_sec` | 0.0 | Segment duration for long audio (0 = no splitting) |
/// | `skip_silence` | false | Drop silent spans before transcription |
/// | `token_cb` | None | Streaming callback invoked for each decoded token |
/// | `prompt` | None | Optional text prompt (set via [`QwenCtx::set_prompt`]) |
/// | `force_language` | None | Force a language (set via [`QwenCtx::set_force_language`]) |
pub struct QwenCtx {
    /// Shared immutable model (weights, mmap, canonical config). Cloned cheaply
    /// (`Arc`) to spawn sibling sessions; see [`QwenModel::new_session`].
    pub model: Arc<QwenModel>,

    /// Per-session working copy of the model config. Starts as a clone of
    /// `model.config` so run-time knobs (e.g. `enc_n_window_infer`, set by the
    /// CLI) stay local to this session and never mutate the shared model.
    pub config: QwenConfig,

    // KV cache
    pub kv_cache: KvCache,

    // Decoder buffers
    pub dec_bufs: DecoderBuffers,

    // Encoder scratch buffers (reusable across calls)
    pub enc_bufs: EncoderBuffers,

    // RoPE cache
    pub rope_cache: RopeCache,

    /// Lazily-grown buffer pool for the R13-B streaming multi-token verifier.
    /// One [`DecoderBuffers`] set per candidate position in a verify window;
    /// reused across chunks/steps so steady-state verification never allocates.
    /// aarch64-only (the verifier core is aarch64-only).
    #[cfg(target_arch = "aarch64")]
    pub verify_pool: VerifyBufferPool,

    // Token streaming callback
    pub token_cb: Option<TokenCallback>,

    // Segmentation settings
    pub segment_sec: f32,
    pub search_sec: f32,

    // Streaming settings
    pub stream_chunk_sec: f32,
    pub stream_rollback: i32,
    pub stream_unfixed_chunks: i32,
    pub stream_max_new_tokens: i32,
    pub past_text_conditioning: bool,
    pub skip_silence: bool,

    // Optional prompt/language
    pub prompt: Option<String>,
    pub force_language: Option<String>,
    pub detected_language: Option<String>,
    /// When `true` and no language is forced, skip the fixed prompt preamble so
    /// the model generates its own `language X` header for auto-detection. The
    /// default (`false`) keeps the byte-identical plain-text decode path.
    pub want_language_detection: bool,
    pub prompt_tokens: Option<Vec<i32>>,
    pub force_prompt_tokens: Option<Vec<i32>>,
    pub prompt_tokens_ready: bool,

    // Perf stats
    pub perf_total_ms: f64,
    pub perf_text_tokens: i32,
    pub perf_audio_ms: f64,
    /// Mel spectrogram + encoder forward pass time combined.
    pub perf_encode_ms: f64,
    pub perf_decode_ms: f64,
}

impl QwenCtx {
    /// Load a Qwen3-ASR model from `model_dir` into a single default session.
    ///
    /// The directory must contain `model*.safetensors` and `vocab.json`.
    /// Returns `None` if any required file is missing or malformed. Equivalent
    /// to `QwenModel::load(dir).map(QwenModel::new_session)`.
    ///
    /// ```rust,no_run
    /// use qwen_asr::context::QwenCtx;
    /// let ctx = QwenCtx::load("qwen3-asr-0.6b").expect("failed to load");
    /// ```
    pub fn load(model_dir: &str) -> Option<Self> {
        QwenModel::load(model_dir).map(QwenCtx::from_model)
    }

    /// The shared immutable model backing this session. Clone the returned
    /// `Arc` to spawn additional concurrent sessions via
    /// [`QwenModel::new_session`].
    pub fn model(&self) -> &Arc<QwenModel> {
        &self.model
    }

    /// Build a fresh session bound to an already-loaded shared [`QwenModel`].
    /// The session gets its own KV cache, scratch buffers and default settings;
    /// the model's weights are shared by `Arc` with no copy.
    pub fn from_model(model: Arc<QwenModel>) -> Self {
        let cfg = model.config.clone();
        let kv_cache = KvCache::new(cfg.dec_layers, 2048, cfg.dec_kv_heads, cfg.dec_head_dim);
        let dec_bufs = DecoderBuffers::new(&cfg);

        QwenCtx {
            model,
            config: cfg,
            kv_cache,
            dec_bufs,
            enc_bufs: EncoderBuffers::new(),
            rope_cache: RopeCache::new(),
            #[cfg(target_arch = "aarch64")]
            verify_pool: VerifyBufferPool::new(),
            token_cb: None,
            segment_sec: 0.0,
            search_sec: 3.0,
            stream_chunk_sec: 8.0,
            stream_rollback: 5,
            stream_unfixed_chunks: 99,
            stream_max_new_tokens: 32,
            past_text_conditioning: false,
            skip_silence: false,
            prompt: None,
            force_language: None,
            detected_language: None,
            want_language_detection: false,
            prompt_tokens: None,
            force_prompt_tokens: None,
            prompt_tokens_ready: false,
            perf_total_ms: 0.0,
            perf_text_tokens: 0,
            perf_audio_ms: 0.0,
            perf_encode_ms: 0.0,
            perf_decode_ms: 0.0,
        }
    }

    /// Set an optional text prompt to guide transcription. Pass an empty string to clear.
    #[allow(clippy::result_unit_err)]
    pub fn set_prompt(&mut self, prompt: &str) -> Result<(), ()> {
        if prompt.is_empty() {
            self.prompt = None;
        } else {
            self.prompt = Some(prompt.to_string());
        }
        self.prompt_tokens_ready = false;
        Ok(())
    }

    /// Force a specific language (e.g. `"English"`, `"Chinese"`). Pass an empty
    /// string for auto-detection. Returns `Err(())` if the language is not recognized.
    #[allow(clippy::result_unit_err)]
    pub fn set_force_language(&mut self, language: &str) -> Result<(), ()> {
        if language.is_empty() {
            self.force_language = None;
            self.prompt_tokens_ready = false;
            return Ok(());
        }

        match normalize_language(language) {
            Some(normalized) => {
                self.force_language = Some(normalized);
                self.prompt_tokens_ready = false;
                Ok(())
            }
            None => Err(()),
        }
    }

    pub fn prepare_prompt_tokens(&mut self, tokenizer: &QwenTokenizer) -> bool {
        if self.prompt_tokens_ready {
            return true;
        }

        self.prompt_tokens = None;
        self.force_prompt_tokens = None;

        if let Some(ref prompt) = self.prompt {
            match tokenizer.encode(prompt) {
                Some(tokens) => self.prompt_tokens = Some(tokens),
                None => {
                    eprintln!("qwen: failed to encode --prompt text");
                    return false;
                }
            }
        }

        if let Some(ref lang) = self.force_language {
            let force_text = format!("language {}", lang);
            match tokenizer.encode(&force_text) {
                Some(mut lang_tokens) => {
                    lang_tokens.push(TOKEN_ASR_TEXT);
                    self.force_prompt_tokens = Some(lang_tokens);
                }
                None => {
                    eprintln!("qwen: failed to encode --language text");
                    return false;
                }
            }
        } else if !self.want_language_detection {
            // Default path: prefill the fixed `[<language>, <English>, <asr_text>]`
            // preamble. Removing this regresses WER (0.0708 -> 0.0729), so it is
            // only skipped when language auto-detection is explicitly requested.
            self.force_prompt_tokens = Some(vec![11528, 6364, TOKEN_ASR_TEXT]);
        }

        self.prompt_tokens_ready = true;
        true
    }

    pub fn reset_perf(&mut self) {
        self.perf_total_ms = 0.0;
        self.perf_text_tokens = 0;
        self.perf_audio_ms = 0.0;
        self.perf_encode_ms = 0.0;
        self.perf_decode_ms = 0.0;
    }
}
