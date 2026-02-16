/// QwenCtx: top-level state owning all resources.

use crate::config::*;
use crate::decoder::*;
use crate::encoder::*;
use crate::encoder::EncoderBuffers;
use crate::kernels;
use crate::safetensors::MultiSafetensors;
use crate::tokenizer::QwenTokenizer;

pub type TokenCallback = Box<dyn Fn(&str) + Send>;

pub struct QwenCtx {
    pub config: QwenConfig,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub _safetensors: MultiSafetensors, // kept alive for mmap'd BF16 pointers
    pub model_dir: String,

    // KV cache
    pub kv_cache: KvCache,

    // Decoder buffers
    pub dec_bufs: DecoderBuffers,

    // Encoder scratch buffers (reusable across calls)
    pub enc_bufs: EncoderBuffers,

    // RoPE cache
    pub rope_cache: RopeCache,

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
    pub prompt_tokens: Option<Vec<i32>>,
    pub force_prompt_tokens: Option<Vec<i32>>,
    pub prompt_tokens_ready: bool,

    // Perf stats
    pub perf_total_ms: f64,
    pub perf_text_tokens: i32,
    pub perf_audio_ms: f64,
    pub perf_encode_ms: f64,
    pub perf_decode_ms: f64,
}

impl QwenCtx {
    pub fn load(model_dir: &str) -> Option<Self> {
        if kernels::verbose() >= 1 {
            eprintln!("Loading model from {}", model_dir);
        }

        let ms = MultiSafetensors::open(model_dir)?;

        // Detect model variant from tensor shapes
        let info = crate::config::DetectInfo {
            has_enc_layer_18: ms.has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight"),
            lm_head_shape: ms.find("thinker.lm_head.weight").map(|(_, t)| t.shape.as_slice()),
            embed_tokens_shape: ms.find("thinker.model.embed_tokens.weight").map(|(_, t)| t.shape.as_slice()),
            gate_proj_shape: ms.find("thinker.model.layers.0.mlp.gate_proj.weight").map(|(_, t)| t.shape.as_slice()),
        };
        let cfg = QwenConfig::detect(&info);

        if kernels::verbose() >= 1 {
            let variant = if cfg.dec_hidden >= 2048 { "1.7B" } else { "0.6B" };
            let model_type = if cfg.is_aligner() { "ForcedAligner" } else { "ASR" };
            eprintln!("Detected: Qwen3-{}-{}", model_type, variant);
            if cfg.is_aligner() {
                eprintln!("  classify_num={}, timestamp_segment_time={:.0}ms",
                          cfg.classify_num, cfg.timestamp_segment_time);
                eprintln!("  encoder: {}d {}L, decoder: {}d {}L",
                          cfg.enc_d_model, cfg.enc_layers, cfg.dec_hidden, cfg.dec_layers);
            }
        }

        // Load encoder
        if kernels::verbose() >= 1 {
            eprintln!("Loading encoder weights...");
        }
        let encoder = Encoder::load(&ms, &cfg)?;

        // Load decoder
        if kernels::verbose() >= 1 {
            eprintln!("Loading decoder weights...");
        }
        let decoder = Decoder::load(&ms, &cfg)?;

        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let kv_cache = KvCache::new(cfg.dec_layers, 2048, kv_dim);
        let dec_bufs = DecoderBuffers::new(&cfg);

        if kernels::verbose() >= 1 {
            eprintln!("Model loaded.");
        }

        Some(QwenCtx {
            config: cfg,
            encoder,
            decoder,
            _safetensors: ms,
            model_dir: model_dir.to_string(),
            kv_cache,
            dec_bufs,
            enc_bufs: EncoderBuffers::new(),
            rope_cache: RopeCache::new(),
            token_cb: None,
            segment_sec: 0.0,
            search_sec: 3.0,
            stream_chunk_sec: 2.0,
            stream_rollback: 5,
            stream_unfixed_chunks: 2,
            stream_max_new_tokens: 32,
            past_text_conditioning: false,
            skip_silence: false,
            prompt: None,
            force_language: None,
            prompt_tokens: None,
            force_prompt_tokens: None,
            prompt_tokens_ready: false,
            perf_total_ms: 0.0,
            perf_text_tokens: 0,
            perf_audio_ms: 0.0,
            perf_encode_ms: 0.0,
            perf_decode_ms: 0.0,
        })
    }

    pub fn set_prompt(&mut self, prompt: &str) -> Result<(), ()> {
        if prompt.is_empty() {
            self.prompt = None;
        } else {
            self.prompt = Some(prompt.to_string());
        }
        self.prompt_tokens_ready = false;
        Ok(())
    }

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
