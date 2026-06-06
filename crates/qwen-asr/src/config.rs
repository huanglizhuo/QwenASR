//! Model configuration and automatic variant detection.

pub const SAMPLE_RATE: i32 = 16000;
pub const MEL_BINS: usize = 128;
pub const HOP_LENGTH: usize = 160;
pub const WINDOW_SIZE: usize = 400;
pub const VOCAB_SIZE: usize = 151936;

pub const MAX_ENC_LAYERS: usize = 24;
pub const MAX_DEC_LAYERS: usize = 28;

// Special token IDs
pub const TOKEN_IM_START: i32 = 151644;
pub const TOKEN_IM_END: i32 = 151645;
pub const TOKEN_ENDOFTEXT: i32 = 151643;
pub const TOKEN_AUDIO_START: i32 = 151669;
pub const TOKEN_AUDIO_END: i32 = 151670;
pub const TOKEN_AUDIO_PAD: i32 = 151676;
pub const TOKEN_ASR_TEXT: i32 = 151704;
pub const TOKEN_TIMESTAMP: i32 = 151705;

// Conv2D stem constants
pub const CONV_HIDDEN: usize = 480;
pub const CONV_KERNEL: usize = 3;

#[derive(Clone)]
pub struct QwenConfig {
    // Audio encoder
    pub enc_d_model: usize,
    pub enc_layers: usize,
    pub enc_heads: usize,
    pub enc_head_dim: usize,
    pub enc_ffn_dim: usize,
    pub enc_output_dim: usize,
    pub enc_n_window: usize,
    pub enc_n_window_infer: usize,
    pub enc_chunk_size: usize,
    pub enc_conv_proj_dim: usize,

    // LLM decoder
    pub dec_hidden: usize,
    pub dec_layers: usize,
    pub dec_heads: usize,
    pub dec_kv_heads: usize,
    pub dec_head_dim: usize,
    pub dec_intermediate: usize,
    pub vocab_size: usize,
    pub dec_rms_norm_eps: f32,
    pub dec_rope_theta: f32,

    // Forced aligner fields (0 = normal ASR model)
    pub classify_num: usize,
    pub timestamp_segment_time: f32,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            enc_d_model: 0,
            enc_layers: 0,
            enc_heads: 0,
            enc_head_dim: 0,
            enc_ffn_dim: 0,
            enc_output_dim: 0,
            enc_n_window: 50,
            enc_n_window_infer: 800,
            enc_chunk_size: 100,
            enc_conv_proj_dim: CONV_HIDDEN * 16,
            dec_hidden: 0,
            dec_layers: 28,
            dec_heads: 16,
            dec_kv_heads: 8,
            dec_head_dim: 128,
            dec_intermediate: 0,
            vocab_size: VOCAB_SIZE,
            dec_rms_norm_eps: 1e-6,
            dec_rope_theta: 1e6,
            classify_num: 0,
            timestamp_segment_time: 0.0,
        }
    }
}

impl QwenConfig {
    /// Returns the effective lm_head output dimension.
    pub fn lm_head_dim(&self) -> usize {
        if self.classify_num > 0 { self.classify_num } else { self.vocab_size }
    }

    /// Whether this config is for a forced aligner model.
    pub fn is_aligner(&self) -> bool {
        self.classify_num > 0
    }
}

/// Tensor shape info passed from safetensors for model detection.
pub struct DetectInfo<'a> {
    pub has_enc_layer_18: bool,
    /// Shape of thinker.lm_head.weight (if present)
    pub lm_head_shape: Option<&'a [i64]>,
    /// Shape of thinker.model.embed_tokens.weight
    pub embed_tokens_shape: Option<&'a [i64]>,
    /// Shape of thinker.model.layers.0.mlp.gate_proj.weight
    pub gate_proj_shape: Option<&'a [i64]>,
}

impl QwenConfig {
    /// Detect model config from a GGUF file.
    ///
    /// Reads standard llama.cpp KV pairs first. Falls back to tensor-shape detection
    /// (same as safetensors path) for keys that are absent.
    pub fn detect_from_gguf(gguf: &crate::gguf::GgufFile) -> Self {
        let mut cfg = Self::default();

        // Decoder params from KV — try qwen3vl.* (ggml-org) then qwen3.* prefix
        let kv_u32 = |key_vl: &str, key_plain: &str| -> Option<u32> {
            gguf.get_kv_u32(key_vl).or_else(|| gguf.get_kv_u32(key_plain))
        };
        let kv_f32 = |key_vl: &str, key_plain: &str| -> Option<f32> {
            gguf.get_kv_f32(key_vl).or_else(|| gguf.get_kv_f32(key_plain))
        };
        if let Some(v) = kv_u32("qwen3vl.block_count",                    "qwen3.block_count")                    { cfg.dec_layers       = v as usize; }
        if let Some(v) = kv_u32("qwen3vl.embedding_length",               "qwen3.embedding_length")               { cfg.dec_hidden       = v as usize; }
        if let Some(v) = kv_u32("qwen3vl.feed_forward_length",            "qwen3.feed_forward_length")            { cfg.dec_intermediate = v as usize; }
        if let Some(v) = kv_u32("qwen3vl.attention.head_count",           "qwen3.attention.head_count")           { cfg.dec_heads        = v as usize; }
        if let Some(v) = kv_u32("qwen3vl.attention.head_count_kv",        "qwen3.attention.head_count_kv")        { cfg.dec_kv_heads     = v as usize; }
        if let Some(v) = kv_u32("qwen3vl.attention.key_length",           "qwen3.attention.key_length")           { cfg.dec_head_dim     = v as usize; }
        if let Some(v) = kv_f32("qwen3vl.rope.freq_base",                 "qwen3.rope.freq_base")                 { cfg.dec_rope_theta   = v; }
        if let Some(v) = kv_f32("qwen3vl.attention.layer_norm_rms_epsilon","qwen3.attention.layer_norm_rms_epsilon") { cfg.dec_rms_norm_eps = v; }

        // If KV pairs were present, dec_hidden and dec_layers are now set.
        // If not, fall back to tensor-shape detection using whichever names are present:
        if cfg.dec_hidden == 0 || cfg.dec_layers == 0 {
            // Try both ggml-org (output.weight) and HuggingFace-style names.
            let lm_head_i64: Option<Vec<i64>> = gguf.find("thinker.lm_head.weight")
                .or_else(|| gguf.find("output.weight"))
                .map(|t| t.shape.iter().map(|&x| x as i64).collect());
            let embed_tok_i64: Option<Vec<i64>> = gguf.find("thinker.model.embed_tokens.weight")
                .or_else(|| gguf.find("token_embd.weight"))
                .map(|t| t.shape.iter().map(|&x| x as i64).collect());
            let gate_proj_i64: Option<Vec<i64>> = gguf.find("thinker.model.layers.0.mlp.gate_proj.weight")
                .or_else(|| gguf.find("blk.0.ffn_gate.weight"))
                .map(|t| t.shape.iter().map(|&x| x as i64).collect());
            // GGUF shapes are Vec<u64>; DetectInfo expects &[i64] — convert via temp Vecs.
            let info = crate::config::DetectInfo {
                has_enc_layer_18: gguf.has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight")
                    || gguf.has_tensor("a.blk.18.attn_q.weight"),
                lm_head_shape:       lm_head_i64.as_deref(),
                embed_tokens_shape:  embed_tok_i64.as_deref(),
                gate_proj_shape:     gate_proj_i64.as_deref(),
            };
            return Self::detect(&info);
        }

        // Encoder detection — use tensor shapes (no standard KV for audio encoder)
        // Check both HuggingFace-style and mmproj (a.*) names
        let has_enc_layer_18 = gguf.has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight")
            || gguf.has_tensor("a.blk.18.attn_q.weight");
        if has_enc_layer_18 {
            cfg.enc_d_model = 1024;
            cfg.enc_layers = 24;
            cfg.enc_heads = 16;
            cfg.enc_head_dim = 64;
            cfg.enc_ffn_dim = 4096;
        } else {
            cfg.enc_d_model = 896;
            cfg.enc_layers = 18;
            cfg.enc_heads = 14;
            cfg.enc_head_dim = 64;
            cfg.enc_ffn_dim = 3584;
        }
        cfg.enc_output_dim = cfg.dec_hidden;

        // Fill defaults for fields not in KV
        if cfg.dec_intermediate == 0 {
            cfg.dec_intermediate = if cfg.dec_hidden >= 2048 { 6144 } else { 3072 };
        }
        if cfg.dec_heads == 0 { cfg.dec_heads = 16; }
        if cfg.dec_kv_heads == 0 { cfg.dec_kv_heads = 8; }
        if cfg.dec_head_dim == 0 { cfg.dec_head_dim = 128; }

        // Forced aligner detection — try both naming conventions
        let lm_head_t = gguf.find("thinker.lm_head.weight").or_else(|| gguf.find("output.weight"));
        if let Some(t) = lm_head_t {
            if t.shape.len() >= 2 {
                // GGUF shape is innermost-first: [hidden_dim, vocab_or_classify]
                let out_dim = t.shape[t.shape.len() - 1] as usize;
                if out_dim != VOCAB_SIZE {
                    cfg.classify_num = out_dim;
                    cfg.timestamp_segment_time = 80.0;
                }
            }
        }

        cfg.enc_chunk_size = cfg.enc_n_window * 2;
        cfg
    }

    /// Detect model variant from safetensors tensor shapes.
    /// Handles ASR 0.6B, ASR 1.7B, and ForcedAligner 0.6B (which has 1.7B encoder + 0.6B decoder).
    pub fn detect(info: &DetectInfo) -> Self {
        let mut cfg = Self::default();

        // Determine decoder hidden size from embed_tokens shape [vocab_size, hidden_dim]
        let dec_hidden = info.embed_tokens_shape
            .and_then(|s| if s.len() == 2 { Some(s[1] as usize) } else { None })
            .unwrap_or(if info.has_enc_layer_18 { 2048 } else { 1024 });

        // Determine decoder intermediate from gate_proj shape [intermediate, hidden]
        let dec_intermediate = info.gate_proj_shape
            .and_then(|s| if s.len() == 2 { Some(s[0] as usize) } else { None })
            .unwrap_or(if dec_hidden >= 2048 { 6144 } else { 3072 });

        // Encoder architecture: 24 layers = "large" encoder, 18 layers = "small" encoder
        if info.has_enc_layer_18 {
            // Large encoder (used by both 1.7B ASR and aligner 0.6B)
            cfg.enc_d_model = 1024;
            cfg.enc_layers = 24;
            cfg.enc_heads = 16;
            cfg.enc_head_dim = 64;
            cfg.enc_ffn_dim = 4096;
        } else {
            // Small encoder (0.6B ASR)
            cfg.enc_d_model = 896;
            cfg.enc_layers = 18;
            cfg.enc_heads = 14;
            cfg.enc_head_dim = 64;
            cfg.enc_ffn_dim = 3584;
        }

        // enc_output_dim always matches dec_hidden (proj projects encoder output to decoder space)
        cfg.enc_output_dim = dec_hidden;
        cfg.dec_hidden = dec_hidden;
        cfg.dec_intermediate = dec_intermediate;

        // Detect forced aligner: lm_head has shape [classify_num, hidden_dim]
        // where classify_num != vocab_size (typically 5000)
        if let Some(shape) = info.lm_head_shape {
            if shape.len() == 2 && (shape[0] as usize) != VOCAB_SIZE {
                cfg.classify_num = shape[0] as usize;
                cfg.timestamp_segment_time = 80.0; // 80ms per time bin
            }
        }

        cfg.enc_chunk_size = cfg.enc_n_window * 2;
        cfg
    }
}

pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
    "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
];

pub fn normalize_language(language: &str) -> Option<String> {
    let trimmed = language.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut chars = trimmed.chars();
    let first = chars.next()?.to_uppercase().to_string();
    let rest: String = chars.map(|c| c.to_lowercase().next().unwrap_or(c)).collect();
    let normalized = format!("{}{}", first, rest);

    if SUPPORTED_LANGUAGES.contains(&normalized.as_str()) {
        Some(normalized)
    } else {
        None
    }
}
