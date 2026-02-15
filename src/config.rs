/// Model configuration and variant detection.

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
        }
    }
}

impl QwenConfig {
    /// Detect model variant by probing for encoder layer 18 in safetensors.
    /// Returns configured QwenConfig for 0.6B or 1.7B.
    pub fn detect(has_layer_18: bool) -> Self {
        let mut cfg = Self::default();

        if has_layer_18 {
            // 1.7B model
            cfg.enc_d_model = 1024;
            cfg.enc_layers = 24;
            cfg.enc_heads = 16;
            cfg.enc_head_dim = 64;
            cfg.enc_ffn_dim = 4096;
            cfg.enc_output_dim = 2048;
            cfg.dec_hidden = 2048;
            cfg.dec_intermediate = 6144;
        } else {
            // 0.6B model
            cfg.enc_d_model = 896;
            cfg.enc_layers = 18;
            cfg.enc_heads = 14;
            cfg.enc_head_dim = 64;
            cfg.enc_ffn_dim = 3584;
            cfg.enc_output_dim = 1024;
            cfg.dec_hidden = 1024;
            cfg.dec_intermediate = 3072;
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
