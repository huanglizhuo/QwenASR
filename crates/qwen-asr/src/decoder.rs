//! Qwen3 LLM decoder with GQA, KV cache, and generation.

use crate::config::*;
use crate::gguf::{self, GgufFile, GgmlType};
use crate::kernels;
use crate::safetensors::MultiSafetensors;

// ========================================================================
// Weight representation — covers safetensors BF16, runtime INT8, and GGUF quants
// ========================================================================

/// Quantized weight payload for a single projection matrix (decode path).
pub enum QWeight {
    /// mmap'd BF16 pointer (safetensors zero-copy).
    Bf16Ptr(*const u16),
    /// Runtime-quantized INT8 with per-row f32 scales.
    Int8Owned { data: Vec<i8>, scales: Vec<f32> },
    /// GGUF Q8_0: mmap'd blocks [f16 scale, i8×32] (34 bytes/block).
    Q8_0Ptr(*const u8),
    /// GGUF Q4_K: mmap'd super-blocks (144 bytes each, 256 elements).
    Q4KPtr(*const u8),
    /// GGUF Q4_0: mmap'd blocks (18 bytes/block, 32 elements).
    Q4_0Ptr(*const u8),
}

unsafe impl Send for QWeight {}
unsafe impl Sync for QWeight {}

/// Gate+Up weight pair for SwiGLU FFN (decode path).
pub enum GateUpWeights {
    /// Interleaved BF16 owned buffer (safetensors decode path).
    Bf16Fused(Vec<u16>),
    /// Interleaved INT8 + per-row scales (aarch64 INT8 decode path).
    Int8Fused { data: Vec<i8>, scales: Vec<f32> },
    /// Separate GGUF-quantized gate and up (not interleaved).
    Separate { gate: QWeight, up: QWeight },
}

unsafe impl Send for GateUpWeights {}
unsafe impl Sync for GateUpWeights {}

// ========================================================================
// Layer weights
// ========================================================================

pub struct DecLayer {
    // Decode path — dispatched via QWeight / GateUpWeights
    pub wq:      QWeight,
    pub wk:      QWeight,
    pub wv:      QWeight,
    pub wo:      QWeight,
    pub gate_up: GateUpWeights,
    pub down:    QWeight,

    // Prefill path — always F32, dequanted at load time
    pub wq_f32_prefill:      Vec<f32>,
    pub wk_f32_prefill:      Vec<f32>,
    pub wv_f32_prefill:      Vec<f32>,
    pub wo_f32_prefill:      Vec<f32>,
    pub gate_up_f32_prefill: Vec<f32>,  // interleaved gate+up rows
    pub down_f32_prefill:    Vec<f32>,

    // RMSNorms (always F32)
    pub q_norm_weight:  Vec<f32>,
    pub k_norm_weight:  Vec<f32>,
    pub input_norm:     Vec<f32>,
    pub post_attn_norm: Vec<f32>,
}

unsafe impl Send for DecLayer {}
unsafe impl Sync for DecLayer {}

// ========================================================================
// Decoder
// ========================================================================

pub struct Decoder {
    /// Token embedding table.
    pub tok_embeddings: QWeight,
    pub layers: Vec<DecLayer>,
    pub norm: Vec<f32>,
    /// Separate lm_head (forced aligner) or `None` = tied with tok_embeddings.
    pub lm_head: Option<QWeight>,
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

fn load_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let result = ms.get_f32(name);
    if result.is_none() {
        eprintln!("decoder: weight not found: {}", name);
    }
    result
}

fn load_bf16_direct(ms: &MultiSafetensors, name: &str) -> Option<*const u16> {
    let result = ms.get_bf16_direct(name);
    if result.is_none() {
        eprintln!("decoder: weight not found: {}", name);
    }
    result
}

fn load_bf16_as_f32(
    ms: &MultiSafetensors,
    name: &str,
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    let ptr = load_bf16_direct(ms, name)?;
    let n = rows * cols;
    let mut out = vec![0.0f32; n];
    unsafe {
        let src = std::slice::from_raw_parts(ptr, n);
        kernels::bf16_to_f32_buf(&mut out, src);
    }
    Some(out)
}

impl Decoder {
    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig) -> Option<Self> {
        let tok_emb_ptr = load_bf16_direct(ms, "thinker.model.embed_tokens.weight")?;

        let mut layers = Vec::new();
        for i in 0..cfg.dec_layers {
            let lp = format!("thinker.model.layers.{}", i);

            let wq_ptr = load_bf16_direct(ms, &format!("{}.self_attn.q_proj.weight", lp))?;
            let wk_ptr = load_bf16_direct(ms, &format!("{}.self_attn.k_proj.weight", lp))?;
            let wv_ptr = load_bf16_direct(ms, &format!("{}.self_attn.v_proj.weight", lp))?;
            let wo_ptr = load_bf16_direct(ms, &format!("{}.self_attn.o_proj.weight", lp))?;
            let q_dim = cfg.dec_heads * cfg.dec_head_dim;
            let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
            let hidden = cfg.dec_hidden;
            let inter = cfg.dec_intermediate;

            let wq_f32 = load_bf16_as_f32(ms, &format!("{}.self_attn.q_proj.weight", lp), q_dim, hidden)?;
            let wk_f32 = load_bf16_as_f32(ms, &format!("{}.self_attn.k_proj.weight", lp), kv_dim, hidden)?;
            let wv_f32 = load_bf16_as_f32(ms, &format!("{}.self_attn.v_proj.weight", lp), kv_dim, hidden)?;
            let wo_f32 = load_bf16_as_f32(ms, &format!("{}.self_attn.o_proj.weight", lp), hidden, q_dim)?;

            let q_norm = load_f32(ms, &format!("{}.self_attn.q_norm.weight", lp))?;
            let k_norm = load_f32(ms, &format!("{}.self_attn.k_norm.weight", lp))?;
            let input_norm = load_f32(ms, &format!("{}.input_layernorm.weight", lp))?;
            let post_attn_norm = load_f32(ms, &format!("{}.post_attention_layernorm.weight", lp))?;

            let gate_ptr = load_bf16_direct(ms, &format!("{}.mlp.gate_proj.weight", lp))?;
            let up_ptr = load_bf16_direct(ms, &format!("{}.mlp.up_proj.weight", lp))?;
            let down_ptr = load_bf16_direct(ms, &format!("{}.mlp.down_proj.weight", lp))?;

            // Fuse gate+up rows: [gate[0], up[0], gate[1], up[1], ...]
            let mut gate_up_fused = vec![0u16; 2 * inter * hidden];
            unsafe {
                let gate_sl = std::slice::from_raw_parts(gate_ptr, inter * hidden);
                let up_sl   = std::slice::from_raw_parts(up_ptr,   inter * hidden);
                for r in 0..inter {
                    gate_up_fused[2*r*hidden..(2*r+1)*hidden].copy_from_slice(&gate_sl[r*hidden..(r+1)*hidden]);
                    gate_up_fused[(2*r+1)*hidden..(2*r+2)*hidden].copy_from_slice(&up_sl[r*hidden..(r+1)*hidden]);
                }
            }
            let mut gate_up_f32 = vec![0.0f32; 2 * inter * hidden];
            kernels::bf16_to_f32_buf(&mut gate_up_f32, &gate_up_fused);
            let down_f32 = load_bf16_as_f32(ms, &format!("{}.mlp.down_proj.weight", lp), hidden, inter)?;

            // Build decode-path weights: INT8 on aarch64, BF16 on x86_64
            let (wq_dec, wk_dec, wv_dec, wo_dec, gate_up_dec, down_dec) = {
                #[cfg(target_arch = "aarch64")]
                {
                    let (wq_i, wq_s) = kernels::quantize_bf16_weights_to_int8(wq_ptr, q_dim, hidden);
                    let (wk_i, wk_s) = kernels::quantize_bf16_weights_to_int8(wk_ptr, kv_dim, hidden);
                    let (wv_i, wv_s) = kernels::quantize_bf16_weights_to_int8(wv_ptr, kv_dim, hidden);
                    let (wo_i, wo_s) = kernels::quantize_bf16_weights_to_int8(wo_ptr, hidden, q_dim);
                    let (gu_i, gu_s) = kernels::quantize_bf16_weights_to_int8(gate_up_fused.as_ptr(), 2 * inter, hidden);
                    let (dn_i, dn_s) = kernels::quantize_bf16_weights_to_int8(down_ptr, hidden, inter);
                    (
                        QWeight::Int8Owned { data: wq_i, scales: wq_s },
                        QWeight::Int8Owned { data: wk_i, scales: wk_s },
                        QWeight::Int8Owned { data: wv_i, scales: wv_s },
                        QWeight::Int8Owned { data: wo_i, scales: wo_s },
                        GateUpWeights::Int8Fused { data: gu_i, scales: gu_s },
                        QWeight::Int8Owned { data: dn_i, scales: dn_s },
                    )
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    (
                        QWeight::Bf16Ptr(wq_ptr),
                        QWeight::Bf16Ptr(wk_ptr),
                        QWeight::Bf16Ptr(wv_ptr),
                        QWeight::Bf16Ptr(wo_ptr),
                        GateUpWeights::Bf16Fused(gate_up_fused),
                        QWeight::Bf16Ptr(down_ptr),
                    )
                }
            };

            layers.push(DecLayer {
                wq: wq_dec,
                wk: wk_dec,
                wv: wv_dec,
                wo: wo_dec,
                gate_up: gate_up_dec,
                down: down_dec,
                wq_f32_prefill: wq_f32,
                wk_f32_prefill: wk_f32,
                wv_f32_prefill: wv_f32,
                wo_f32_prefill: wo_f32,
                gate_up_f32_prefill: gate_up_f32,
                down_f32_prefill: down_f32,
                q_norm_weight: q_norm,
                k_norm_weight: k_norm,
                input_norm,
                post_attn_norm,
            });
        }

        let norm = load_f32(ms, "thinker.model.norm.weight")?;

        // lm_head: separate for forced aligner, tied for ASR
        let lm_head_ptr = if cfg.classify_num > 0 {
            Some(load_bf16_direct(ms, "thinker.lm_head.weight")?)
        } else {
            ms.get_bf16_direct("thinker.lm_head.weight")
        };

        // Build lm_head decode weight
        let lm_out_dim = cfg.lm_head_dim();
        let lm_in_dim = cfg.dec_hidden;
        let lm_head = lm_head_ptr.map(|ptr| {
            #[cfg(target_arch = "aarch64")]
            {
                let (d, s) = kernels::quantize_bf16_weights_to_int8(ptr, lm_out_dim, lm_in_dim);
                QWeight::Int8Owned { data: d, scales: s }
            }
            #[cfg(not(target_arch = "aarch64"))]
            { QWeight::Bf16Ptr(ptr) }
        });

        // lm_head for ASR: use tok_embeddings (tied), quantize once
        let tok_emb_dec = {
            #[cfg(target_arch = "aarch64")]
            {
                // If no separate lm_head, lm_head uses tok_emb INT8
                // tok_embeddings for lookup stays as Bf16Ptr
                QWeight::Bf16Ptr(tok_emb_ptr)
            }
            #[cfg(not(target_arch = "aarch64"))]
            { QWeight::Bf16Ptr(tok_emb_ptr) }
        };

        // If no separate lm_head, build INT8 lm_head from tok_emb for fast argmax on aarch64
        let lm_head = if lm_head.is_none() {
            #[cfg(target_arch = "aarch64")]
            {
                let (d, s) = kernels::quantize_bf16_weights_to_int8(tok_emb_ptr, lm_out_dim, lm_in_dim);
                Some(QWeight::Int8Owned { data: d, scales: s })
            }
            #[cfg(not(target_arch = "aarch64"))]
            { None }
        } else {
            lm_head
        };

        Some(Decoder {
            tok_embeddings: tok_emb_dec,
            layers,
            norm,
            lm_head,
        })
    }

    /// Load decoder weights from a GGUF file.
    pub fn load_from_gguf(gguf: &GgufFile, cfg: &QwenConfig) -> Option<Self> {
        let load_qw = |name: &str| -> Option<QWeight> {
            let t = gguf.find(name).or_else(|| {
                eprintln!("decoder(gguf): weight not found: {}", name);
                None
            })?;
            let ptr = gguf.tensor_data(t);
            Some(match t.ggml_type {
                GgmlType::BF16 => QWeight::Bf16Ptr(ptr as *const u16),
                GgmlType::Q8_0 => QWeight::Q8_0Ptr(ptr),
                GgmlType::Q4_K => QWeight::Q4KPtr(ptr),
                GgmlType::Q4_0 => QWeight::Q4_0Ptr(ptr),
                _ => {
                    // Fall back: dequant to F32 and convert to BF16 owned
                    let f32_data = gguf.get_f32(t)?;
                    let bf16: Vec<u16> = f32_data.iter().map(|&f| {
                        let bits = f.to_bits();
                        ((bits >> 16) + ((bits >> 15) & 1)) as u16
                    }).collect();
                    // Keep the Vec alive via a leak for now (GGUF path is expected to use
                    // Q8_0/Q4_K; this fallback is for unexpected float types)
                    let raw = bf16.leak().as_ptr();
                    QWeight::Bf16Ptr(raw)
                }
            })
        };

        let load_f32_gguf = |name: &str| -> Option<Vec<f32>> {
            let t = gguf.find(name).or_else(|| {
                eprintln!("decoder(gguf): weight not found: {}", name);
                None
            })?;
            gguf.get_f32(t)
        };

        let tok_emb_t = gguf.find("thinker.model.embed_tokens.weight").or_else(|| {
            eprintln!("decoder(gguf): embed_tokens not found");
            None
        })?;
        let tok_emb_ptr = gguf.tensor_data(tok_emb_t);
        let tok_embeddings = match tok_emb_t.ggml_type {
            GgmlType::BF16 => QWeight::Bf16Ptr(tok_emb_ptr as *const u16),
            GgmlType::Q8_0 => QWeight::Q8_0Ptr(tok_emb_ptr),
            GgmlType::Q4_K => QWeight::Q4KPtr(tok_emb_ptr),
            GgmlType::Q4_0 => QWeight::Q4_0Ptr(tok_emb_ptr),
            _ => QWeight::Bf16Ptr(tok_emb_ptr as *const u16),
        };

        let mut layers = Vec::new();
        for i in 0..cfg.dec_layers {
            let lp = format!("thinker.model.layers.{}", i);
            let hidden = cfg.dec_hidden;
            let inter = cfg.dec_intermediate;

            let wq = load_qw(&format!("{}.self_attn.q_proj.weight", lp))?;
            let wk = load_qw(&format!("{}.self_attn.k_proj.weight", lp))?;
            let wv = load_qw(&format!("{}.self_attn.v_proj.weight", lp))?;
            let wo = load_qw(&format!("{}.self_attn.o_proj.weight", lp))?;

            // Prefill path: always dequant to F32
            let wq_f32 = load_f32_gguf(&format!("{}.self_attn.q_proj.weight", lp))?;
            let wk_f32 = load_f32_gguf(&format!("{}.self_attn.k_proj.weight", lp))?;
            let wv_f32 = load_f32_gguf(&format!("{}.self_attn.v_proj.weight", lp))?;
            let wo_f32 = load_f32_gguf(&format!("{}.self_attn.o_proj.weight", lp))?;

            let q_norm = load_f32_gguf(&format!("{}.self_attn.q_norm.weight", lp))?;
            let k_norm = load_f32_gguf(&format!("{}.self_attn.k_norm.weight", lp))?;
            let input_norm = load_f32_gguf(&format!("{}.input_layernorm.weight", lp))?;
            let post_attn_norm = load_f32_gguf(&format!("{}.post_attention_layernorm.weight", lp))?;

            let gate_qw = load_qw(&format!("{}.mlp.gate_proj.weight", lp))?;
            let up_qw   = load_qw(&format!("{}.mlp.up_proj.weight", lp))?;
            let down_qw = load_qw(&format!("{}.mlp.down_proj.weight", lp))?;

            // Prefill gate+up: dequant separately then interleave rows
            let gate_f32 = load_f32_gguf(&format!("{}.mlp.gate_proj.weight", lp))?;
            let up_f32   = load_f32_gguf(&format!("{}.mlp.up_proj.weight", lp))?;
            let mut gate_up_f32 = vec![0.0f32; 2 * inter * hidden];
            for r in 0..inter {
                gate_up_f32[2*r*hidden..(2*r+1)*hidden].copy_from_slice(&gate_f32[r*hidden..(r+1)*hidden]);
                gate_up_f32[(2*r+1)*hidden..(2*r+2)*hidden].copy_from_slice(&up_f32[r*hidden..(r+1)*hidden]);
            }
            let down_f32 = load_f32_gguf(&format!("{}.mlp.down_proj.weight", lp))?;

            layers.push(DecLayer {
                wq,
                wk,
                wv,
                wo,
                gate_up: GateUpWeights::Separate { gate: gate_qw, up: up_qw },
                down: down_qw,
                wq_f32_prefill: wq_f32,
                wk_f32_prefill: wk_f32,
                wv_f32_prefill: wv_f32,
                wo_f32_prefill: wo_f32,
                gate_up_f32_prefill: gate_up_f32,
                down_f32_prefill: down_f32,
                q_norm_weight: q_norm,
                k_norm_weight: k_norm,
                input_norm,
                post_attn_norm,
            });
        }

        let norm = load_f32_gguf("thinker.model.norm.weight")?;

        let lm_head = if let Some(t) = gguf.find("thinker.lm_head.weight") {
            let ptr = gguf.tensor_data(t);
            Some(match t.ggml_type {
                GgmlType::BF16 => QWeight::Bf16Ptr(ptr as *const u16),
                GgmlType::Q8_0 => QWeight::Q8_0Ptr(ptr),
                GgmlType::Q4_K => QWeight::Q4KPtr(ptr),
                GgmlType::Q4_0 => QWeight::Q4_0Ptr(ptr),
                _ => QWeight::Bf16Ptr(ptr as *const u16),
            })
        } else {
            None
        };

        Some(Decoder { tok_embeddings, layers, norm, lm_head })
    }
}

// ========================================================================
// KV Cache
// ========================================================================

pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
    pub max_seq: usize,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    /// Layout: `[layer][head][pos][head_dim]` — head-contiguous for cache-friendly attention.
    pub fn new(n_layers: usize, max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let total = n_layers * n_kv_heads * max_seq * head_dim;
        KvCache {
            k: vec![0.0f32; total],
            v: vec![0.0f32; total],
            len: 0,
            max_seq,
            n_layers,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn grow(&mut self, required: usize) {
        if required <= self.max_seq {
            return;
        }

        let mut new_max = self.max_seq;
        while new_max < required {
            new_max *= 2;
        }

        let old_head_stride = self.max_seq * self.head_dim;
        let new_head_stride = new_max * self.head_dim;
        let total = self.n_layers * self.n_kv_heads * new_head_stride;

        let mut new_k = vec![0.0f32; total];
        let mut new_v = vec![0.0f32; total];

        let copy_len = self.len * self.head_dim;
        for l in 0..self.n_layers {
            for h in 0..self.n_kv_heads {
                let old_off = (l * self.n_kv_heads + h) * old_head_stride;
                let new_off = (l * self.n_kv_heads + h) * new_head_stride;
                new_k[new_off..new_off + copy_len]
                    .copy_from_slice(&self.k[old_off..old_off + copy_len]);
                new_v[new_off..new_off + copy_len]
                    .copy_from_slice(&self.v[old_off..old_off + copy_len]);
            }
        }

        self.k = new_k;
        self.v = new_v;
        self.max_seq = new_max;
    }

    /// Write K for all heads at a given position (from interleaved kv_dim buffer).
    pub fn k_write_pos(&mut self, layer: usize, pos: usize, src: &[f32]) {
        let head_stride = self.max_seq * self.head_dim;
        for h in 0..self.n_kv_heads {
            let dst_off = (layer * self.n_kv_heads + h) * head_stride + pos * self.head_dim;
            let src_off = h * self.head_dim;
            self.k[dst_off..dst_off + self.head_dim]
                .copy_from_slice(&src[src_off..src_off + self.head_dim]);
        }
    }

    /// Write V for all heads at a given position (from interleaved kv_dim buffer).
    pub fn v_write_pos(&mut self, layer: usize, pos: usize, src: &[f32]) {
        let head_stride = self.max_seq * self.head_dim;
        for h in 0..self.n_kv_heads {
            let dst_off = (layer * self.n_kv_heads + h) * head_stride + pos * self.head_dim;
            let src_off = h * self.head_dim;
            self.v[dst_off..dst_off + self.head_dim]
                .copy_from_slice(&src[src_off..src_off + self.head_dim]);
        }
    }

    /// Base pointer for K data of a specific layer (head-contiguous layout).
    /// Layout within layer: `[head][pos][head_dim]`, stride between heads = max_seq * head_dim.
    pub fn k_layer_base(&self, layer: usize) -> *const f32 {
        let off = layer * self.n_kv_heads * self.max_seq * self.head_dim;
        unsafe { self.k.as_ptr().add(off) }
    }

    pub fn v_layer_base(&self, layer: usize) -> *const f32 {
        let off = layer * self.n_kv_heads * self.max_seq * self.head_dim;
        unsafe { self.v.as_ptr().add(off) }
    }

    /// Stride between heads (in floats): max_seq * head_dim.
    pub fn head_stride(&self) -> usize {
        self.max_seq * self.head_dim
    }
}

// ========================================================================
// RoPE Cache
// ========================================================================

pub struct RopeCache {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub inv_freq: Vec<f32>,
    pub cap: usize,
    pub head_dim: usize,
}

impl Default for RopeCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RopeCache {
    pub fn new() -> Self {
        RopeCache {
            cos: Vec::new(),
            sin: Vec::new(),
            inv_freq: Vec::new(),
            cap: 0,
            head_dim: 0,
        }
    }

    pub fn ensure(&mut self, required_pos: usize, head_dim: usize, theta: f32) {
        if self.head_dim != head_dim || self.inv_freq.is_empty() {
            let half = head_dim / 2;
            self.inv_freq = (0..half)
                .map(|d| 1.0 / theta.powf((2 * d) as f32 / head_dim as f32))
                .collect();
            self.head_dim = head_dim;
        }

        if required_pos <= self.cap {
            return;
        }

        let mut new_cap = if self.cap > 0 { self.cap } else { 1024 };
        while new_cap < required_pos {
            new_cap *= 2;
        }

        self.cos.resize(new_cap * head_dim, 0.0);
        self.sin.resize(new_cap * head_dim, 0.0);

        let half = head_dim / 2;
        for pos in self.cap..new_cap {
            let p = pos as f32;
            for d in 0..half {
                let angle = p * self.inv_freq[d];
                let c = angle.cos();
                let s = angle.sin();
                self.cos[pos * head_dim + d] = c;
                self.cos[pos * head_dim + half + d] = c;
                self.sin[pos * head_dim + d] = s;
                self.sin[pos * head_dim + half + d] = s;
            }
        }

        self.cap = new_cap;
    }

    pub fn cos_at(&self, pos: usize) -> &[f32] {
        &self.cos[pos * self.head_dim..(pos + 1) * self.head_dim]
    }

    pub fn sin_at(&self, pos: usize) -> &[f32] {
        &self.sin[pos * self.head_dim..(pos + 1) * self.head_dim]
    }

    pub fn cos_range(&self, start: usize, len: usize) -> &[f32] {
        &self.cos[start * self.head_dim..(start + len) * self.head_dim]
    }

    pub fn sin_range(&self, start: usize, len: usize) -> &[f32] {
        &self.sin[start * self.head_dim..(start + len) * self.head_dim]
    }
}

// ========================================================================
// Decoder Forward
// ========================================================================

pub struct DecoderBuffers {
    // Single-token decode buffers
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub ffn_out: Vec<f32>,

    // Prefill buffers
    pub pref_x: Vec<f32>,
    pub pref_x_norm: Vec<f32>,
    pub pref_q: Vec<f32>,
    pub pref_k: Vec<f32>,
    pub pref_v: Vec<f32>,
    pub pref_attn_out: Vec<f32>,
    pub pref_proj_out: Vec<f32>,
    pub pref_ffn_out: Vec<f32>,
    pub pref_gate_up: Vec<f32>,
    pub pref_gate: Vec<f32>,
    pub pref_seq_cap: usize,

    // Reusable scratch for BF16→F32 conversion in prefill path
    pub bf16_scratch: Vec<f32>,
}

impl DecoderBuffers {
    pub fn new(cfg: &QwenConfig) -> Self {
        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let intermediate = cfg.dec_intermediate;

        // Largest weight matrix is gate_up_fused: 2 * intermediate * hidden
        let max_weight = (2 * intermediate * dim).max(q_dim * dim).max(kv_dim * dim);

        DecoderBuffers {
            x: vec![0.0f32; dim],
            x_norm: vec![0.0f32; dim],
            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; q_dim],
            proj_out: vec![0.0f32; dim],
            gate_buf: vec![0.0f32; 2 * intermediate],
            ffn_out: vec![0.0f32; intermediate],
            pref_x: Vec::new(),
            pref_x_norm: Vec::new(),
            pref_q: Vec::new(),
            pref_k: Vec::new(),
            pref_v: Vec::new(),
            pref_attn_out: Vec::new(),
            pref_proj_out: Vec::new(),
            pref_ffn_out: Vec::new(),
            pref_gate_up: Vec::new(),
            pref_gate: Vec::new(),
            pref_seq_cap: 0,
            bf16_scratch: vec![0.0f32; max_weight],
        }
    }

    pub fn ensure_prefill(&mut self, seq_len: usize, cfg: &QwenConfig) {
        if seq_len <= self.pref_seq_cap {
            return;
        }

        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let intermediate = cfg.dec_intermediate;

        let mut new_cap = if self.pref_seq_cap > 0 {
            self.pref_seq_cap
        } else {
            64
        };
        while new_cap < seq_len {
            new_cap *= 2;
        }

        self.pref_x.resize(new_cap * dim, 0.0);
        self.pref_x_norm.resize(new_cap * dim, 0.0);
        self.pref_q.resize(new_cap * q_dim, 0.0);
        self.pref_k.resize(new_cap * kv_dim, 0.0);
        self.pref_v.resize(new_cap * kv_dim, 0.0);
        self.pref_attn_out.resize(new_cap * q_dim, 0.0);
        self.pref_proj_out.resize(new_cap * dim, 0.0);
        self.pref_ffn_out.resize(new_cap * dim, 0.0);
        self.pref_gate_up.resize(new_cap * 2 * intermediate, 0.0);
        self.pref_gate.resize(new_cap * intermediate, 0.0);
        self.pref_seq_cap = new_cap;
    }
}

/// Decoder prefill: process multiple tokens.
pub fn decoder_prefill(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // Ensure KV cache
    let needed = kv_cache.len + seq_len;
    if needed > kv_cache.max_seq {
        kv_cache.grow(needed + 1024);
    }

    bufs.ensure_prefill(seq_len, cfg);

    let x = &mut bufs.pref_x[..seq_len * dim];
    x.copy_from_slice(&input_embeds[..seq_len * dim]);

    let start_pos = kv_cache.len;
    rope.ensure(start_pos + seq_len, head_dim, theta);
    let rope_cos = rope.cos_range(start_pos, seq_len);
    let rope_sin = rope.sin_range(start_pos, seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();

    for (layer_idx, layer) in decoder.layers.iter().enumerate() {
        let x_norm = &mut bufs.pref_x_norm[..seq_len * dim];
        kernels::rms_norm(
            x_norm,
            &bufs.pref_x[..seq_len * dim],
            &layer.input_norm,
            seq_len,
            dim,
            eps,
        );

        let q = &mut bufs.pref_q[..seq_len * q_dim];
        let k = &mut bufs.pref_k[..seq_len * kv_dim];
        let v = &mut bufs.pref_v[..seq_len * kv_dim];

        kernels::linear_nobias(q, x_norm, &layer.wq_f32_prefill, seq_len, dim, q_dim);
        kernels::linear_nobias(k, x_norm, &layer.wk_f32_prefill, seq_len, dim, kv_dim);
        kernels::linear_nobias(v, x_norm, &layer.wv_f32_prefill, seq_len, dim, kv_dim);

        kernels::rms_norm_per_head(q, &layer.q_norm_weight, seq_len, n_heads, head_dim, eps);
        kernels::rms_norm_per_head(k, &layer.k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        kernels::apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        kernels::apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        // Store K, V in cache (scatter to head-contiguous layout)
        for s in 0..seq_len {
            kv_cache.k_write_pos(
                layer_idx,
                start_pos + s,
                &bufs.pref_k[s * kv_dim..(s + 1) * kv_dim],
            );
            kv_cache.v_write_pos(
                layer_idx,
                start_pos + s,
                &bufs.pref_v[s * kv_dim..(s + 1) * kv_dim],
            );
        }

        let total_seq = start_pos + seq_len;
        let k_base = kv_cache.k_layer_base(layer_idx);
        let v_base = kv_cache.v_layer_base(layer_idx);
        let head_stride = kv_cache.head_stride();

        let attn_out = &mut bufs.pref_attn_out[..seq_len * q_dim];
        kernels::causal_attention(
            attn_out,
            q,
            k_base,
            v_base,
            head_stride,
            seq_len,
            total_seq,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            start_pos,
        );

        let proj_out = &mut bufs.pref_proj_out[..seq_len * dim];
        kernels::linear_nobias(proj_out, attn_out, &layer.wo_f32_prefill, seq_len, q_dim, dim);
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], proj_out, seq_len * dim);

        // Post-attention RMSNorm + SwiGLU MLP
        let x_norm2 = &mut bufs.pref_x_norm[..seq_len * dim];
        kernels::rms_norm(
            x_norm2,
            &bufs.pref_x[..seq_len * dim],
            &layer.post_attn_norm,
            seq_len,
            dim,
            eps,
        );

        let gate_up = &mut bufs.pref_gate_up[..seq_len * 2 * intermediate];
        kernels::linear_nobias(gate_up, x_norm2, &layer.gate_up_f32_prefill, seq_len, dim, 2 * intermediate);

        let gate = &mut bufs.pref_gate[..seq_len * intermediate];
        kernels::swiglu_multiply(gate, gate_up, seq_len, intermediate);

        let ffn_out = &mut bufs.pref_ffn_out[..seq_len * dim];
        kernels::linear_nobias(ffn_out, gate, &layer.down_f32_prefill, seq_len, intermediate, dim);
        kernels::add_inplace(&mut bufs.pref_x[..seq_len * dim], ffn_out, seq_len * dim);
    }

    kv_cache.len = start_pos + seq_len;
}

/// Decoder single-token forward: returns greedy token ID.
pub fn decoder_forward(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embed: &[f32],
) -> i32 {
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    bufs.x[..dim].copy_from_slice(&input_embed[..dim]);

    let pos = kv_cache.len;

    if pos >= kv_cache.max_seq {
        kv_cache.grow(pos + 1024);
    }

    rope.ensure(pos + 1, head_dim, theta);
    let rope_cos = rope.cos_at(pos);
    let rope_sin = rope.sin_at(pos);

    let scale = 1.0 / (head_dim as f32).sqrt();

    for (layer_idx, layer) in decoder.layers.iter().enumerate() {
        kernels::rms_norm(
            &mut bufs.x_norm[..dim],
            &bufs.x[..dim],
            &layer.input_norm,
            1,
            dim,
            eps,
        );

        // QKV projection — dispatch on QWeight type
        match (&layer.wq, &layer.wk, &layer.wv) {
            (
                QWeight::Int8Owned { data: dq, scales: sq },
                QWeight::Int8Owned { data: dk, scales: sk },
                QWeight::Int8Owned { data: dv, scales: sv },
            ) => kernels::linear_nobias_int8_qkv(
                &mut bufs.q[..q_dim], &mut bufs.k[..kv_dim], &mut bufs.v[..kv_dim],
                &bufs.x_norm[..dim], dq, sq, dk, sk, dv, sv, dim, q_dim, kv_dim,
            ),
            (wq, wk, wv) => {
                kernels::qweight_matvec(&mut bufs.q[..q_dim],   &bufs.x_norm[..dim], wq, q_dim, dim);
                kernels::qweight_matvec(&mut bufs.k[..kv_dim],  &bufs.x_norm[..dim], wk, kv_dim, dim);
                kernels::qweight_matvec(&mut bufs.v[..kv_dim],  &bufs.x_norm[..dim], wv, kv_dim, dim);
            }
        }

        kernels::rms_norm_per_head(
            &mut bufs.q[..q_dim],
            &layer.q_norm_weight,
            1,
            n_heads,
            head_dim,
            eps,
        );
        kernels::rms_norm_per_head(
            &mut bufs.k[..kv_dim],
            &layer.k_norm_weight,
            1,
            n_kv_heads,
            head_dim,
            eps,
        );

        kernels::apply_rope_neox(
            &mut bufs.q[..q_dim],
            rope_cos,
            rope_sin,
            1,
            n_heads,
            head_dim,
        );
        kernels::apply_rope_neox(
            &mut bufs.k[..kv_dim],
            rope_cos,
            rope_sin,
            1,
            n_kv_heads,
            head_dim,
        );

        kv_cache.k_write_pos(layer_idx, pos, &bufs.k[..kv_dim]);
        kv_cache.v_write_pos(layer_idx, pos, &bufs.v[..kv_dim]);

        let total_seq = pos + 1;
        let k_base = kv_cache.k_layer_base(layer_idx);
        let v_base = kv_cache.v_layer_base(layer_idx);
        let head_stride = kv_cache.head_stride();

        kernels::causal_attention(
            &mut bufs.attn_out[..q_dim],
            &bufs.q[..q_dim],
            k_base,
            v_base,
            head_stride,
            1,
            total_seq,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            pos,
        );

        // O-projection: x += attn_out @ wo — dispatch on weight type
        match &layer.wo {
            QWeight::Int8Owned { data, scales } => {
                kernels::linear_nobias_int8_addto(&mut bufs.x[..dim], &bufs.attn_out[..q_dim], data, scales, q_dim, dim);
            }
            QWeight::Bf16Ptr(ptr) => {
                kernels::linear_nobias_bf16_addto(&mut bufs.x[..dim], &bufs.attn_out[..q_dim], *ptr, q_dim, dim);
            }
            w => {
                kernels::qweight_matvec(&mut bufs.proj_out[..dim], &bufs.attn_out[..q_dim], w, dim, q_dim);
                kernels::add_inplace(&mut bufs.x[..dim], &bufs.proj_out[..dim], dim);
            }
        }

        kernels::rms_norm(
            &mut bufs.x_norm[..dim],
            &bufs.x[..dim],
            &layer.post_attn_norm,
            1,
            dim,
            eps,
        );

        // gate_up SwiGLU — dispatch on GateUpWeights variant
        match &layer.gate_up {
            GateUpWeights::Int8Fused { data, scales } => {
                kernels::linear_nobias_int8_swiglu(&mut bufs.ffn_out[..intermediate], &bufs.x_norm[..dim], data, scales, dim, intermediate);
            }
            GateUpWeights::Bf16Fused(fused) => {
                kernels::linear_nobias_bf16_swiglu(&mut bufs.ffn_out[..intermediate], &bufs.x_norm[..dim], fused.as_ptr(), dim, intermediate);
            }
            GateUpWeights::Separate { gate, up } => {
                kernels::qweight_swiglu_separate(&mut bufs.ffn_out[..intermediate], &bufs.x_norm[..dim], gate, up, &mut bufs.gate_buf[..2*intermediate], dim, intermediate);
            }
        }

        // down-projection: x += ffn_out @ down — dispatch on weight type
        match &layer.down {
            QWeight::Int8Owned { data, scales } => {
                kernels::linear_nobias_int8_addto(&mut bufs.x[..dim], &bufs.ffn_out[..intermediate], data, scales, intermediate, dim);
            }
            QWeight::Bf16Ptr(ptr) => {
                kernels::linear_nobias_bf16_addto(&mut bufs.x[..dim], &bufs.ffn_out[..intermediate], *ptr, intermediate, dim);
            }
            w => {
                kernels::qweight_matvec(&mut bufs.proj_out[..dim], &bufs.ffn_out[..intermediate], w, dim, intermediate);
                kernels::add_inplace(&mut bufs.x[..dim], &bufs.proj_out[..dim], dim);
            }
        }
    }

    kv_cache.len = pos + 1;

    // Final norm + streaming argmax (use x_norm as temp to avoid heap allocation)
    kernels::rms_norm(
        &mut bufs.x_norm[..dim],
        &bufs.x[..dim],
        &decoder.norm,
        1,
        dim,
        eps,
    );
    bufs.x[..dim].copy_from_slice(&bufs.x_norm[..dim]);
    let lm_out_dim = cfg.lm_head_dim();

    let lm_weight = decoder.lm_head.as_ref().unwrap_or(&decoder.tok_embeddings);
    kernels::argmax_qweight(&bufs.x[..dim], lm_weight, dim, lm_out_dim) as i32
}

/// Decoder prefill that returns per-position logits (for forced aligner).
/// Returns `[seq_len × out_dim]` logits where out_dim = classify_num.
pub fn decoder_prefill_logits(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut DecoderBuffers,
    input_embeds: &[f32],
    seq_len: usize,
) -> Vec<f32> {
    let dim = cfg.dec_hidden;
    let eps = cfg.dec_rms_norm_eps;
    let out_dim = cfg.lm_head_dim();

    // Run the standard prefill to get hidden states
    decoder_prefill(decoder, cfg, kv_cache, rope, bufs, input_embeds, seq_len);

    // After prefill, pref_x contains the final hidden states for all positions.
    // Apply final RMS norm and lm_head projection.
    let x = &bufs.pref_x[..seq_len * dim];
    let mut x_norm = vec![0.0f32; seq_len * dim];
    kernels::rms_norm(&mut x_norm, x, &decoder.norm, seq_len, dim, eps);

    let lm_weight = decoder.lm_head.as_ref().unwrap_or(&decoder.tok_embeddings);

    // Project each position through lm_head: [seq_len × dim] × [out_dim × dim]^T
    let mut logits = vec![0.0f32; seq_len * out_dim];
    match lm_weight {
        QWeight::Bf16Ptr(ptr) => unsafe {
            kernels::linear_nobias_bf16_scratch(&mut logits, &x_norm, *ptr, seq_len, dim, out_dim, &mut bufs.bf16_scratch);
        },
        QWeight::Int8Owned { .. } => {
            for s in 0..seq_len {
                let x_row = &x_norm[s * dim..(s + 1) * dim];
                let out_row = &mut logits[s * out_dim..(s + 1) * out_dim];
                kernels::qweight_matvec(out_row, x_row, lm_weight, out_dim, dim);
            }
        }
        w => {
            for s in 0..seq_len {
                let x_row = &x_norm[s * dim..(s + 1) * dim];
                let out_row = &mut logits[s * out_dim..(s + 1) * out_dim];
                kernels::qweight_matvec(out_row, x_row, w, out_dim, dim);
            }
        }
    }

    logits
}

/// Extract one token's embedding row from any `QWeight` type.
pub fn tok_embed_to_f32(dst: &mut [f32], w: &QWeight, token_id: i32, dim: usize) {
    let tid = token_id as usize;
    match w {
        QWeight::Bf16Ptr(ptr) => unsafe {
            let src = std::slice::from_raw_parts(ptr.add(tid * dim), dim);
            kernels::bf16_to_f32_buf(dst, src);
        },
        QWeight::Int8Owned { data, scales } => {
            let scale = scales[tid];
            let base = tid * dim;
            for i in 0..dim {
                dst[i] = data[base + i] as f32 * scale;
            }
        }
        QWeight::Q8_0Ptr(ptr) => {
            gguf::dequant_q8_0_row(dst, *ptr, tid, dim);
        }
        QWeight::Q4KPtr(ptr) => {
            gguf::dequant_q4k_row(dst, *ptr, tid, dim);
        }
        QWeight::Q4_0Ptr(ptr) => {
            gguf::dequant_q4_0_row(dst, *ptr, tid, dim);
        }
    }
}

/// Legacy alias kept for callers that haven't migrated yet.
///
/// # Safety
/// `tok_emb_bf16` must point to valid BF16 data for at least `(token_id + 1) * dim` elements.
pub unsafe fn tok_embed_bf16_to_f32(
    dst: &mut [f32],
    tok_emb_bf16: *const u16,
    token_id: i32,
    dim: usize,
) {
    let w = QWeight::Bf16Ptr(tok_emb_bf16);
    tok_embed_to_f32(dst, &w, token_id, dim);
}
