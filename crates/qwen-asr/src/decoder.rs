//! Qwen3 LLM decoder with GQA, KV cache, and generation.

use crate::config::*;
use crate::kernels;
use crate::safetensors::MultiSafetensors;

use crate::kernels::superpage_vec;

/// Quantize a BF16 weight matrix into INT8 and store the (large) INT8 buffer
/// in superpage-aligned memory.
fn quantize_to_superpage(
    w_bf16: *const u16,
    out_dim: usize,
    in_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    // Safety: callers pass tensor pointers from the mmap'd model file, which
    // cover at least out_dim * in_dim BF16 values and outlive this call.
    let (int8, scales) =
        unsafe { kernels::quantize_bf16_weights_to_int8(w_bf16, out_dim, in_dim) };
    let mut sp = superpage_vec::<i8>(int8.len());
    sp.copy_from_slice(&int8);
    (sp, scales)
}

pub struct DecLayer {
    pub wq_weight_bf16: *const u16,
    pub wk_weight_bf16: *const u16,
    pub wv_weight_bf16: *const u16,
    pub wo_weight_bf16: *const u16,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub gate_weight_bf16: *const u16,
    pub up_weight_bf16: *const u16,
    pub down_weight_bf16: *const u16,
    /// Owned interleaved bf16 fusion of gate+up — populated only for non-aligner
    /// configs (single-token decode path uses it). Empty for forced-aligner since
    /// aligner only ever runs prefill, which streams gate/up separately.
    pub gate_up_fused_bf16: Vec<u16>,
    /// INT8 quantized attention weights + per-row scales — populated only for
    /// non-aligner configs (used by aarch64 single-token decode). Empty for aligner.
    pub wq_int8: Vec<i8>,
    pub wq_int8_scales: Vec<f32>,
    pub wk_int8: Vec<i8>,
    pub wk_int8_scales: Vec<f32>,
    pub wv_int8: Vec<i8>,
    pub wv_int8_scales: Vec<f32>,
    pub wo_int8: Vec<i8>,
    pub wo_int8_scales: Vec<f32>,
    pub gate_up_int8: Vec<i8>,
    pub gate_up_int8_scales: Vec<f32>,
    pub down_int8: Vec<i8>,
    pub down_int8_scales: Vec<f32>,
}

unsafe impl Send for DecLayer {}
unsafe impl Sync for DecLayer {}

pub struct Decoder {
    pub tok_embeddings_bf16: *const u16,
    pub layers: Vec<DecLayer>,
    pub norm: Vec<f32>,
    /// Separate lm_head for forced aligner (None = tied weights with tok_embeddings)
    pub lm_head_bf16: Option<*const u16>,
    /// INT8 quantized lm_head weights for fast argmax — None for aligner (uses bf16 path).
    pub lm_head_int8: Option<Vec<i8>>,
    pub lm_head_int8_scales: Option<Vec<f32>>,
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

/// Load a single decoder layer. For non-aligner configs we still build the
/// INT8 + interleaved-bf16 fusion that the single-token decode path needs;
/// for aligner configs (which only ever runs prefill) we skip those — saving
/// roughly 700 MB across the model — and let the prefill code stream BF16
/// weights through a shared f32 scratch buffer.
fn load_dec_layer(ms: &MultiSafetensors, cfg: &QwenConfig, i: usize) -> Option<DecLayer> {
    let lp = format!("thinker.model.layers.{}", i);

    let wq = load_bf16_direct(ms, &format!("{}.self_attn.q_proj.weight", lp))?;
    let wk = load_bf16_direct(ms, &format!("{}.self_attn.k_proj.weight", lp))?;
    let wv = load_bf16_direct(ms, &format!("{}.self_attn.v_proj.weight", lp))?;
    let wo = load_bf16_direct(ms, &format!("{}.self_attn.o_proj.weight", lp))?;
    let q_dim = cfg.dec_heads * cfg.dec_head_dim;
    let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
    let hidden = cfg.dec_hidden;
    let inter = cfg.dec_intermediate;

    let q_norm = load_f32(ms, &format!("{}.self_attn.q_norm.weight", lp))?;
    let k_norm = load_f32(ms, &format!("{}.self_attn.k_norm.weight", lp))?;
    let input_norm = load_f32(ms, &format!("{}.input_layernorm.weight", lp))?;
    let post_attn_norm = load_f32(ms, &format!("{}.post_attention_layernorm.weight", lp))?;

    let gate_bf16 = load_bf16_direct(ms, &format!("{}.mlp.gate_proj.weight", lp))?;
    let up_bf16 = load_bf16_direct(ms, &format!("{}.mlp.up_proj.weight", lp))?;
    let down_bf16 = load_bf16_direct(ms, &format!("{}.mlp.down_proj.weight", lp))?;

    let is_aligner = cfg.is_aligner();

    // Non-aligner: build interleaved bf16 fusion + INT8 quant for fast decode.
    // Aligner: leave them empty (saves ~700 MB across the model).
    let (gate_up_fused_bf16, wq_int8, wq_int8_scales, wk_int8, wk_int8_scales,
         wv_int8, wv_int8_scales, wo_int8, wo_int8_scales,
         gate_up_int8, gate_up_int8_scales, down_int8, down_int8_scales) = if is_aligner {
        (Vec::new(),
         Vec::new(), Vec::new(), Vec::new(), Vec::new(),
         Vec::new(), Vec::new(), Vec::new(), Vec::new(),
         Vec::new(), Vec::new(), Vec::new(), Vec::new())
    } else {
        // Fuse gate+up: interleave rows
        let mut gate_up_fused = vec![0u16; 2 * inter * hidden];
        unsafe {
            let gate_slice = std::slice::from_raw_parts(gate_bf16, inter * hidden);
            let up_slice = std::slice::from_raw_parts(up_bf16, inter * hidden);
            for r in 0..inter {
                gate_up_fused[2 * r * hidden..(2 * r + 1) * hidden]
                    .copy_from_slice(&gate_slice[r * hidden..(r + 1) * hidden]);
                gate_up_fused[(2 * r + 1) * hidden..(2 * r + 2) * hidden]
                    .copy_from_slice(&up_slice[r * hidden..(r + 1) * hidden]);
            }
        }

        let (wq_int8, wq_int8_scales) = quantize_to_superpage(wq, q_dim, hidden);
        let (wk_int8, wk_int8_scales) = quantize_to_superpage(wk, kv_dim, hidden);
        let (wv_int8, wv_int8_scales) = quantize_to_superpage(wv, kv_dim, hidden);
        let (wo_int8, wo_int8_scales) = quantize_to_superpage(wo, hidden, q_dim);
        let (gate_up_int8, gate_up_int8_scales) =
            quantize_to_superpage(gate_up_fused.as_ptr(), 2 * inter, hidden);
        let (down_int8, down_int8_scales) = quantize_to_superpage(down_bf16, hidden, inter);

        (gate_up_fused,
         wq_int8, wq_int8_scales, wk_int8, wk_int8_scales,
         wv_int8, wv_int8_scales, wo_int8, wo_int8_scales,
         gate_up_int8, gate_up_int8_scales, down_int8, down_int8_scales)
    };

    Some(DecLayer {
        wq_weight_bf16: wq,
        wk_weight_bf16: wk,
        wv_weight_bf16: wv,
        wo_weight_bf16: wo,
        q_norm_weight: q_norm,
        k_norm_weight: k_norm,
        input_norm,
        post_attn_norm,
        gate_weight_bf16: gate_bf16,
        up_weight_bf16: up_bf16,
        down_weight_bf16: down_bf16,
        gate_up_fused_bf16,
        wq_int8,
        wq_int8_scales,
        wk_int8,
        wk_int8_scales,
        wv_int8,
        wv_int8_scales,
        wo_int8,
        wo_int8_scales,
        gate_up_int8,
        gate_up_int8_scales,
        down_int8,
        down_int8_scales,
    })
}

impl Decoder {
    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig) -> Option<Self> {
        let tok_embeddings_bf16 = load_bf16_direct(ms, "thinker.model.embed_tokens.weight")?;

        // Per-layer weight loading is independent and conversion-heavy
        // (bf16->f32 prefill + INT8 quantization), so load layers in parallel.
        let nlayers = cfg.dec_layers;
        let nthreads = kernels::get_num_cpus().min(nlayers).max(1);
        let chunk = nlayers.div_ceil(nthreads);
        let mut indexed: Vec<(usize, DecLayer)> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..nthreads {
                let start = t * chunk;
                let end = ((t + 1) * chunk).min(nlayers);
                if start >= end {
                    break;
                }
                handles.push(s.spawn(move || {
                    let mut out = Vec::with_capacity(end - start);
                    for i in start..end {
                        out.push((i, load_dec_layer(ms, cfg, i)?));
                    }
                    Some(out)
                }));
            }
            let mut all: Vec<(usize, DecLayer)> = Vec::with_capacity(nlayers);
            for h in handles {
                all.extend(h.join().ok()??);
            }
            Some(all)
        })?;
        indexed.sort_by_key(|(i, _)| *i);
        let layers: Vec<DecLayer> = indexed.into_iter().map(|(_, l)| l).collect();

        let norm = load_f32(ms, "thinker.model.norm.weight")?;

        // Load separate lm_head if present (forced aligner has untied lm_head)
        let lm_head_bf16 = if cfg.classify_num > 0 {
            let ptr = load_bf16_direct(ms, "thinker.lm_head.weight")?;
            Some(ptr)
        } else {
            // For normal ASR, lm_head is tied with tok_embeddings (no separate weight)
            ms.get_bf16_direct("thinker.lm_head.weight")
        };

        // Quantize lm_head to INT8 for fast argmax — skipped for aligner since
        // aligner never does single-token decode (all logits read out of prefill).
        let (lm_int8_opt, lm_scales_opt) = if cfg.is_aligner() {
            (None, None)
        } else {
            let lm_weight = lm_head_bf16.unwrap_or(tok_embeddings_bf16);
            let lm_out_dim = cfg.lm_head_dim();
            let lm_in_dim = cfg.dec_hidden;
            let (lm_int8, lm_scales) = quantize_to_superpage(lm_weight, lm_out_dim, lm_in_dim);
            (Some(lm_int8), Some(lm_scales))
        };

        Some(Decoder {
            tok_embeddings_bf16,
            layers,
            norm,
            lm_head_bf16,
            lm_head_int8: lm_int8_opt,
            lm_head_int8_scales: lm_scales_opt,
        })
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
            k: superpage_vec::<f32>(total),
            v: superpage_vec::<f32>(total),
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

        let mut new_k = superpage_vec::<f32>(total);
        let mut new_v = superpage_vec::<f32>(total);

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

    /// INT8 quantization scratch for the fused single-token decode region
    /// (aarch64): quantized layer input (`dim`), attention output (`q_dim`)
    /// and FFN activation (`intermediate`). Reused across layers/tokens so the
    /// region never allocates.
    pub x_int8: Vec<i8>,
    pub attn_int8: Vec<i8>,
    pub ffn_int8: Vec<i8>,

    // Prefill buffers
    pub pref_x: Vec<f32>,
    pub pref_x_norm: Vec<f32>,
    pub pref_q: Vec<f32>,
    pub pref_k: Vec<f32>,
    pub pref_v: Vec<f32>,
    pub pref_attn_out: Vec<f32>,
    pub pref_proj_out: Vec<f32>,
    pub pref_ffn_out: Vec<f32>,
    /// Separate gate and up projection outputs — replaces the old single fused
    /// pref_gate_up buffer (which paired with the 700 MB owned bf16 fusion).
    pub pref_gate: Vec<f32>,
    pub pref_up: Vec<f32>,
    pub pref_seq_cap: usize,

    /// Reusable scratch for BF16→F32 conversion in prefill path. Sized to the
    /// largest weight matrix the decoder ever streams in (down_proj or gate/up).
    pub bf16_scratch: Vec<f32>,
}

impl DecoderBuffers {
    pub fn new(cfg: &QwenConfig) -> Self {
        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let intermediate = cfg.dec_intermediate;

        // Largest weight matrix the prefill streams: down_proj (hidden*inter)
        // or gate/up (inter*hidden). lm_head can be bigger for non-aligner —
        // grown lazily by ensure_scratch.
        let max_weight = (intermediate * dim).max(q_dim * dim).max(kv_dim * dim);

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
            x_int8: vec![0i8; dim],
            attn_int8: vec![0i8; q_dim],
            ffn_int8: vec![0i8; intermediate],
            pref_x: Vec::new(),
            pref_x_norm: Vec::new(),
            pref_q: Vec::new(),
            pref_k: Vec::new(),
            pref_v: Vec::new(),
            pref_attn_out: Vec::new(),
            pref_proj_out: Vec::new(),
            pref_ffn_out: Vec::new(),
            pref_gate: Vec::new(),
            pref_up: Vec::new(),
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
        self.pref_gate.resize(new_cap * intermediate, 0.0);
        self.pref_up.resize(new_cap * intermediate, 0.0);
        self.pref_seq_cap = new_cap;
    }

    pub fn ensure_scratch(&mut self, n: usize) {
        if self.bf16_scratch.len() < n {
            self.bf16_scratch.resize(n, 0.0);
        }
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

    // Make sure the BF16→F32 scratch is sized to the largest weight matrix the
    // prefill will stream through: down_proj (hidden × intermediate). Some
    // attention matrices are smaller; gate/up are (intermediate × hidden) which
    // is the same size.
    let max_weight = (intermediate * dim).max(q_dim * dim).max(kv_dim * dim);
    bufs.ensure_scratch(max_weight);

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

        unsafe {
            kernels::linear_nobias_bf16_scratch(
                q, x_norm, layer.wq_weight_bf16, seq_len, dim, q_dim,
                &mut bufs.bf16_scratch,
            );
            kernels::linear_nobias_bf16_scratch(
                k, x_norm, layer.wk_weight_bf16, seq_len, dim, kv_dim,
                &mut bufs.bf16_scratch,
            );
            kernels::linear_nobias_bf16_scratch(
                v, x_norm, layer.wv_weight_bf16, seq_len, dim, kv_dim,
                &mut bufs.bf16_scratch,
            );
        }

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
        unsafe {
            kernels::linear_nobias_bf16_scratch(
                proj_out, attn_out, layer.wo_weight_bf16, seq_len, q_dim, dim,
                &mut bufs.bf16_scratch,
            );
        }
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

        // Two separate GEMMs into pref_gate/pref_up, then fused swiglu — replaces
        // the prior interleaved-fusion path that needed an extra ~700 MB owned
        // bf16 matrix across the decoder.
        let gate = &mut bufs.pref_gate[..seq_len * intermediate];
        unsafe {
            kernels::linear_nobias_bf16_scratch(
                gate, x_norm2, layer.gate_weight_bf16, seq_len, dim, intermediate,
                &mut bufs.bf16_scratch,
            );
        }
        let up = &mut bufs.pref_up[..seq_len * intermediate];
        unsafe {
            kernels::linear_nobias_bf16_scratch(
                up, x_norm2, layer.up_weight_bf16, seq_len, dim, intermediate,
                &mut bufs.bf16_scratch,
            );
        }

        // gate <- silu(gate) * up
        kernels::swiglu_separate_inplace(gate, up, seq_len, intermediate);

        let ffn_out = &mut bufs.pref_ffn_out[..seq_len * dim];
        unsafe {
            kernels::linear_nobias_bf16_scratch(
                ffn_out, gate, layer.down_weight_bf16, seq_len, intermediate, dim,
                &mut bufs.bf16_scratch,
            );
        }
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

    // aarch64: run the whole 28-layer single-token loop inside ONE persistent
    // parallel region. Workers stay resident across every stage of every layer,
    // synchronized by spin barriers, instead of one thread-pool dispatch/join
    // cycle per stage (~5 per layer). Row/head partitions are identical to the
    // standalone threaded kernels, and each output element is written by
    // exactly one thread, so results are bit-identical to the dispatch-per-stage
    // path. Serial glue (norms, RoPE, KV write, activation quantization) runs on
    // tid 0 between barriers — microsecond-scale work at dim 1024.
    #[cfg(target_arch = "aarch64")]
    {
        let total_seq = pos + 1;
        let n_kv = kv_cache.n_kv_heads;
        let kv_head_stride = kv_cache.head_stride();
        let layers: &[DecLayer] = &decoder.layers;

        // Shared mutable state, published across threads only via barriers.
        // scales[0]: QKV input, [1]: O-proj input, [2]: SwiGLU input, [3]: down-proj input.
        let mut scales = [0.0f32; 4];
        let scales_ptr = scales.as_mut_ptr() as usize;

        let x_ptr = bufs.x.as_mut_ptr() as usize;
        let x_norm_ptr = bufs.x_norm.as_mut_ptr() as usize;
        let q_ptr = bufs.q.as_mut_ptr() as usize;
        let k_ptr = bufs.k.as_mut_ptr() as usize;
        let v_ptr = bufs.v.as_mut_ptr() as usize;
        let attn_out_ptr = bufs.attn_out.as_mut_ptr() as usize;
        let ffn_out_ptr = bufs.ffn_out.as_mut_ptr() as usize;
        let x_int8_ptr = bufs.x_int8.as_mut_ptr() as usize;
        let attn_int8_ptr = bufs.attn_int8.as_mut_ptr() as usize;
        let ffn_int8_ptr = bufs.ffn_int8.as_mut_ptr() as usize;
        let kv_k_ptr = kv_cache.k.as_mut_ptr() as usize;
        let kv_v_ptr = kv_cache.v.as_mut_ptr() as usize;

        kernels::parallel_region(|barrier, tid, nt| {
            for (layer_idx, layer) in layers.iter().enumerate() {
                let layer_kv_off = layer_idx * n_kv * kv_head_stride;
                let k_base = unsafe { (kv_k_ptr as *const f32).add(layer_kv_off) };
                let v_base = unsafe { (kv_v_ptr as *const f32).add(layer_kv_off) };

                // Stage: pre-attention norm + input quantization (tid 0).
                if tid == 0 {
                    let x = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, dim) };
                    let x_norm = unsafe { std::slice::from_raw_parts_mut(x_norm_ptr as *mut f32, dim) };
                    kernels::rms_norm(x_norm, x, &layer.input_norm, 1, dim, eps);
                    let x_int8 = unsafe { std::slice::from_raw_parts_mut(x_int8_ptr as *mut i8, dim) };
                    unsafe { *(scales_ptr as *mut f32) = kernels::quantize_into(x_int8, x_norm); }
                }
                barrier.wait();

                // Stage: fused QKV projection, split over q|k|v output rows.
                {
                    let (s, e) = kernels::range_for(tid, nt, q_dim + 2 * kv_dim);
                    let x_scale = unsafe { *(scales_ptr as *const f32) };
                    unsafe {
                        kernels::int8_qkv_range(
                            q_ptr as *mut f32, k_ptr as *mut f32, v_ptr as *mut f32,
                            x_int8_ptr as *const i8, x_scale,
                            layer.wq_int8.as_ptr(), layer.wq_int8_scales.as_ptr(),
                            layer.wk_int8.as_ptr(), layer.wk_int8_scales.as_ptr(),
                            layer.wv_int8.as_ptr(), layer.wv_int8_scales.as_ptr(),
                            dim, q_dim, kv_dim, s, e,
                        );
                    }
                }
                barrier.wait();

                // Stage: q/k norms, RoPE, KV-cache write (tid 0 serial glue).
                if tid == 0 {
                    let q = unsafe { std::slice::from_raw_parts_mut(q_ptr as *mut f32, q_dim) };
                    let k = unsafe { std::slice::from_raw_parts_mut(k_ptr as *mut f32, kv_dim) };
                    kernels::rms_norm_per_head(q, &layer.q_norm_weight, 1, n_heads, head_dim, eps);
                    kernels::rms_norm_per_head(k, &layer.k_norm_weight, 1, n_kv_heads, head_dim, eps);
                    kernels::apply_rope_neox(q, rope_cos, rope_sin, 1, n_heads, head_dim);
                    kernels::apply_rope_neox(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

                    // Head-contiguous KV write at `pos` (same as k/v_write_pos).
                    let v = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, kv_dim) };
                    for h in 0..n_kv {
                        let dst = layer_kv_off + h * kv_head_stride + pos * head_dim;
                        let src = h * head_dim;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                k.as_ptr().add(src), (kv_k_ptr as *mut f32).add(dst), head_dim);
                            std::ptr::copy_nonoverlapping(
                                v.as_ptr().add(src), (kv_v_ptr as *mut f32).add(dst), head_dim);
                        }
                    }
                }
                barrier.wait();

                // Stage: causal attention, split by KV-head group (GQA-paired).
                if let Some((h0, h1)) = kernels::attn_head_range(tid, nt, 1, n_heads, n_kv_heads) {
                    let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr as *mut f32, q_dim) };
                    let q = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, q_dim) };
                    kernels::causal_attention_heads(
                        attn_out, q, k_base, v_base, kv_head_stride,
                        1, total_seq, n_heads, n_kv_heads, head_dim, scale, pos, h0, h1,
                    );
                }
                barrier.wait();

                // Stage: quantize attention output for the O-projection (tid 0).
                if tid == 0 {
                    let attn_out = unsafe { std::slice::from_raw_parts(attn_out_ptr as *const f32, q_dim) };
                    let attn_int8 = unsafe { std::slice::from_raw_parts_mut(attn_int8_ptr as *mut i8, q_dim) };
                    unsafe { *(scales_ptr as *mut f32).add(1) = kernels::quantize_into(attn_int8, attn_out); }
                }
                barrier.wait();

                // Stage: O-projection with fused residual add (x += attn @ wo).
                // Each output row of x is owned by exactly one thread.
                {
                    let (s, e) = kernels::range_for(tid, nt, dim);
                    let x_scale = unsafe { *(scales_ptr as *const f32).add(1) };
                    unsafe {
                        kernels::int8_matvec_range(
                            x_ptr as *mut f32, attn_int8_ptr as *const i8, x_scale,
                            layer.wo_int8.as_ptr(), layer.wo_int8_scales.as_ptr(),
                            Some(x_ptr as *const f32), q_dim, s, e,
                        );
                    }
                }
                barrier.wait();

                // Stage: post-attention norm + input quantization (tid 0).
                if tid == 0 {
                    let x = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, dim) };
                    let x_norm = unsafe { std::slice::from_raw_parts_mut(x_norm_ptr as *mut f32, dim) };
                    kernels::rms_norm(x_norm, x, &layer.post_attn_norm, 1, dim, eps);
                    let x_int8 = unsafe { std::slice::from_raw_parts_mut(x_int8_ptr as *mut i8, dim) };
                    unsafe { *(scales_ptr as *mut f32).add(2) = kernels::quantize_into(x_int8, x_norm); }
                }
                barrier.wait();

                // Stage: fused gate_up + SwiGLU, split over intermediate rows.
                {
                    let (s, e) = kernels::range_for(tid, nt, intermediate);
                    let x_scale = unsafe { *(scales_ptr as *const f32).add(2) };
                    unsafe {
                        kernels::int8_swiglu_range(
                            ffn_out_ptr as *mut f32, x_int8_ptr as *const i8, x_scale,
                            layer.gate_up_int8.as_ptr(), layer.gate_up_int8_scales.as_ptr(),
                            dim, s, e,
                        );
                    }
                }
                barrier.wait();

                // Stage: quantize FFN activation for the down-projection (tid 0).
                if tid == 0 {
                    let ffn_out = unsafe { std::slice::from_raw_parts(ffn_out_ptr as *const f32, intermediate) };
                    let ffn_int8 = unsafe { std::slice::from_raw_parts_mut(ffn_int8_ptr as *mut i8, intermediate) };
                    unsafe { *(scales_ptr as *mut f32).add(3) = kernels::quantize_into(ffn_int8, ffn_out); }
                }
                barrier.wait();

                // Stage: down-projection with fused residual add (x += ffn @ down).
                {
                    let (s, e) = kernels::range_for(tid, nt, dim);
                    let x_scale = unsafe { *(scales_ptr as *const f32).add(3) };
                    unsafe {
                        kernels::int8_matvec_range(
                            x_ptr as *mut f32, ffn_int8_ptr as *const i8, x_scale,
                            layer.down_int8.as_ptr(), layer.down_int8_scales.as_ptr(),
                            Some(x_ptr as *const f32), intermediate, s, e,
                        );
                    }
                }
                // Next layer's tid-0 norm reads every row of x; the region join
                // covers the last layer, and this barrier covers the rest.
                barrier.wait();
            }
        });
    }

    // Non-aarch64 (BF16) path: unchanged dispatch-per-stage layer loop.
    #[cfg(not(target_arch = "aarch64"))]
    for (layer_idx, layer) in decoder.layers.iter().enumerate() {
        kernels::rms_norm(
            &mut bufs.x_norm[..dim],
            &bufs.x[..dim],
            &layer.input_norm,
            1,
            dim,
            eps,
        );

        unsafe {
            kernels::linear_nobias_bf16(&mut bufs.q[..q_dim], &bufs.x_norm[..dim], layer.wq_weight_bf16, 1, dim, q_dim);
            kernels::linear_nobias_bf16(&mut bufs.k[..kv_dim], &bufs.x_norm[..dim], layer.wk_weight_bf16, 1, dim, kv_dim);
            kernels::linear_nobias_bf16(&mut bufs.v[..kv_dim], &bufs.x_norm[..dim], layer.wv_weight_bf16, 1, dim, kv_dim);
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

        // O-projection with fused residual add: x += attn_out @ wo
        kernels::linear_nobias_bf16_addto(
            &mut bufs.x[..dim],
            &bufs.attn_out[..q_dim],
            layer.wo_weight_bf16,
            q_dim,
            dim,
        );

        kernels::rms_norm(
            &mut bufs.x_norm[..dim],
            &bufs.x[..dim],
            &layer.post_attn_norm,
            1,
            dim,
            eps,
        );

        // gate_up + SwiGLU
        kernels::linear_nobias_bf16_swiglu(
            &mut bufs.ffn_out[..intermediate],
            &bufs.x_norm[..dim],
            layer.gate_up_fused_bf16.as_ptr(),
            dim,
            intermediate,
        );

        // down-projection with fused residual add: x += ffn_out @ down
        kernels::linear_nobias_bf16_addto(
            &mut bufs.x[..dim],
            &bufs.ffn_out[..intermediate],
            layer.down_weight_bf16,
            intermediate,
            dim,
        );
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

    // Overlap the lm_head argmax with preparation for the next decode step.
    // The next position is fixed regardless of which token wins, so we can
    // grow the KV cache and extend the RoPE tables in parallel with scoring
    // the vocabulary. The embedding lookup itself still waits for the argmax.
    let next_pos = kv_cache.len;
    let mut next_token: i32 = 0;
    std::thread::scope(|s| {
        s.spawn(|| {
            #[cfg(target_arch = "aarch64")]
            if let (Some(ref int8_data), Some(ref scales)) =
                (&decoder.lm_head_int8, &decoder.lm_head_int8_scales)
            {
                next_token =
                    kernels::argmax_matvec_int8(&bufs.x[..dim], int8_data, scales, dim, lm_out_dim)
                        as i32;
            } else {
                let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);
                next_token =
                    kernels::argmax_matvec_bf16(&bufs.x[..dim], lm_weight, dim, lm_out_dim) as i32;
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);
                next_token =
                    kernels::argmax_matvec_bf16(&bufs.x[..dim], lm_weight, dim, lm_out_dim) as i32;
            }
        });
        s.spawn(|| {
            kv_cache.grow(next_pos + 1);
            rope.ensure(next_pos + 1, head_dim, theta);
        });
    });
    next_token
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

    let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);

    // lm_head can be larger than the per-layer matrices (out_dim × dim where
    // out_dim is vocab_size or classify_num), so grow the scratch buffer first.
    bufs.ensure_scratch(out_dim * dim);

    // Project each position through lm_head: [seq_len × dim] × [out_dim × dim]^T → [seq_len × out_dim]
    let mut logits = vec![0.0f32; seq_len * out_dim];
    unsafe {
        kernels::linear_nobias_bf16_scratch(
            &mut logits,
            &x_norm,
            lm_weight,
            seq_len,
            dim,
            out_dim,
            &mut bufs.bf16_scratch,
        );
    }

    logits
}

/// Convert a token embedding from bf16 to f32.
///
/// # Safety
/// tok_emb_bf16 must point to valid memory for at least (token_id + 1) * dim bf16 values.
pub unsafe fn tok_embed_bf16_to_f32(
    dst: &mut [f32],
    tok_emb_bf16: *const u16,
    token_id: i32,
    dim: usize,
) {
    let src = unsafe { std::slice::from_raw_parts(tok_emb_bf16.add(token_id as usize * dim), dim) };
    kernels::bf16_to_f32_buf(dst, src);
}
