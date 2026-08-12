//! Qwen3 LLM decoder with GQA, KV cache, and generation.

use crate::config::*;
use crate::int8_sidecar::{self, SidecarLayout, SidecarMmap, WeightBuf};
use crate::kernels;
use crate::safetensors::MultiSafetensors;

use crate::kernels::superpage_vec;

/// Reinterpret an INT8 slice as raw bytes (for sidecar serialization).
fn i8_bytes(s: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) }
}
/// Reinterpret an f32 slice as raw bytes (for sidecar serialization).
fn f32_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// Interleave gate+up rows into the fused buffer used as the INT8 quantization
/// source (and, on arches without INT8 decode kernels, the runtime bf16
/// SwiGLU decode weight).
fn build_gate_up_fused(
    gate_bf16: *const u16,
    up_bf16: *const u16,
    inter: usize,
    hidden: usize,
) -> Vec<u16> {
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
    gate_up_fused
}

/// Quantize a BF16 weight matrix into INT8 and store the (large) INT8 buffer
/// in superpage-aligned memory.
fn quantize_to_superpage(w_bf16: *const u16, out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    // Safety: callers pass tensor pointers from the mmap'd model file, which
    // cover at least out_dim * in_dim BF16 values and outlive this call.
    let (int8, scales) = unsafe { kernels::quantize_bf16_weights_to_int8(w_bf16, out_dim, in_dim) };
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
    /// Owned interleaved bf16 fusion of gate+up. Only the non-SIMD-arch decode
    /// path consumes it at runtime (`linear_nobias_bf16_swiglu`); on
    /// aarch64/x86_64 the single-token decode path uses `gate_up_int8` instead,
    /// so the fused buffer is used only transiently at load time as the INT8
    /// quantization source and
    /// this field is left empty (R12-A) to save ~350 MB RSS. Also empty for the
    /// forced-aligner (aligner only ever runs prefill, which streams gate/up
    /// separately).
    pub gate_up_fused_bf16: Vec<u16>,
    /// INT8 quantized attention weights + per-row scales — populated only for
    /// non-aligner configs (used by aarch64/x86_64 single-token decode). Empty
    /// for aligner.
    ///
    /// Each INT8/scale buffer is a [`WeightBuf`]: either an owned superpage `Vec`
    /// (freshly quantized this run) or a slice borrowed IN PLACE from the mmap'd
    /// INT8 sidecar (R12-H1). The hot kernels see the same `*const i8`/`&[i8]`
    /// either way (via `Deref`), so the decode path is unchanged.
    pub wq_int8: WeightBuf<i8>,
    pub wq_int8_scales: WeightBuf<f32>,
    pub wk_int8: WeightBuf<i8>,
    pub wk_int8_scales: WeightBuf<f32>,
    pub wv_int8: WeightBuf<i8>,
    pub wv_int8_scales: WeightBuf<f32>,
    pub wo_int8: WeightBuf<i8>,
    pub wo_int8_scales: WeightBuf<f32>,
    /// INT8 quantized FFN weights (per-row scales) for the aarch64/x86_64
    /// single-token decode path — fused interleaved gate_up and down_proj.
    /// Quantized from the
    /// original BF16 weights. Empty for aligner.
    pub gate_up_int8: WeightBuf<i8>,
    pub gate_up_int8_scales: WeightBuf<f32>,
    pub down_int8: WeightBuf<i8>,
    pub down_int8_scales: WeightBuf<f32>,
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
    /// Owned (freshly quantized) or borrowed from the mmap'd sidecar (R12-H1).
    pub lm_head_int8: Option<WeightBuf<i8>>,
    pub lm_head_int8_scales: Option<WeightBuf<f32>>,
    /// mmap of the INT8 sidecar, kept alive so every `WeightBuf::Mapped` above
    /// stays valid for the decoder's lifetime. `None` on a cold run (owned Vecs).
    pub _int8_sidecar: Option<SidecarMmap>,
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
fn load_dec_layer(
    ms: &MultiSafetensors,
    cfg: &QwenConfig,
    i: usize,
    sidecar: Option<&SidecarMmap>,
    layout: &SidecarLayout,
) -> Option<DecLayer> {
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
    let (
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
    ) = if is_aligner {
        (
            Vec::new(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
            WeightBuf::empty(),
        )
    } else if let Some(sc) = sidecar {
        // Warm path: borrow every INT8 buffer IN PLACE from the mmap'd sidecar.
        // No quantization, no gate_up fusion on aarch64/x86_64 (where the
        // fused bf16 is unused at runtime — INT8 is the decode weight). Other
        // arches still need the fused bf16 for their SwiGLU decode path, so
        // build it there.
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        let gate_up_fused_kept: Vec<u16> = Vec::new();
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let gate_up_fused_kept: Vec<u16> = build_gate_up_fused(gate_bf16, up_bf16, inter, hidden);

        let b = |j: usize| sc.i8_buf(layout.layer_buf(i, j));
        let s = |j: usize| sc.f32_buf(layout.layer_buf(i, j));
        (
            gate_up_fused_kept,
            b(0),
            s(1),
            b(2),
            s(3),
            b(4),
            s(5),
            b(6),
            s(7),
            b(8),
            s(9),
            b(10),
            s(11),
        )
    } else {
        // Cold path: quantize into owned superpage Vecs (as before). The fused
        // gate_up buffer is the INT8 quantization source; on aarch64/x86_64 it
        // is dropped afterward (INT8 is the decode weight), on other arches it
        // is kept as the bf16 SwiGLU decode weight.
        let gate_up_fused = build_gate_up_fused(gate_bf16, up_bf16, inter, hidden);

        let (wq_int8, wq_int8_scales) = quantize_to_superpage(wq, q_dim, hidden);
        let (wk_int8, wk_int8_scales) = quantize_to_superpage(wk, kv_dim, hidden);
        let (wv_int8, wv_int8_scales) = quantize_to_superpage(wv, kv_dim, hidden);
        let (wo_int8, wo_int8_scales) = quantize_to_superpage(wo, hidden, q_dim);
        // FFN weights are INT8 quantized (per-row scales) from the original BF16
        // values — the interleaved fusion is a copy of the BF16 rows, so this is
        // single-rounded.
        let (gate_up_int8, gate_up_int8_scales) =
            quantize_to_superpage(gate_up_fused.as_ptr(), 2 * inter, hidden);
        let (down_int8, down_int8_scales) = quantize_to_superpage(down_bf16, hidden, inter);

        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        let gate_up_fused_kept = {
            drop(gate_up_fused);
            Vec::new()
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let gate_up_fused_kept = gate_up_fused;

        (
            gate_up_fused_kept,
            wq_int8.into(),
            wq_int8_scales.into(),
            wk_int8.into(),
            wk_int8_scales.into(),
            wv_int8.into(),
            wv_int8_scales.into(),
            wo_int8.into(),
            wo_int8_scales.into(),
            gate_up_int8.into(),
            gate_up_int8_scales.into(),
            down_int8.into(),
            down_int8_scales.into(),
        )
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
    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig, model_dir: &str) -> Option<Self> {
        let tok_embeddings_bf16 = load_bf16_direct(ms, "thinker.model.embed_tokens.weight")?;

        // R12-H1: try to mmap a valid INT8 sidecar next to the model. On a warm
        // run this borrows every INT8 weight IN PLACE (no quantization, no bulk
        // copy). Aligner has no INT8 weights, so it never uses a sidecar.
        let use_sidecar = !cfg.is_aligner() && int8_sidecar::enabled();
        let layout = SidecarLayout::compute(cfg);
        let sc_path = int8_sidecar::sidecar_path(model_dir);
        let sidecar = if use_sidecar {
            SidecarMmap::open_valid(&sc_path, model_dir, cfg, &layout)
        } else {
            None
        };
        let have_sidecar = sidecar.is_some();

        // Per-layer weight loading is independent and conversion-heavy
        // (bf16->f32 prefill + INT8 quantization), so load layers in parallel.
        let nlayers = cfg.dec_layers;
        let nthreads = kernels::get_num_cpus().min(nlayers).max(1);
        let chunk = nlayers.div_ceil(nthreads);
        let sidecar_ref = sidecar.as_ref();
        let layout_ref = &layout;
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
                        out.push((i, load_dec_layer(ms, cfg, i, sidecar_ref, layout_ref)?));
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
        // Warm sidecar: borrow it IN PLACE instead of quantizing.
        let (lm_int8_opt, lm_scales_opt): (Option<WeightBuf<i8>>, Option<WeightBuf<f32>>) =
            if cfg.is_aligner() {
                (None, None)
            } else if let Some(sc) = sidecar_ref {
                let idx = layout.lm_head_idx(nlayers);
                (Some(sc.i8_buf(idx)), Some(sc.f32_buf(idx + 1)))
            } else {
                let lm_weight = lm_head_bf16.unwrap_or(tok_embeddings_bf16);
                let lm_out_dim = cfg.lm_head_dim();
                let lm_in_dim = cfg.dec_hidden;
                let (lm_int8, lm_scales) = quantize_to_superpage(lm_weight, lm_out_dim, lm_in_dim);
                (Some(lm_int8.into()), Some(lm_scales.into()))
            };

        // Cold run with sidecar enabled: persist the freshly-quantized buffers so
        // the next run mmaps them. Best-effort — failure just means we re-quantize
        // next time. First run pays this one-time write on top of quantization.
        if use_sidecar && !have_sidecar {
            let mut bufs: Vec<&[u8]> = Vec::with_capacity(nlayers * 12 + 2);
            for l in &layers {
                bufs.push(i8_bytes(&l.wq_int8));
                bufs.push(f32_bytes(&l.wq_int8_scales));
                bufs.push(i8_bytes(&l.wk_int8));
                bufs.push(f32_bytes(&l.wk_int8_scales));
                bufs.push(i8_bytes(&l.wv_int8));
                bufs.push(f32_bytes(&l.wv_int8_scales));
                bufs.push(i8_bytes(&l.wo_int8));
                bufs.push(f32_bytes(&l.wo_int8_scales));
                bufs.push(i8_bytes(&l.gate_up_int8));
                bufs.push(f32_bytes(&l.gate_up_int8_scales));
                bufs.push(i8_bytes(&l.down_int8));
                bufs.push(f32_bytes(&l.down_int8_scales));
            }
            if let (Some(lm), Some(lms)) = (lm_int8_opt.as_ref(), lm_scales_opt.as_ref()) {
                bufs.push(i8_bytes(lm));
                bufs.push(f32_bytes(lms));
                int8_sidecar::write_sidecar(&sc_path, model_dir, cfg, &layout, &bufs);
            }
        }

        Some(Decoder {
            tok_embeddings_bf16,
            layers,
            norm,
            lm_head_bf16,
            lm_head_int8: lm_int8_opt,
            lm_head_int8_scales: lm_scales_opt,
            _int8_sidecar: sidecar,
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

    // R13-Android stage 1: route the prefill projections through the resident
    // INT8 weights on the no-BLAS aarch64 (Android) build. Default OFF (opt in
    // with `QWEN_ASR_INT8_PREFILL=1`) — the stage-1 WER gate regressed (see
    // docs/research/experiments.md), so the bf16→f32 GEMM stays the default path.
    // Guard against aligner configs, whose INT8 weight buffers are empty.
    // Desktop/BLAS builds never compile this branch — prefill stays on AMX f32
    // (see R12-F2).
    #[cfg(all(
        feature = "int8-prefill",
        not(feature = "blas"),
        target_arch = "aarch64"
    ))]
    let use_int8 = kernels::int8_prefill_enabled()
        && decoder.layers.first().map_or(false, |l| {
            !l.wq_int8.is_empty() && !l.gate_up_int8.is_empty() && !l.down_int8.is_empty()
        });
    #[cfg(not(all(
        feature = "int8-prefill",
        not(feature = "blas"),
        target_arch = "aarch64"
    )))]
    let use_int8 = false;

    // Per-position INT8 activation scratch, reused across all layers (largest
    // in_dim is `intermediate` for the down projection). Only allocated for the
    // INT8 prefill path.
    #[cfg(all(
        feature = "int8-prefill",
        not(feature = "blas"),
        target_arch = "aarch64"
    ))]
    let (mut xq_buf, mut xq_scales) = if use_int8 {
        let m = intermediate.max(q_dim).max(dim);
        (vec![0i8; seq_len * m], vec![0.0f32; seq_len])
    } else {
        (Vec::new(), Vec::new())
    };

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

        if use_int8 {
            #[cfg(all(
                feature = "int8-prefill",
                not(feature = "blas"),
                target_arch = "aarch64"
            ))]
            unsafe {
                kernels::quantize_rows_into(
                    &mut xq_buf[..seq_len * dim],
                    &mut xq_scales[..seq_len],
                    x_norm,
                    seq_len,
                    dim,
                );
                let xq = &xq_buf[..seq_len * dim];
                let xs = &xq_scales[..seq_len];
                kernels::int8_prefill_matvec(
                    q,
                    xq,
                    xs,
                    layer.wq_int8.as_ptr(),
                    layer.wq_int8_scales.as_ptr(),
                    dim,
                    q_dim,
                    seq_len,
                );
                kernels::int8_prefill_matvec(
                    k,
                    xq,
                    xs,
                    layer.wk_int8.as_ptr(),
                    layer.wk_int8_scales.as_ptr(),
                    dim,
                    kv_dim,
                    seq_len,
                );
                kernels::int8_prefill_matvec(
                    v,
                    xq,
                    xs,
                    layer.wv_int8.as_ptr(),
                    layer.wv_int8_scales.as_ptr(),
                    dim,
                    kv_dim,
                    seq_len,
                );
            }
        } else {
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    q,
                    x_norm,
                    layer.wq_weight_bf16,
                    seq_len,
                    dim,
                    q_dim,
                    &mut bufs.bf16_scratch,
                );
                kernels::linear_nobias_bf16_scratch(
                    k,
                    x_norm,
                    layer.wk_weight_bf16,
                    seq_len,
                    dim,
                    kv_dim,
                    &mut bufs.bf16_scratch,
                );
                kernels::linear_nobias_bf16_scratch(
                    v,
                    x_norm,
                    layer.wv_weight_bf16,
                    seq_len,
                    dim,
                    kv_dim,
                    &mut bufs.bf16_scratch,
                );
            }
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
        if use_int8 {
            #[cfg(all(
                feature = "int8-prefill",
                not(feature = "blas"),
                target_arch = "aarch64"
            ))]
            unsafe {
                kernels::quantize_rows_into(
                    &mut xq_buf[..seq_len * q_dim],
                    &mut xq_scales[..seq_len],
                    attn_out,
                    seq_len,
                    q_dim,
                );
                kernels::int8_prefill_matvec(
                    proj_out,
                    &xq_buf[..seq_len * q_dim],
                    &xq_scales[..seq_len],
                    layer.wo_int8.as_ptr(),
                    layer.wo_int8_scales.as_ptr(),
                    q_dim,
                    dim,
                    seq_len,
                );
            }
        } else {
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    proj_out,
                    attn_out,
                    layer.wo_weight_bf16,
                    seq_len,
                    q_dim,
                    dim,
                    &mut bufs.bf16_scratch,
                );
            }
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
        if use_int8 {
            // Fused INT8 gate_up + SwiGLU into `gate` directly (matches the
            // single-token `int8_swiglu_range` path — no separate up buffer).
            #[cfg(all(
                feature = "int8-prefill",
                not(feature = "blas"),
                target_arch = "aarch64"
            ))]
            unsafe {
                kernels::quantize_rows_into(
                    &mut xq_buf[..seq_len * dim],
                    &mut xq_scales[..seq_len],
                    x_norm2,
                    seq_len,
                    dim,
                );
                kernels::int8_prefill_swiglu(
                    gate,
                    &xq_buf[..seq_len * dim],
                    &xq_scales[..seq_len],
                    layer.gate_up_int8.as_ptr(),
                    layer.gate_up_int8_scales.as_ptr(),
                    dim,
                    intermediate,
                    seq_len,
                );
            }
        } else {
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    gate,
                    x_norm2,
                    layer.gate_weight_bf16,
                    seq_len,
                    dim,
                    intermediate,
                    &mut bufs.bf16_scratch,
                );
            }
            let up = &mut bufs.pref_up[..seq_len * intermediate];
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    up,
                    x_norm2,
                    layer.up_weight_bf16,
                    seq_len,
                    dim,
                    intermediate,
                    &mut bufs.bf16_scratch,
                );
            }
            // gate <- silu(gate) * up
            kernels::swiglu_separate_inplace(gate, up, seq_len, intermediate);
        }

        let ffn_out = &mut bufs.pref_ffn_out[..seq_len * dim];
        if use_int8 {
            #[cfg(all(
                feature = "int8-prefill",
                not(feature = "blas"),
                target_arch = "aarch64"
            ))]
            unsafe {
                kernels::quantize_rows_into(
                    &mut xq_buf[..seq_len * intermediate],
                    &mut xq_scales[..seq_len],
                    gate,
                    seq_len,
                    intermediate,
                );
                kernels::int8_prefill_matvec(
                    ffn_out,
                    &xq_buf[..seq_len * intermediate],
                    &xq_scales[..seq_len],
                    layer.down_int8.as_ptr(),
                    layer.down_int8_scales.as_ptr(),
                    intermediate,
                    dim,
                    seq_len,
                );
            }
        } else {
            unsafe {
                kernels::linear_nobias_bf16_scratch(
                    ffn_out,
                    gate,
                    layer.down_weight_bf16,
                    seq_len,
                    intermediate,
                    dim,
                    &mut bufs.bf16_scratch,
                );
            }
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
    let lm_out_dim = cfg.lm_head_dim();

    bufs.x[..dim].copy_from_slice(&input_embed[..dim]);

    let pos = kv_cache.len;

    if pos >= kv_cache.max_seq {
        kv_cache.grow(pos + 1024);
    }

    rope.ensure(pos + 1, head_dim, theta);
    let rope_cos = rope.cos_at(pos);
    let rope_sin = rope.sin_at(pos);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // aarch64/x86_64: run the whole 28-layer single-token loop inside ONE
    // persistent parallel region. Workers stay resident across every stage of
    // every layer, synchronized by spin barriers, instead of one thread-pool
    // dispatch/join cycle per stage (~5 per layer). Row/head partitions are
    // identical to the standalone threaded kernels, and each output element is
    // written by exactly one thread, so results are bit-identical to the
    // dispatch-per-stage path. Serial glue (norms, RoPE, KV write, activation
    // quantization) runs on tid 0 between barriers — microsecond-scale work at
    // dim 1024.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
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

        // Fused lm_head epilogue (R14-A2): with the INT8 lm_head available and
        // a multi-threaded pool, the final norm + vocabulary argmax run as the
        // last stages INSIDE the region (below), replicating
        // kernels::argmax_matvec_int8's quantize, row partition, and strict->
        // reduce exactly, so the returned token is bit-identical while the
        // token costs one pool wake/join cycle fewer and no per-token
        // thread::scope OS-thread spawns. Single-thread mode (parallel segment
        // workers force their kernels single-threaded) and the BF16 fallback
        // keep the old epilogue after the region.
        let lm_fused = match (&decoder.lm_head_int8, &decoder.lm_head_int8_scales) {
            (Some(d), Some(s)) if kernels::get_num_threads() > 1 => {
                Some((d.as_ptr() as usize, s.as_ptr() as usize))
            }
            _ => None,
        };
        let norm_w: &[f32] = &decoder.norm;
        let mut best_idx = [0usize; kernels::MAX_THREADS];
        let mut best_val = [-1e30f32; kernels::MAX_THREADS];
        let best_idx_ptr = best_idx.as_mut_ptr() as usize;
        let best_val_ptr = best_val.as_mut_ptr() as usize;
        let mut fused_token: i32 = 0;
        let fused_token_ptr = &mut fused_token as *mut i32 as usize;

        kernels::parallel_region(|barrier, tid, nt| {
            // Per-thread SwiGLU gate_up scratch, allocated once per decode step
            // instead of once per layer: removes 28 heap allocations per token
            // per thread from the fused region. Sized for this thread's max
            // row range (range_for chunks by div_ceil; +1 slack).
            let mut swiglu_scratch = vec![0.0f32; 2 * (intermediate.div_ceil(nt) + 1)];
            for (layer_idx, layer) in layers.iter().enumerate() {
                let layer_kv_off = layer_idx * n_kv * kv_head_stride;
                let k_base = unsafe { (kv_k_ptr as *const f32).add(layer_kv_off) };
                let v_base = unsafe { (kv_v_ptr as *const f32).add(layer_kv_off) };

                // Stage: pre-attention norm + input quantization (tid 0).
                if tid == 0 {
                    let x = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, dim) };
                    let x_norm =
                        unsafe { std::slice::from_raw_parts_mut(x_norm_ptr as *mut f32, dim) };
                    kernels::rms_norm(x_norm, x, &layer.input_norm, 1, dim, eps);
                    let x_int8 =
                        unsafe { std::slice::from_raw_parts_mut(x_int8_ptr as *mut i8, dim) };
                    unsafe {
                        *(scales_ptr as *mut f32) = kernels::quantize_into(x_int8, x_norm);
                    }
                }
                barrier.wait();

                // Stage: fused QKV projection, split over q|k|v output rows.
                {
                    let (s, e) = kernels::range_for(tid, nt, q_dim + 2 * kv_dim);
                    let x_scale = unsafe { *(scales_ptr as *const f32) };
                    unsafe {
                        kernels::int8_qkv_range(
                            q_ptr as *mut f32,
                            k_ptr as *mut f32,
                            v_ptr as *mut f32,
                            x_int8_ptr as *const i8,
                            x_scale,
                            layer.wq_int8.as_ptr(),
                            layer.wq_int8_scales.as_ptr(),
                            layer.wk_int8.as_ptr(),
                            layer.wk_int8_scales.as_ptr(),
                            layer.wv_int8.as_ptr(),
                            layer.wv_int8_scales.as_ptr(),
                            dim,
                            q_dim,
                            kv_dim,
                            s,
                            e,
                        );
                    }
                }
                barrier.wait();

                // Stage: q/k norms, RoPE, KV-cache write (tid 0 serial glue).
                if tid == 0 {
                    let q = unsafe { std::slice::from_raw_parts_mut(q_ptr as *mut f32, q_dim) };
                    let k = unsafe { std::slice::from_raw_parts_mut(k_ptr as *mut f32, kv_dim) };
                    kernels::rms_norm_per_head(q, &layer.q_norm_weight, 1, n_heads, head_dim, eps);
                    kernels::rms_norm_per_head(
                        k,
                        &layer.k_norm_weight,
                        1,
                        n_kv_heads,
                        head_dim,
                        eps,
                    );
                    kernels::apply_rope_neox(q, rope_cos, rope_sin, 1, n_heads, head_dim);
                    kernels::apply_rope_neox(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

                    // Head-contiguous KV write at `pos` (same as k/v_write_pos).
                    let v = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, kv_dim) };
                    for h in 0..n_kv {
                        let dst = layer_kv_off + h * kv_head_stride + pos * head_dim;
                        let src = h * head_dim;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                k.as_ptr().add(src),
                                (kv_k_ptr as *mut f32).add(dst),
                                head_dim,
                            );
                            std::ptr::copy_nonoverlapping(
                                v.as_ptr().add(src),
                                (kv_v_ptr as *mut f32).add(dst),
                                head_dim,
                            );
                        }
                    }
                }
                barrier.wait();

                // Stage: causal attention, split by KV-head group (GQA-paired).
                if let Some((h0, h1)) = kernels::attn_head_range(tid, nt, 1, n_heads, n_kv_heads) {
                    let attn_out =
                        unsafe { std::slice::from_raw_parts_mut(attn_out_ptr as *mut f32, q_dim) };
                    let q = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, q_dim) };
                    kernels::causal_attention_heads(
                        attn_out,
                        q,
                        k_base,
                        v_base,
                        kv_head_stride,
                        1,
                        total_seq,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        scale,
                        pos,
                        h0,
                        h1,
                    );
                }
                barrier.wait();

                // Stage: quantize attention output for the O-projection (tid 0).
                if tid == 0 {
                    let attn_out =
                        unsafe { std::slice::from_raw_parts(attn_out_ptr as *const f32, q_dim) };
                    let attn_int8 =
                        unsafe { std::slice::from_raw_parts_mut(attn_int8_ptr as *mut i8, q_dim) };
                    unsafe {
                        *(scales_ptr as *mut f32).add(1) =
                            kernels::quantize_into(attn_int8, attn_out);
                    }
                }
                barrier.wait();

                // Stage: O-projection with fused residual add (x += attn @ wo).
                // Each output row of x is owned by exactly one thread.
                {
                    let (s, e) = kernels::range_for(tid, nt, dim);
                    let x_scale = unsafe { *(scales_ptr as *const f32).add(1) };
                    unsafe {
                        kernels::int8_matvec_range(
                            x_ptr as *mut f32,
                            attn_int8_ptr as *const i8,
                            x_scale,
                            layer.wo_int8.as_ptr(),
                            layer.wo_int8_scales.as_ptr(),
                            Some(x_ptr as *const f32),
                            q_dim,
                            s,
                            e,
                        );
                    }
                }
                barrier.wait();

                // Stage: post-attention norm + input quantization (tid 0).
                if tid == 0 {
                    let x = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, dim) };
                    let x_norm =
                        unsafe { std::slice::from_raw_parts_mut(x_norm_ptr as *mut f32, dim) };
                    kernels::rms_norm(x_norm, x, &layer.post_attn_norm, 1, dim, eps);
                    let x_int8 =
                        unsafe { std::slice::from_raw_parts_mut(x_int8_ptr as *mut i8, dim) };
                    unsafe {
                        *(scales_ptr as *mut f32).add(2) = kernels::quantize_into(x_int8, x_norm);
                    }
                }
                barrier.wait();

                // Stage: fused gate_up + SwiGLU, split over intermediate rows.
                {
                    let (s, e) = kernels::range_for(tid, nt, intermediate);
                    let x_scale = unsafe { *(scales_ptr as *const f32).add(2) };
                    unsafe {
                        kernels::int8_swiglu_range(
                            ffn_out_ptr as *mut f32,
                            x_int8_ptr as *const i8,
                            x_scale,
                            layer.gate_up_int8.as_ptr(),
                            layer.gate_up_int8_scales.as_ptr(),
                            dim,
                            s,
                            e,
                            &mut swiglu_scratch,
                        );
                    }
                }
                barrier.wait();

                // Stage: quantize FFN activation for the down-projection (tid 0).
                if tid == 0 {
                    let ffn_out = unsafe {
                        std::slice::from_raw_parts(ffn_out_ptr as *const f32, intermediate)
                    };
                    let ffn_int8 = unsafe {
                        std::slice::from_raw_parts_mut(ffn_int8_ptr as *mut i8, intermediate)
                    };
                    unsafe {
                        *(scales_ptr as *mut f32).add(3) =
                            kernels::quantize_into(ffn_int8, ffn_out);
                    }
                }
                barrier.wait();

                // Stage: down-projection with fused residual add (x += ffn @ down).
                {
                    let (s, e) = kernels::range_for(tid, nt, dim);
                    let x_scale = unsafe { *(scales_ptr as *const f32).add(3) };
                    unsafe {
                        kernels::int8_matvec_range(
                            x_ptr as *mut f32,
                            ffn_int8_ptr as *const i8,
                            x_scale,
                            layer.down_int8.as_ptr(),
                            layer.down_int8_scales.as_ptr(),
                            Some(x_ptr as *const f32),
                            intermediate,
                            s,
                            e,
                        );
                    }
                }
                // Next layer's tid-0 norm reads every row of x; the region join
                // covers the last layer, and this barrier covers the rest.
                barrier.wait();
            }

            // Fused lm_head epilogue stages (R14-A2). Bit-identity vs the old
            // post-region epilogue: same rms_norm + copy-back, same
            // quantize_into, same row partition and strict-> reduce order as
            // kernels::argmax_matvec_int8 (chunk = lm_out_dim.div_ceil(nt)).
            if let Some((lm_int8_ptr, lm_scales_ptr)) = lm_fused {
                // Stage: final norm + copy-back + quantize (tid 0 serial glue).
                if tid == 0 {
                    let x = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, dim) };
                    let x_norm =
                        unsafe { std::slice::from_raw_parts_mut(x_norm_ptr as *mut f32, dim) };
                    kernels::rms_norm(x_norm, x, norm_w, 1, dim, eps);
                    let x_mut = unsafe { std::slice::from_raw_parts_mut(x_ptr as *mut f32, dim) };
                    x_mut.copy_from_slice(x_norm);
                    let x_int8 =
                        unsafe { std::slice::from_raw_parts_mut(x_int8_ptr as *mut i8, dim) };
                    unsafe {
                        *(scales_ptr as *mut f32) = kernels::quantize_into(x_int8, x_norm);
                    }
                }
                barrier.wait();

                // Stage: argmax over vocab rows, partitioned exactly like
                // kernels::argmax_matvec_int8 (empty ranges record (0, -1e30)).
                {
                    let chunk = lm_out_dim.div_ceil(nt);
                    let start = tid * chunk;
                    let end = (start + chunk).min(lm_out_dim);
                    let x_scale = unsafe { *(scales_ptr as *const f32) };
                    let (best, bv) = if start < end {
                        let w_scales = unsafe {
                            std::slice::from_raw_parts(lm_scales_ptr as *const f32, lm_out_dim)
                        };
                        unsafe {
                            kernels::int8_argmax_range(
                                x_int8_ptr as *const i8,
                                x_scale,
                                lm_int8_ptr as *const i8,
                                w_scales,
                                dim,
                                start,
                                end,
                            )
                        }
                    } else {
                        (0, -1e30f32)
                    };
                    unsafe {
                        *(best_idx_ptr as *mut usize).add(tid) = best;
                        *(best_val_ptr as *mut f32).add(tid) = bv;
                    }
                }
                barrier.wait();

                // Stage: strict-> reduce in tid order (tid 0), same tie-break
                // as argmax_matvec_int8's reduce.
                if tid == 0 {
                    let mut bi = unsafe { *(best_idx_ptr as *const usize) };
                    let mut bv = unsafe { *(best_val_ptr as *const f32) };
                    for i in 1..nt {
                        let v = unsafe { *(best_val_ptr as *const f32).add(i) };
                        if v > bv {
                            bv = v;
                            bi = unsafe { *(best_idx_ptr as *const usize).add(i) };
                        }
                    }
                    unsafe {
                        *(fused_token_ptr as *mut i32) = bi as i32;
                    }
                }
            }
        });

        if lm_fused.is_some() {
            // The fused epilogue produced the token inside the region; skip the
            // outer final norm/argmax and the thread::scope spawns. Ordering
            // matches the old path: len is committed first, then the next
            // step's KV capacity and RoPE tables are prepared.
            kv_cache.len = pos + 1;
            let next_pos = kv_cache.len;
            kv_cache.grow(next_pos + 1);
            rope.ensure(next_pos + 1, head_dim, theta);
            return fused_token;
        }
    }

    // Non-SIMD-arch (BF16) path: unchanged dispatch-per-stage layer loop.
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
            kernels::linear_nobias_bf16(
                &mut bufs.q[..q_dim],
                &bufs.x_norm[..dim],
                layer.wq_weight_bf16,
                1,
                dim,
                q_dim,
            );
            kernels::linear_nobias_bf16(
                &mut bufs.k[..kv_dim],
                &bufs.x_norm[..dim],
                layer.wk_weight_bf16,
                1,
                dim,
                kv_dim,
            );
            kernels::linear_nobias_bf16(
                &mut bufs.v[..kv_dim],
                &bufs.x_norm[..dim],
                layer.wv_weight_bf16,
                1,
                dim,
                kv_dim,
            );
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

    // Overlap the lm_head argmax with preparation for the next decode step.
    // The next position is fixed regardless of which token wins, so we can
    // grow the KV cache and extend the RoPE tables in parallel with scoring
    // the vocabulary. The embedding lookup itself still waits for the argmax.
    let next_pos = kv_cache.len;
    let mut next_token: i32 = 0;
    if kernels::get_num_threads() <= 1 {
        // Single-thread mode (e.g. a parallel-segment worker forcing its kernels
        // single-threaded): run the two halves inline. Spawning helper threads
        // here would dispatch to the shared global thread pool, which is unsafe
        // to drive concurrently from multiple segment workers.
        next_token = lm_head_argmax(decoder, &bufs.x[..dim], dim, lm_out_dim);
        kv_cache.grow(next_pos + 1);
        rope.ensure(next_pos + 1, head_dim, theta);
    } else {
        // Overlap the lm_head argmax with preparation for the next decode step.
        std::thread::scope(|s| {
            s.spawn(|| {
                next_token = lm_head_argmax(decoder, &bufs.x[..dim], dim, lm_out_dim);
            });
            s.spawn(|| {
                kv_cache.grow(next_pos + 1);
                rope.ensure(next_pos + 1, head_dim, theta);
            });
        });
    }
    next_token
}

/// One live decode session inside a lockstep batched step (R12-E3): the
/// per-segment mutable state plus this step's input embedding. Weights come
/// from the shared [`Decoder`].
#[cfg(target_arch = "aarch64")]
pub struct BatchedSession<'a> {
    pub kv_cache: &'a mut KvCache,
    pub rope: &'a mut RopeCache,
    pub bufs: &'a mut DecoderBuffers,
    pub input_embed: &'a [f32],
}

/// Advance `sessions.len()` decode sessions by ONE token each in lockstep
/// (R12-E3). Mirrors the fused single-token [`decoder_forward`] region stage
/// for stage, but the heavy projections use the R12-E2 batched kernels: each
/// streamed weight row is applied to every live session before moving on, so
/// decode weight DRAM traffic per token drops by the batch factor. Each
/// session's output is bit-identical to running [`decoder_forward`] on it
/// alone:
/// - heavy stages (QKV / O-proj / SwiGLU / down-proj / lm_head argmax) are
///   INT8 integer dots with a fixed per-row float combine — exact regardless
///   of row partitioning or batching (kernel exactness tests, R12-E2);
/// - glue stages (norms, RoPE, KV write, activation quantization) run the
///   same serial per-session code, striped across threads by session index;
/// - attention runs per (session, GQA group) work item through the same
///   [`kernels::causal_attention_heads`] the single-session region uses
///   (per-group output is independent of the item partition).
///
/// Sessions may sit at different positions (`kv_cache.len`); each uses its own
/// RoPE row and KV history. `out_tokens[i]` receives session `i`'s next token.
/// `sessions.len()` must be in `1..=MAX_BATCH`; `b == 1` delegates to the
/// tuned single-session path (identical math).
#[cfg(target_arch = "aarch64")]
pub fn decoder_forward_batched(
    decoder: &Decoder,
    cfg: &QwenConfig,
    sessions: &mut [BatchedSession],
    out_tokens: &mut [i32],
) {
    const MB: usize = kernels::MAX_BATCH;
    let b = sessions.len();
    assert!((1..=MB).contains(&b) && out_tokens.len() >= b);
    if b == 1 {
        let s = &mut sessions[0];
        out_tokens[0] = decoder_forward(decoder, cfg, s.kv_cache, s.rope, s.bufs, s.input_embed);
        return;
    }

    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let layers: &[DecLayer] = &decoder.layers;

    // Per-session setup + pointer tables (usize so the region closure is Sync;
    // same publication pattern as the single-session fused region).
    let mut pos = [0usize; MB];
    let mut kv_stride = [0usize; MB];
    let mut x_ptr = [0usize; MB];
    let mut x_norm_ptr = [0usize; MB];
    let mut q_ptr = [0usize; MB];
    let mut k_ptr = [0usize; MB];
    let mut v_ptr = [0usize; MB];
    let mut attn_out_ptr = [0usize; MB];
    let mut ffn_out_ptr = [0usize; MB];
    let mut x_int8_ptr = [0usize; MB];
    let mut attn_int8_ptr = [0usize; MB];
    let mut ffn_int8_ptr = [0usize; MB];
    let mut kv_k_ptr = [0usize; MB];
    let mut kv_v_ptr = [0usize; MB];
    let mut rope_cos_ptr = [0usize; MB];
    let mut rope_sin_ptr = [0usize; MB];

    for (bi, s) in sessions.iter_mut().enumerate() {
        s.bufs.x[..dim].copy_from_slice(&s.input_embed[..dim]);
        let p = s.kv_cache.len;
        if p >= s.kv_cache.max_seq {
            s.kv_cache.grow(p + 1024);
        }
        s.rope.ensure(p + 1, head_dim, theta);
        pos[bi] = p;
        kv_stride[bi] = s.kv_cache.head_stride();
        x_ptr[bi] = s.bufs.x.as_mut_ptr() as usize;
        x_norm_ptr[bi] = s.bufs.x_norm.as_mut_ptr() as usize;
        q_ptr[bi] = s.bufs.q.as_mut_ptr() as usize;
        k_ptr[bi] = s.bufs.k.as_mut_ptr() as usize;
        v_ptr[bi] = s.bufs.v.as_mut_ptr() as usize;
        attn_out_ptr[bi] = s.bufs.attn_out.as_mut_ptr() as usize;
        ffn_out_ptr[bi] = s.bufs.ffn_out.as_mut_ptr() as usize;
        x_int8_ptr[bi] = s.bufs.x_int8.as_mut_ptr() as usize;
        attn_int8_ptr[bi] = s.bufs.attn_int8.as_mut_ptr() as usize;
        ffn_int8_ptr[bi] = s.bufs.ffn_int8.as_mut_ptr() as usize;
        kv_k_ptr[bi] = s.kv_cache.k.as_mut_ptr() as usize;
        kv_v_ptr[bi] = s.kv_cache.v.as_mut_ptr() as usize;
        rope_cos_ptr[bi] = s.rope.cos_at(p).as_ptr() as usize;
        rope_sin_ptr[bi] = s.rope.sin_at(p).as_ptr() as usize;
    }

    // Shared activation scales, published across threads only via barriers.
    // scales[4*bi + 0]: QKV input, +1: O-proj input, +2: SwiGLU input,
    // +3: down-proj input (per session).
    let mut scales = [0.0f32; 4 * MB];
    let scales_addr = scales.as_mut_ptr() as usize;

    kernels::parallel_region(|barrier, tid, nt| {
        // Per-stage pointer/scale table builders (stack arrays, per thread).
        let mut_f32 = |tab: &[usize; MB]| {
            let mut a = [std::ptr::null_mut::<f32>(); MB];
            for (dst, &src) in a[..b].iter_mut().zip(tab[..b].iter()) {
                *dst = src as *mut f32;
            }
            a
        };
        let const_f32 = |tab: &[usize; MB]| {
            let mut a = [std::ptr::null::<f32>(); MB];
            for (dst, &src) in a[..b].iter_mut().zip(tab[..b].iter()) {
                *dst = src as *const f32;
            }
            a
        };
        let const_i8 = |tab: &[usize; MB]| {
            let mut a = [std::ptr::null::<i8>(); MB];
            for (dst, &src) in a[..b].iter_mut().zip(tab[..b].iter()) {
                *dst = src as *const i8;
            }
            a
        };
        let scales_at = |slot: usize| {
            let mut a = [0.0f32; MB];
            for (bi, v) in a[..b].iter_mut().enumerate() {
                *v = unsafe { *(scales_addr as *const f32).add(4 * bi + slot) };
            }
            a
        };

        for (layer_idx, layer) in layers.iter().enumerate() {
            // Stage: pre-attention norm + input quantization (striped).
            let mut bi = tid;
            while bi < b {
                unsafe {
                    let x = std::slice::from_raw_parts(x_ptr[bi] as *const f32, dim);
                    let x_norm = std::slice::from_raw_parts_mut(x_norm_ptr[bi] as *mut f32, dim);
                    kernels::rms_norm(x_norm, x, &layer.input_norm, 1, dim, eps);
                    let x_int8 = std::slice::from_raw_parts_mut(x_int8_ptr[bi] as *mut i8, dim);
                    *(scales_addr as *mut f32).add(4 * bi) = kernels::quantize_into(x_int8, x_norm);
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: fused QKV projection, batched (rows outer, sessions inner).
            {
                let (s, e) = kernels::range_for(tid, nt, q_dim + 2 * kv_dim);
                let qp = mut_f32(&q_ptr);
                let kp = mut_f32(&k_ptr);
                let vp = mut_f32(&v_ptr);
                let xi = const_i8(&x_int8_ptr);
                let xs = scales_at(0);
                unsafe {
                    kernels::int8_qkv_range_batched(
                        b,
                        &qp[..b],
                        &kp[..b],
                        &vp[..b],
                        &xi[..b],
                        &xs[..b],
                        layer.wq_int8.as_ptr(),
                        layer.wq_int8_scales.as_ptr(),
                        layer.wk_int8.as_ptr(),
                        layer.wk_int8_scales.as_ptr(),
                        layer.wv_int8.as_ptr(),
                        layer.wv_int8_scales.as_ptr(),
                        dim,
                        q_dim,
                        kv_dim,
                        s,
                        e,
                    );
                }
            }
            barrier.wait();

            // Stage: q/k norms, RoPE, KV-cache write (striped serial glue).
            let mut bi = tid;
            while bi < b {
                unsafe {
                    let q = std::slice::from_raw_parts_mut(q_ptr[bi] as *mut f32, q_dim);
                    let k = std::slice::from_raw_parts_mut(k_ptr[bi] as *mut f32, kv_dim);
                    kernels::rms_norm_per_head(q, &layer.q_norm_weight, 1, n_heads, head_dim, eps);
                    kernels::rms_norm_per_head(
                        k,
                        &layer.k_norm_weight,
                        1,
                        n_kv_heads,
                        head_dim,
                        eps,
                    );
                    let cos = std::slice::from_raw_parts(rope_cos_ptr[bi] as *const f32, head_dim);
                    let sin = std::slice::from_raw_parts(rope_sin_ptr[bi] as *const f32, head_dim);
                    kernels::apply_rope_neox(q, cos, sin, 1, n_heads, head_dim);
                    kernels::apply_rope_neox(k, cos, sin, 1, n_kv_heads, head_dim);

                    // Head-contiguous KV write at pos[bi] (same as k/v_write_pos).
                    let v = std::slice::from_raw_parts(v_ptr[bi] as *const f32, kv_dim);
                    let layer_kv_off = layer_idx * n_kv_heads * kv_stride[bi];
                    for h in 0..n_kv_heads {
                        let dst = layer_kv_off + h * kv_stride[bi] + pos[bi] * head_dim;
                        let src = h * head_dim;
                        std::ptr::copy_nonoverlapping(
                            k.as_ptr().add(src),
                            (kv_k_ptr[bi] as *mut f32).add(dst),
                            head_dim,
                        );
                        std::ptr::copy_nonoverlapping(
                            v.as_ptr().add(src),
                            (kv_v_ptr[bi] as *mut f32).add(dst),
                            head_dim,
                        );
                    }
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: causal attention, striped over (session, GQA group) items —
            // sessions have different sequence lengths, so each item carries its
            // session's pos / total_seq / KV base.
            {
                let hpk = n_heads / n_kv_heads;
                let n_items = b * n_kv_heads;
                let mut item = tid;
                while item < n_items {
                    let bi = item / n_kv_heads;
                    let g = item % n_kv_heads;
                    let layer_kv_off = layer_idx * n_kv_heads * kv_stride[bi];
                    unsafe {
                        let attn_out =
                            std::slice::from_raw_parts_mut(attn_out_ptr[bi] as *mut f32, q_dim);
                        let q = std::slice::from_raw_parts(q_ptr[bi] as *const f32, q_dim);
                        let k_base = (kv_k_ptr[bi] as *const f32).add(layer_kv_off);
                        let v_base = (kv_v_ptr[bi] as *const f32).add(layer_kv_off);
                        kernels::causal_attention_heads(
                            attn_out,
                            q,
                            k_base,
                            v_base,
                            kv_stride[bi],
                            1,
                            pos[bi] + 1,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            scale,
                            pos[bi],
                            g * hpk,
                            (g + 1) * hpk,
                        );
                    }
                    item += nt;
                }
            }
            barrier.wait();

            // Stage: quantize attention output for the O-projection (striped).
            let mut bi = tid;
            while bi < b {
                unsafe {
                    let attn_out =
                        std::slice::from_raw_parts(attn_out_ptr[bi] as *const f32, q_dim);
                    let attn_int8 =
                        std::slice::from_raw_parts_mut(attn_int8_ptr[bi] as *mut i8, q_dim);
                    *(scales_addr as *mut f32).add(4 * bi + 1) =
                        kernels::quantize_into(attn_int8, attn_out);
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: O-projection with fused residual add (x += attn @ wo), batched.
            {
                let (s, e) = kernels::range_for(tid, nt, dim);
                let yp = mut_f32(&x_ptr);
                let bp = const_f32(&x_ptr);
                let ai = const_i8(&attn_int8_ptr);
                let xs = scales_at(1);
                unsafe {
                    kernels::int8_matvec_range_batched(
                        b,
                        &yp[..b],
                        &ai[..b],
                        &xs[..b],
                        layer.wo_int8.as_ptr(),
                        layer.wo_int8_scales.as_ptr(),
                        Some(&bp[..b]),
                        q_dim,
                        s,
                        e,
                    );
                }
            }
            barrier.wait();

            // Stage: post-attention norm + input quantization (striped).
            let mut bi = tid;
            while bi < b {
                unsafe {
                    let x = std::slice::from_raw_parts(x_ptr[bi] as *const f32, dim);
                    let x_norm = std::slice::from_raw_parts_mut(x_norm_ptr[bi] as *mut f32, dim);
                    kernels::rms_norm(x_norm, x, &layer.post_attn_norm, 1, dim, eps);
                    let x_int8 = std::slice::from_raw_parts_mut(x_int8_ptr[bi] as *mut i8, dim);
                    *(scales_addr as *mut f32).add(4 * bi + 2) =
                        kernels::quantize_into(x_int8, x_norm);
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: fused gate_up + SwiGLU, batched over intermediate rows.
            {
                let (s, e) = kernels::range_for(tid, nt, intermediate);
                let fp = mut_f32(&ffn_out_ptr);
                let xi = const_i8(&x_int8_ptr);
                let xs = scales_at(2);
                unsafe {
                    kernels::int8_swiglu_range_batched(
                        b,
                        &fp[..b],
                        &xi[..b],
                        &xs[..b],
                        layer.gate_up_int8.as_ptr(),
                        layer.gate_up_int8_scales.as_ptr(),
                        dim,
                        s,
                        e,
                    );
                }
            }
            barrier.wait();

            // Stage: quantize FFN activation for the down-projection (striped).
            let mut bi = tid;
            while bi < b {
                unsafe {
                    let ffn_out =
                        std::slice::from_raw_parts(ffn_out_ptr[bi] as *const f32, intermediate);
                    let ffn_int8 =
                        std::slice::from_raw_parts_mut(ffn_int8_ptr[bi] as *mut i8, intermediate);
                    *(scales_addr as *mut f32).add(4 * bi + 3) =
                        kernels::quantize_into(ffn_int8, ffn_out);
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: down-projection with fused residual add (x += ffn @ down), batched.
            {
                let (s, e) = kernels::range_for(tid, nt, dim);
                let yp = mut_f32(&x_ptr);
                let bp = const_f32(&x_ptr);
                let fi = const_i8(&ffn_int8_ptr);
                let xs = scales_at(3);
                unsafe {
                    kernels::int8_matvec_range_batched(
                        b,
                        &yp[..b],
                        &fi[..b],
                        &xs[..b],
                        layer.down_int8.as_ptr(),
                        layer.down_int8_scales.as_ptr(),
                        Some(&bp[..b]),
                        intermediate,
                        s,
                        e,
                    );
                }
            }
            barrier.wait();
        }

        // Stage: final RMS norm per session (striped) — mirrors the
        // single-session epilogue (x = rms_norm(x, decoder.norm)).
        let mut bi = tid;
        while bi < b {
            unsafe {
                let x = std::slice::from_raw_parts(x_ptr[bi] as *const f32, dim);
                let x_norm = std::slice::from_raw_parts_mut(x_norm_ptr[bi] as *mut f32, dim);
                kernels::rms_norm(x_norm, x, &decoder.norm, 1, dim, eps);
                let x_mut = std::slice::from_raw_parts_mut(x_ptr[bi] as *mut f32, dim);
                x_mut.copy_from_slice(x_norm);
            }
            bi += nt;
        }
    });

    for (bi, s) in sessions.iter_mut().enumerate() {
        s.kv_cache.len = pos[bi] + 1;
    }

    // lm_head argmax: stream the head weights once for ALL live sessions.
    // Same index-stable tie-break as the single-session path (R12-E2).
    let lm_out_dim = cfg.lm_head_dim();
    if let (Some(int8_data), Some(lm_scales)) =
        (&decoder.lm_head_int8, &decoder.lm_head_int8_scales)
    {
        let xs: Vec<&[f32]> = sessions.iter().map(|s| &s.bufs.x[..dim]).collect();
        let best = kernels::argmax_matvec_int8_batched(&xs, int8_data, lm_scales, dim, lm_out_dim);
        for (dst, &t) in out_tokens[..b].iter_mut().zip(best.iter()) {
            *dst = t as i32;
        }
    } else {
        for (bi, s) in sessions.iter().enumerate() {
            out_tokens[bi] = lm_head_argmax(decoder, &s.bufs.x[..dim], dim, lm_out_dim);
        }
    }

    // Speculative bookkeeping for the next step (same as the single-session tail).
    for s in sessions.iter_mut() {
        let next_pos = s.kv_cache.len;
        s.kv_cache.grow(next_pos + 1);
        s.rope.ensure(next_pos + 1, head_dim, theta);
    }
}

/// Lazily-grown pool of decode-only [`DecoderBuffers`] for the multi-token
/// verifier (R13-A). A verify step needs `k` independent buffer sets (one per
/// candidate position); each set touches only the small single-token decode
/// fields (`x`, `q`, `k`, `v`, `attn_out`, `ffn_out`, the INT8 scratch — a few
/// hundred KB total), leaving the large prefill vectors empty. The pool grows
/// on demand and is reused across verify steps so steady-state verification
/// never allocates. Intended to live on the future caller (`QwenCtx`, R13-B).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[derive(Default)]
pub struct VerifyBufferPool {
    pool: Vec<DecoderBuffers>,
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl VerifyBufferPool {
    pub fn new() -> Self {
        VerifyBufferPool { pool: Vec::new() }
    }

    /// Ensure the pool holds at least `k` buffer sets and return a mutable slice
    /// of the first `k`. Newly created sets allocate only the decode-only fields
    /// (prefill vecs stay empty); existing sets are reused untouched.
    pub fn ensure(&mut self, k: usize, cfg: &QwenConfig) -> &mut [DecoderBuffers] {
        while self.pool.len() < k {
            self.pool.push(DecoderBuffers::new(cfg));
        }
        &mut self.pool[..k]
    }
}

/// Greedy acceptance rule for a multi-token verify step (R13-A), pure over the
/// verifier output. `drafts` are the `k - 1` speculative tokens `d_1..d_{k-1}`
/// (the caller's pending token `t0` is *not* included); `out_argmax` are the `k`
/// greedy argmaxes returned by [`decoder_forward_verify`], where `out_argmax[i]`
/// is the true next token after committing the input at verify-position `i`.
///
/// Returns `accepted` = the longest run `a` (`0 <= a <= k - 1`) such that
/// `drafts[i - 1] == out_argmax[i - 1]` for every `1 <= i <= a`. The caller then
/// commits `t0, d_1..d_a` plus the free next token `out_argmax[a]`, and rolls
/// the shared cache back to `kv_cache.len = base + a + 1`. `out_argmax[a]`
/// becomes the next step's `t0`.
pub fn verify_accepted_len(drafts: &[i32], out_argmax: &[i32]) -> usize {
    let mut a = 0usize;
    while a < drafts.len() && a < out_argmax.len() && drafts[a] == out_argmax[a] {
        a += 1;
    }
    a
}

/// Multi-token greedy verifier core (R13-A): advance ONE decode session by `k`
/// consecutive positions `base..base+k` (`base = kv_cache.len` on entry) in a
/// single fused parallel region, streaming each decode weight row once for all
/// `k` positions (the R12-E2 batched INT8 kernels). All `k` lanes share the one
/// `kv_cache` / `rope`; lane `i` sits at position `base + i`.
///
/// `input_embeds` is `k × dim`: the embeddings of `[t0, d_1, .., d_{k-1}]` where
/// `t0` is the caller's pending token and `d_1..d_{k-1}` are draft tokens. On
/// return `out_argmax[i]` is the greedy argmax after position `base + i`, and
/// `kv_cache.len = base + k` (all `k` K/V rows written). The caller applies the
/// acceptance rule ([`verify_accepted_len`]) and rolls `kv_cache.len` back to
/// the committed length; stale K/V rows beyond it are overwritten on the next
/// step (no cleanup needed).
///
/// Exactness: each lane's math is bit-identical to running [`decoder_forward`]
/// on that position sequentially. Heavy stages are INT8 integer dots with a
/// fixed per-row float combine (order-exact regardless of batching, R12-E2);
/// glue stages (norms, RoPE, KV write, quantization) run the same serial
/// per-position code; attention runs the same [`kernels::causal_attention_heads`]
/// per (lane, GQA group). The KV-write stage barrier precedes the attention
/// stage, so lane `i` reads the rows written by lanes `j < i` at positions
/// `base..base+i` — the same causal structure as prefill. Because lanes write
/// distinct `pos` rows, the shared cache has no cross-lane write conflicts.
///
/// `k` must be in `1..=MAX_BATCH`; `k == 1` delegates to [`decoder_forward`] to
/// keep the tuned single-token epilogue.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub fn decoder_forward_verify(
    decoder: &Decoder,
    cfg: &QwenConfig,
    kv_cache: &mut KvCache,
    rope: &mut RopeCache,
    bufs: &mut [DecoderBuffers],
    input_embeds: &[f32],
    k: usize,
    out_argmax: &mut [i32],
) {
    const MB: usize = kernels::MAX_BATCH;
    let dim = cfg.dec_hidden;
    assert!((1..=MB).contains(&k), "verify k must be in 1..=MAX_BATCH");
    assert!(bufs.len() >= k, "need >= k buffer sets");
    assert!(input_embeds.len() >= k * dim, "need k x dim input embeds");
    assert!(out_argmax.len() >= k, "need k output slots");

    if k == 1 {
        out_argmax[0] = decoder_forward(
            decoder,
            cfg,
            kv_cache,
            rope,
            &mut bufs[0],
            &input_embeds[..dim],
        );
        return;
    }

    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let layers: &[DecLayer] = &decoder.layers;

    let base = kv_cache.len;

    // Grow/ensure ONCE up front, BEFORE capturing raw pointers: the k lanes
    // share ONE cache, so it must not reallocate after the pointers are taken.
    // Positions base..base+k-1 need max_seq >= base + k and RoPE cap >= base + k.
    kv_cache.grow(base + k);
    rope.ensure(base + k, head_dim, theta);

    // Shared cache pointers/stride — identical for every lane (all alias the
    // same buffer); lanes differ only by their KV *position* (base + bi).
    let kv_stride_val = kv_cache.head_stride();
    let kv_k_addr = kv_cache.k.as_mut_ptr() as usize;
    let kv_v_addr = kv_cache.v.as_mut_ptr() as usize;

    // Per-lane setup + pointer tables (usize so the region closure is Sync;
    // same publication pattern as the single-session fused region).
    let mut pos = [0usize; MB];
    let mut kv_stride = [0usize; MB];
    let mut x_ptr = [0usize; MB];
    let mut x_norm_ptr = [0usize; MB];
    let mut q_ptr = [0usize; MB];
    let mut k_ptr = [0usize; MB];
    let mut v_ptr = [0usize; MB];
    let mut attn_out_ptr = [0usize; MB];
    let mut ffn_out_ptr = [0usize; MB];
    let mut x_int8_ptr = [0usize; MB];
    let mut attn_int8_ptr = [0usize; MB];
    let mut ffn_int8_ptr = [0usize; MB];
    let mut kv_k_ptr = [0usize; MB];
    let mut kv_v_ptr = [0usize; MB];
    let mut rope_cos_ptr = [0usize; MB];
    let mut rope_sin_ptr = [0usize; MB];

    for (bi, buf) in bufs[..k].iter_mut().enumerate() {
        buf.x[..dim].copy_from_slice(&input_embeds[bi * dim..(bi + 1) * dim]);
        pos[bi] = base + bi;
        kv_stride[bi] = kv_stride_val;
        x_ptr[bi] = buf.x.as_mut_ptr() as usize;
        x_norm_ptr[bi] = buf.x_norm.as_mut_ptr() as usize;
        q_ptr[bi] = buf.q.as_mut_ptr() as usize;
        k_ptr[bi] = buf.k.as_mut_ptr() as usize;
        v_ptr[bi] = buf.v.as_mut_ptr() as usize;
        attn_out_ptr[bi] = buf.attn_out.as_mut_ptr() as usize;
        ffn_out_ptr[bi] = buf.ffn_out.as_mut_ptr() as usize;
        x_int8_ptr[bi] = buf.x_int8.as_mut_ptr() as usize;
        attn_int8_ptr[bi] = buf.attn_int8.as_mut_ptr() as usize;
        ffn_int8_ptr[bi] = buf.ffn_int8.as_mut_ptr() as usize;
        kv_k_ptr[bi] = kv_k_addr;
        kv_v_ptr[bi] = kv_v_addr;
        rope_cos_ptr[bi] = rope.cos_at(base + bi).as_ptr() as usize;
        rope_sin_ptr[bi] = rope.sin_at(base + bi).as_ptr() as usize;
    }

    // Shared activation scales, published across threads only via barriers.
    // scales[4*bi + 0]: QKV input, +1: O-proj input, +2: SwiGLU input,
    // +3: down-proj input (per lane).
    let mut scales = [0.0f32; 4 * MB];
    let scales_addr = scales.as_mut_ptr() as usize;

    kernels::parallel_region(|barrier, tid, nt| {
        // Per-stage pointer/scale table builders (stack arrays, per thread).
        let mut_f32 = |tab: &[usize; MB]| {
            let mut a = [std::ptr::null_mut::<f32>(); MB];
            for (dst, &src) in a[..k].iter_mut().zip(tab[..k].iter()) {
                *dst = src as *mut f32;
            }
            a
        };
        let const_f32 = |tab: &[usize; MB]| {
            let mut a = [std::ptr::null::<f32>(); MB];
            for (dst, &src) in a[..k].iter_mut().zip(tab[..k].iter()) {
                *dst = src as *const f32;
            }
            a
        };
        let const_i8 = |tab: &[usize; MB]| {
            let mut a = [std::ptr::null::<i8>(); MB];
            for (dst, &src) in a[..k].iter_mut().zip(tab[..k].iter()) {
                *dst = src as *const i8;
            }
            a
        };
        let scales_at = |slot: usize| {
            let mut a = [0.0f32; MB];
            for (bi, v) in a[..k].iter_mut().enumerate() {
                *v = unsafe { *(scales_addr as *const f32).add(4 * bi + slot) };
            }
            a
        };

        for (layer_idx, layer) in layers.iter().enumerate() {
            // Stage: pre-attention norm + input quantization (striped).
            let mut bi = tid;
            while bi < k {
                unsafe {
                    let x = std::slice::from_raw_parts(x_ptr[bi] as *const f32, dim);
                    let x_norm = std::slice::from_raw_parts_mut(x_norm_ptr[bi] as *mut f32, dim);
                    kernels::rms_norm(x_norm, x, &layer.input_norm, 1, dim, eps);
                    let x_int8 = std::slice::from_raw_parts_mut(x_int8_ptr[bi] as *mut i8, dim);
                    *(scales_addr as *mut f32).add(4 * bi) = kernels::quantize_into(x_int8, x_norm);
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: fused QKV projection, batched (rows outer, lanes inner).
            {
                let (s, e) = kernels::range_for(tid, nt, q_dim + 2 * kv_dim);
                let qp = mut_f32(&q_ptr);
                let kp = mut_f32(&k_ptr);
                let vp = mut_f32(&v_ptr);
                let xi = const_i8(&x_int8_ptr);
                let xs = scales_at(0);
                unsafe {
                    kernels::int8_qkv_range_batched(
                        k,
                        &qp[..k],
                        &kp[..k],
                        &vp[..k],
                        &xi[..k],
                        &xs[..k],
                        layer.wq_int8.as_ptr(),
                        layer.wq_int8_scales.as_ptr(),
                        layer.wk_int8.as_ptr(),
                        layer.wk_int8_scales.as_ptr(),
                        layer.wv_int8.as_ptr(),
                        layer.wv_int8_scales.as_ptr(),
                        dim,
                        q_dim,
                        kv_dim,
                        s,
                        e,
                    );
                }
            }
            barrier.wait();

            // Stage: q/k norms, RoPE, KV-cache write (striped serial glue).
            // Each lane writes its own distinct pos row of the shared cache.
            let mut bi = tid;
            while bi < k {
                unsafe {
                    let q = std::slice::from_raw_parts_mut(q_ptr[bi] as *mut f32, q_dim);
                    let kk = std::slice::from_raw_parts_mut(k_ptr[bi] as *mut f32, kv_dim);
                    kernels::rms_norm_per_head(q, &layer.q_norm_weight, 1, n_heads, head_dim, eps);
                    kernels::rms_norm_per_head(
                        kk,
                        &layer.k_norm_weight,
                        1,
                        n_kv_heads,
                        head_dim,
                        eps,
                    );
                    let cos = std::slice::from_raw_parts(rope_cos_ptr[bi] as *const f32, head_dim);
                    let sin = std::slice::from_raw_parts(rope_sin_ptr[bi] as *const f32, head_dim);
                    kernels::apply_rope_neox(q, cos, sin, 1, n_heads, head_dim);
                    kernels::apply_rope_neox(kk, cos, sin, 1, n_kv_heads, head_dim);

                    // Head-contiguous KV write at pos[bi] (same as k/v_write_pos).
                    let v = std::slice::from_raw_parts(v_ptr[bi] as *const f32, kv_dim);
                    let layer_kv_off = layer_idx * n_kv_heads * kv_stride[bi];
                    for h in 0..n_kv_heads {
                        let dst = layer_kv_off + h * kv_stride[bi] + pos[bi] * head_dim;
                        let src = h * head_dim;
                        std::ptr::copy_nonoverlapping(
                            kk.as_ptr().add(src),
                            (kv_k_ptr[bi] as *mut f32).add(dst),
                            head_dim,
                        );
                        std::ptr::copy_nonoverlapping(
                            v.as_ptr().add(src),
                            (kv_v_ptr[bi] as *mut f32).add(dst),
                            head_dim,
                        );
                    }
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: causal attention, striped over (lane, GQA group) items —
            // lanes sit at consecutive positions, so each item carries its
            // lane's pos / total_seq. All lanes share the one KV base; the write
            // stage above (before this barrier) published rows base..base+k-1,
            // and lane bi's total_seq = pos[bi] + 1 keeps it causal (reads only
            // rows written by lanes j <= bi).
            {
                let hpk = n_heads / n_kv_heads;
                let n_items = k * n_kv_heads;
                let mut item = tid;
                while item < n_items {
                    let bi = item / n_kv_heads;
                    let g = item % n_kv_heads;
                    let layer_kv_off = layer_idx * n_kv_heads * kv_stride[bi];
                    unsafe {
                        let attn_out =
                            std::slice::from_raw_parts_mut(attn_out_ptr[bi] as *mut f32, q_dim);
                        let q = std::slice::from_raw_parts(q_ptr[bi] as *const f32, q_dim);
                        let k_base = (kv_k_ptr[bi] as *const f32).add(layer_kv_off);
                        let v_base = (kv_v_ptr[bi] as *const f32).add(layer_kv_off);
                        kernels::causal_attention_heads(
                            attn_out,
                            q,
                            k_base,
                            v_base,
                            kv_stride[bi],
                            1,
                            pos[bi] + 1,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            scale,
                            pos[bi],
                            g * hpk,
                            (g + 1) * hpk,
                        );
                    }
                    item += nt;
                }
            }
            barrier.wait();

            // Stage: quantize attention output for the O-projection (striped).
            let mut bi = tid;
            while bi < k {
                unsafe {
                    let attn_out =
                        std::slice::from_raw_parts(attn_out_ptr[bi] as *const f32, q_dim);
                    let attn_int8 =
                        std::slice::from_raw_parts_mut(attn_int8_ptr[bi] as *mut i8, q_dim);
                    *(scales_addr as *mut f32).add(4 * bi + 1) =
                        kernels::quantize_into(attn_int8, attn_out);
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: O-projection with fused residual add (x += attn @ wo), batched.
            {
                let (s, e) = kernels::range_for(tid, nt, dim);
                let yp = mut_f32(&x_ptr);
                let bp = const_f32(&x_ptr);
                let ai = const_i8(&attn_int8_ptr);
                let xs = scales_at(1);
                unsafe {
                    kernels::int8_matvec_range_batched(
                        k,
                        &yp[..k],
                        &ai[..k],
                        &xs[..k],
                        layer.wo_int8.as_ptr(),
                        layer.wo_int8_scales.as_ptr(),
                        Some(&bp[..k]),
                        q_dim,
                        s,
                        e,
                    );
                }
            }
            barrier.wait();

            // Stage: post-attention norm + input quantization (striped).
            let mut bi = tid;
            while bi < k {
                unsafe {
                    let x = std::slice::from_raw_parts(x_ptr[bi] as *const f32, dim);
                    let x_norm = std::slice::from_raw_parts_mut(x_norm_ptr[bi] as *mut f32, dim);
                    kernels::rms_norm(x_norm, x, &layer.post_attn_norm, 1, dim, eps);
                    let x_int8 = std::slice::from_raw_parts_mut(x_int8_ptr[bi] as *mut i8, dim);
                    *(scales_addr as *mut f32).add(4 * bi + 2) =
                        kernels::quantize_into(x_int8, x_norm);
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: fused gate_up + SwiGLU, batched over intermediate rows.
            {
                let (s, e) = kernels::range_for(tid, nt, intermediate);
                let fp = mut_f32(&ffn_out_ptr);
                let xi = const_i8(&x_int8_ptr);
                let xs = scales_at(2);
                unsafe {
                    kernels::int8_swiglu_range_batched(
                        k,
                        &fp[..k],
                        &xi[..k],
                        &xs[..k],
                        layer.gate_up_int8.as_ptr(),
                        layer.gate_up_int8_scales.as_ptr(),
                        dim,
                        s,
                        e,
                    );
                }
            }
            barrier.wait();

            // Stage: quantize FFN activation for the down-projection (striped).
            let mut bi = tid;
            while bi < k {
                unsafe {
                    let ffn_out =
                        std::slice::from_raw_parts(ffn_out_ptr[bi] as *const f32, intermediate);
                    let ffn_int8 =
                        std::slice::from_raw_parts_mut(ffn_int8_ptr[bi] as *mut i8, intermediate);
                    *(scales_addr as *mut f32).add(4 * bi + 3) =
                        kernels::quantize_into(ffn_int8, ffn_out);
                }
                bi += nt;
            }
            barrier.wait();

            // Stage: down-projection with fused residual add (x += ffn @ down), batched.
            {
                let (s, e) = kernels::range_for(tid, nt, dim);
                let yp = mut_f32(&x_ptr);
                let bp = const_f32(&x_ptr);
                let fi = const_i8(&ffn_int8_ptr);
                let xs = scales_at(3);
                unsafe {
                    kernels::int8_matvec_range_batched(
                        k,
                        &yp[..k],
                        &fi[..k],
                        &xs[..k],
                        layer.down_int8.as_ptr(),
                        layer.down_int8_scales.as_ptr(),
                        Some(&bp[..k]),
                        intermediate,
                        s,
                        e,
                    );
                }
            }
            barrier.wait();
        }

        // Stage: final RMS norm per lane (striped) — mirrors the single-session
        // epilogue (x = rms_norm(x, decoder.norm)).
        let mut bi = tid;
        while bi < k {
            unsafe {
                let x = std::slice::from_raw_parts(x_ptr[bi] as *const f32, dim);
                let x_norm = std::slice::from_raw_parts_mut(x_norm_ptr[bi] as *mut f32, dim);
                kernels::rms_norm(x_norm, x, &decoder.norm, 1, dim, eps);
                let x_mut = std::slice::from_raw_parts_mut(x_ptr[bi] as *mut f32, dim);
                x_mut.copy_from_slice(x_norm);
            }
            bi += nt;
        }
    });

    // All k K/V rows written: the shared cache now spans base + k positions.
    kv_cache.len = base + k;

    // lm_head argmax: stream the head weights once for ALL k final hidden
    // states. Same index-stable tie-break as the single-session path (R12-E2).
    let lm_out_dim = cfg.lm_head_dim();
    if let (Some(int8_data), Some(lm_scales)) =
        (&decoder.lm_head_int8, &decoder.lm_head_int8_scales)
    {
        let xs: Vec<&[f32]> = bufs[..k].iter().map(|b| &b.x[..dim]).collect();
        let best = kernels::argmax_matvec_int8_batched(&xs, int8_data, lm_scales, dim, lm_out_dim);
        for (dst, &t) in out_argmax[..k].iter_mut().zip(best.iter()) {
            *dst = t as i32;
        }
    } else {
        for (bi, buf) in bufs[..k].iter().enumerate() {
            out_argmax[bi] = lm_head_argmax(decoder, &buf.x[..dim], dim, lm_out_dim);
        }
    }
}

/// Score the vocabulary and return the argmax token id. Uses the INT8 lm_head
/// on aarch64/x86_64 when available, else the BF16 path. Bit-identical
/// regardless of the effective thread count (the underlying matvec argmax is row-partitioned
/// with index-stable tie-breaking).
#[inline]
fn lm_head_argmax(decoder: &Decoder, x: &[f32], dim: usize, lm_out_dim: usize) -> i32 {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if let (Some(ref int8_data), Some(ref scales)) =
        (&decoder.lm_head_int8, &decoder.lm_head_int8_scales)
    {
        return kernels::argmax_matvec_int8(x, int8_data, scales, dim, lm_out_dim) as i32;
    }
    let lm_weight = decoder.lm_head_bf16.unwrap_or(decoder.tok_embeddings_bf16);
    kernels::argmax_matvec_bf16(x, lm_weight, dim, lm_out_dim) as i32
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
