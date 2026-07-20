//! Audio encoder: Conv2D stem + windowed transformer + projection cascade.

use crate::config::*;
use crate::kernels;
use crate::safetensors::{Dtype, MultiSafetensors};

/// Encoder transformer layer. Large weight matrices are prepacked from BF16 to
/// owned f32 buffers once at load (superpage-aligned) so each forward GEMM
/// consumes f32 directly — no per-call bf16→f32 conversion through scratch.
/// The bf16→f32 widening is exact, so outputs are bit-identical to the old
/// on-the-fly path. Small per-output biases / norm params stay as f32.
pub struct EncLayer {
    pub wq_weight: Vec<f32>,
    pub wq_bias: Vec<f32>,
    pub wk_weight: Vec<f32>,
    pub wk_bias: Vec<f32>,
    pub wv_weight: Vec<f32>,
    pub wv_bias: Vec<f32>,
    pub wo_weight: Vec<f32>,
    pub wo_bias: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Vec<f32>,
    // R13-Android stage 2: INT8 per-row quantized attention/FFN weights
    // `(int8_data, per_row_scales)`. Populated at load only when the
    // `int8-encoder` feature is compiled AND `QWEN_ASR_INT8_ENCODER != 0`;
    // otherwise left empty and the f32 weights above are used. The f32 weights
    // stay resident for the runtime kill-switch fallback. Desktop/BLAS builds
    // never compile these fields.
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub wq_int8: (Vec<i8>, Vec<f32>),
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub wk_int8: (Vec<i8>, Vec<f32>),
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub wv_int8: (Vec<i8>, Vec<f32>),
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub wo_int8: (Vec<i8>, Vec<f32>),
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub fc1_int8: (Vec<i8>, Vec<f32>),
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub fc2_int8: (Vec<i8>, Vec<f32>),
}

pub struct EncoderBuffers {
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_mid: Vec<f32>,
    pub chunk_mel: Vec<f32>,
    pub c1: Vec<f32>,
    pub c2: Vec<f32>,
    pub c3: Vec<f32>,
    pub reshaped: Vec<f32>,
    pub pe: Vec<f32>,
    pub conv_cols: Vec<f32>,
    pub window_starts: Vec<i32>,
    pub cap_tokens: usize,
}

impl Default for EncoderBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderBuffers {
    pub fn new() -> Self {
        EncoderBuffers {
            x: Vec::new(),
            x_norm: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            ffn_mid: Vec::new(),
            chunk_mel: Vec::new(),
            c1: Vec::new(),
            c2: Vec::new(),
            c3: Vec::new(),
            reshaped: Vec::new(),
            pe: Vec::new(),
            conv_cols: Vec::new(),
            window_starts: Vec::new(),
            cap_tokens: 0,
        }
    }

    pub fn ensure(&mut self, total_tokens: usize, d_model: usize, ffn_dim: usize) {
        if total_tokens <= self.cap_tokens {
            return;
        }
        let mut new_cap = if self.cap_tokens > 0 {
            self.cap_tokens
        } else {
            256
        };
        while new_cap < total_tokens {
            new_cap *= 2;
        }
        self.x.resize(new_cap * d_model, 0.0);
        self.x_norm.resize(new_cap * d_model, 0.0);
        self.q.resize(new_cap * d_model, 0.0);
        self.k.resize(new_cap * d_model, 0.0);
        self.v.resize(new_cap * d_model, 0.0);
        self.attn_out.resize(new_cap * d_model, 0.0);
        self.ffn_mid.resize(new_cap * ffn_dim, 0.0);
        self.cap_tokens = new_cap;
    }

    pub fn ensure_stem(&mut self, chunk_w: usize, d_model: usize) {
        let h1 = (128 + 2 - 3) / 2 + 1;
        let w1 = (chunk_w + 2 - 3) / 2 + 1;
        let h2 = (h1 + 2 - 3) / 2 + 1;
        let w2 = (w1 + 2 - 3) / 2 + 1;
        let h3 = (h2 + 2 - 3) / 2 + 1;
        let w3 = (w2 + 2 - 3) / 2 + 1;
        let conv_proj_dim = CONV_HIDDEN * h3;

        self.chunk_mel.resize(128 * chunk_w, 0.0);
        self.c1.resize(CONV_HIDDEN * h1 * w1, 0.0);
        self.c2.resize(CONV_HIDDEN * h2 * w2, 0.0);
        self.c3.resize(CONV_HIDDEN * h3 * w3, 0.0);
        self.reshaped.resize(w3 * conv_proj_dim, 0.0);
        self.pe.resize(w3 * d_model, 0.0);
    }
}

pub struct Encoder {
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub conv3_weight: Vec<f32>,
    pub conv3_bias: Vec<f32>,
    pub conv_out_weight: Vec<f32>,
    pub layers: Vec<EncLayer>,
    pub ln_post_weight: Vec<f32>,
    pub ln_post_bias: Vec<f32>,
    pub proj1_weight: Vec<f32>,
    pub proj1_bias: Vec<f32>,
    pub proj2_weight: Vec<f32>,
    pub proj2_bias: Vec<f32>,
    // R13-Android stage 2: INT8 per-row quantized conv_out / proj1 / proj2
    // weights `(int8_data, per_row_scales)`. Populated at load only when the
    // `int8-encoder` feature is compiled AND `QWEN_ASR_INT8_ENCODER != 0`.
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub conv_out_int8: (Vec<i8>, Vec<f32>),
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub proj1_int8: (Vec<i8>, Vec<f32>),
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    pub proj2_int8: (Vec<i8>, Vec<f32>),
}

const ENC_PREFIX: &str = "thinker.audio_tower.";

fn load_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    let result = ms.get_f32(name);
    if result.is_none() {
        eprintln!("encoder: weight not found: {}", name);
    }
    result
}

/// Prepack a BF16 weight matrix into an owned, superpage-aligned f32 buffer.
///
/// The BF16 bytes are read out of the safetensors file with `pread` in bounded
/// chunks, never through the mmap, so the encoder's ~0.7 GB of BF16 weight pages
/// never enter the resident set (R5-J). The `(bits << 16)` widening is exact, so
/// the prepacked f32 values are bit-identical to reading through the mmap.
///
/// The per-thread scratch is a local of this call, so the E2 parallel per-layer
/// load gives each loader thread its own buffer by construction. `pread` uses an
/// explicit offset, so concurrent reads on the shared fd are safe.
fn load_bf16_as_f32(ms: &MultiSafetensors, name: &str) -> Option<Vec<f32>> {
    // 1 M u16 elements = 2 MB per pread; converts to a 4 MB f32 slice.
    const CHUNK_ELEMS: usize = 1 << 20;

    let (si, meta) = match ms.find(name) {
        Some(v) => v,
        None => {
            eprintln!("encoder: weight not found: {}", name);
            return None;
        }
    };
    if meta.dtype != Dtype::BF16 {
        eprintln!("encoder: expected BF16 weight for {}", name);
        return None;
    }
    let n = meta.numel();
    let mut dst = kernels::superpage_vec::<f32>(n);
    if n == 0 {
        return Some(dst);
    }
    let shard = &ms.shards[si];
    let mut scratch = vec![0u16; CHUNK_ELEMS.min(n)];
    let mut off = 0usize;
    while off < n {
        let this = (n - off).min(scratch.len());
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut u8, this * 2)
        };
        if !shard.read_tensor_bytes(meta, off * 2, bytes) {
            eprintln!("encoder: pread failed for {}", name);
            return None;
        }
        kernels::bf16_to_f32_buf(&mut dst[off..off + this], &scratch[..this]);
        off += this;
    }
    Some(dst)
}

/// Load one encoder transformer layer. Large weight matrices are prepacked from
/// BF16 into owned f32 buffers here (in parallel across layers) so the forward
/// GEMMs never re-convert them.
fn load_enc_layer(ms: &MultiSafetensors, i: usize) -> Option<EncLayer> {
    let lp = format!("{}layers.{}", ENC_PREFIX, i);
    #[allow(unused_mut)]
    let mut layer = EncLayer {
        wq_weight: load_bf16_as_f32(ms, &format!("{}.self_attn.q_proj.weight", lp))?,
        wq_bias: load_f32(ms, &format!("{}.self_attn.q_proj.bias", lp))?,
        wk_weight: load_bf16_as_f32(ms, &format!("{}.self_attn.k_proj.weight", lp))?,
        wk_bias: load_f32(ms, &format!("{}.self_attn.k_proj.bias", lp))?,
        wv_weight: load_bf16_as_f32(ms, &format!("{}.self_attn.v_proj.weight", lp))?,
        wv_bias: load_f32(ms, &format!("{}.self_attn.v_proj.bias", lp))?,
        wo_weight: load_bf16_as_f32(ms, &format!("{}.self_attn.out_proj.weight", lp))?,
        wo_bias: load_f32(ms, &format!("{}.self_attn.out_proj.bias", lp))?,
        attn_norm_weight: load_f32(ms, &format!("{}.self_attn_layer_norm.weight", lp))?,
        attn_norm_bias: load_f32(ms, &format!("{}.self_attn_layer_norm.bias", lp))?,
        fc1_weight: load_bf16_as_f32(ms, &format!("{}.fc1.weight", lp))?,
        fc1_bias: load_f32(ms, &format!("{}.fc1.bias", lp))?,
        fc2_weight: load_bf16_as_f32(ms, &format!("{}.fc2.weight", lp))?,
        fc2_bias: load_f32(ms, &format!("{}.fc2.bias", lp))?,
        ffn_norm_weight: load_f32(ms, &format!("{}.final_layer_norm.weight", lp))?,
        ffn_norm_bias: load_f32(ms, &format!("{}.final_layer_norm.bias", lp))?,
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        wq_int8: (Vec::new(), Vec::new()),
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        wk_int8: (Vec::new(), Vec::new()),
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        wv_int8: (Vec::new(), Vec::new()),
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        wo_int8: (Vec::new(), Vec::new()),
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        fc1_int8: (Vec::new(), Vec::new()),
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        fc2_int8: (Vec::new(), Vec::new()),
    };
    // R13-Android stage 2: quantize the attention/FFN weight GEMMs to INT8
    // per-row at load, only when the feature is compiled AND the runtime switch
    // is on. `out_dim` is the bias length; `in_dim` is the remaining stride.
    #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
    if kernels::int8_encoder_enabled() {
        let q = |w: &[f32], out_dim: usize| {
            kernels::quantize_f32_weights_to_int8(w, out_dim, w.len() / out_dim)
        };
        let d = layer.wq_bias.len();
        layer.wq_int8 = q(&layer.wq_weight, d);
        layer.wk_int8 = q(&layer.wk_weight, layer.wk_bias.len());
        layer.wv_int8 = q(&layer.wv_weight, layer.wv_bias.len());
        layer.wo_int8 = q(&layer.wo_weight, layer.wo_bias.len());
        layer.fc1_int8 = q(&layer.fc1_weight, layer.fc1_bias.len());
        layer.fc2_int8 = q(&layer.fc2_weight, layer.fc2_bias.len());
    }
    Some(layer)
}

/// R13-Android stage 2: quantize `x` per-row and run the INT8 encoder GEMM
/// `y = x @ Wᵀ (+ bias)`, optionally accumulating in place. `xq`/`xs` are reused
/// scratch buffers grown on demand. Only compiled on the no-BLAS aarch64 build.
#[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn enc_int8_linear(
    y: &mut [f32], x: &[f32], w: &(Vec<i8>, Vec<f32>), bias: Option<&[f32]>,
    seq_len: usize, in_dim: usize, out_dim: usize, accumulate: bool,
    xq: &mut Vec<i8>, xs: &mut Vec<f32>,
) {
    if xq.len() < seq_len * in_dim {
        xq.resize(seq_len * in_dim, 0);
    }
    if xs.len() < seq_len {
        xs.resize(seq_len, 0.0);
    }
    kernels::quantize_rows_into(&mut xq[..seq_len * in_dim], &mut xs[..seq_len], x, seq_len, in_dim);
    kernels::int8_encoder_matvec(
        y, &xq[..seq_len * in_dim], &xs[..seq_len],
        w.0.as_ptr(), w.1.as_ptr(), bias, in_dim, out_dim, seq_len, accumulate,
    );
}

impl Encoder {
    /// R13-Android stage 2 diagnostic: whether the INT8 encoder weight GEMMs are
    /// actually live for this loaded model — i.e. the `int8-encoder` feature is
    /// compiled AND the runtime switch was on at load so the weights got
    /// quantized. This is exactly the `use_int8` condition the forward checks, so
    /// it is the ground-truth "encoder INT8 active" signal (not merely the env
    /// switch). Always `false` on desktop/BLAS/non-aarch64 builds by construction.
    pub fn int8_encoder_active(&self) -> bool {
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        {
            return !self.conv_out_int8.0.is_empty();
        }
        #[allow(unreachable_code)]
        false
    }

    pub fn load(ms: &MultiSafetensors, cfg: &QwenConfig) -> Option<Self> {
        let p = ENC_PREFIX;

        let conv1_weight = load_f32(ms, &format!("{}conv2d1.weight", p))?;
        let conv1_bias = load_f32(ms, &format!("{}conv2d1.bias", p))?;
        let conv2_weight = load_f32(ms, &format!("{}conv2d2.weight", p))?;
        let conv2_bias = load_f32(ms, &format!("{}conv2d2.bias", p))?;
        let conv3_weight = load_f32(ms, &format!("{}conv2d3.weight", p))?;
        let conv3_bias = load_f32(ms, &format!("{}conv2d3.bias", p))?;
        let conv_out_weight = load_bf16_as_f32(ms, &format!("{}conv_out.weight", p))?;

        // Prepack each layer's transformer GEMM weights from BF16 to f32 in
        // parallel across layers, so the parallel load absorbs the conversion
        // cost once instead of the (serial) forward paying it every call.
        let nlayers = cfg.enc_layers;
        let nthreads = kernels::get_num_cpus().min(nlayers).max(1);
        let chunk = nlayers.div_ceil(nthreads);
        let mut indexed: Vec<(usize, EncLayer)> = std::thread::scope(|s| {
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
                        out.push((i, load_enc_layer(ms, i)?));
                    }
                    Some(out)
                }));
            }
            let mut all: Vec<(usize, EncLayer)> = Vec::with_capacity(nlayers);
            for h in handles {
                all.extend(h.join().ok()??);
            }
            Some(all)
        })?;
        indexed.sort_by_key(|(i, _)| *i);
        let layers: Vec<EncLayer> = indexed.into_iter().map(|(_, l)| l).collect();

        let ln_post_weight = load_f32(ms, &format!("{}ln_post.weight", p))?;
        let ln_post_bias = load_f32(ms, &format!("{}ln_post.bias", p))?;
        let proj1_weight = load_bf16_as_f32(ms, &format!("{}proj1.weight", p))?;
        let proj1_bias = load_f32(ms, &format!("{}proj1.bias", p))?;
        let proj2_weight = load_bf16_as_f32(ms, &format!("{}proj2.weight", p))?;
        let proj2_bias = load_f32(ms, &format!("{}proj2.bias", p))?;

        // R13-Android stage 2: quantize conv_out / proj1 / proj2 GEMM weights to
        // INT8 per-row at load, only when compiled AND the runtime switch is on.
        // conv_out has no bias, so its out_dim is d_model (in_dim = 480*h3).
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        let (conv_out_int8, proj1_int8, proj2_int8) = if kernels::int8_encoder_enabled() {
            let d = cfg.enc_d_model;
            (
                kernels::quantize_f32_weights_to_int8(&conv_out_weight, d, conv_out_weight.len() / d),
                kernels::quantize_f32_weights_to_int8(&proj1_weight, proj1_bias.len(), proj1_weight.len() / proj1_bias.len()),
                kernels::quantize_f32_weights_to_int8(&proj2_weight, proj2_bias.len(), proj2_weight.len() / proj2_bias.len()),
            )
        } else {
            ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()), (Vec::new(), Vec::new()))
        };

        Some(Encoder {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            conv3_weight,
            conv3_bias,
            conv_out_weight,
            layers,
            ln_post_weight,
            ln_post_bias,
            proj1_weight,
            proj1_bias,
            proj2_weight,
            proj2_bias,
            #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
            conv_out_int8,
            #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
            proj1_int8,
            #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
            proj2_int8,
        })
    }

    /// Run encoder forward pass on mel spectrogram.
    /// mel: [128, mel_frames], returns [total_tokens, output_dim].
    pub fn forward(
        &self,
        cfg: &QwenConfig,
        mel: &[f32],
        mel_frames: usize,
        enc_bufs: Option<&mut EncoderBuffers>,
    ) -> Option<(Vec<f32>, usize)> {
        let d_model = cfg.enc_d_model;
        let n_heads = cfg.enc_heads;
        let head_dim = cfg.enc_head_dim;
        let ffn_dim = cfg.enc_ffn_dim;
        let output_dim = cfg.enc_output_dim;
        let chunk_size = cfg.enc_chunk_size;
        let n_window_infer = cfg.enc_n_window_infer;

        // R13-Android stage 2: route the encoder weight GEMMs through resident
        // INT8 weights on the no-BLAS aarch64 (Android) build when compiled AND
        // the runtime switch is on AND the INT8 weights were quantized at load.
        // Desktop/BLAS builds never compile this branch (stays on the f32
        // `linear`/`linear_accumulate` path). Activation×activation attention
        // GEMMs stay f32 either way.
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        let use_int8 = kernels::int8_encoder_enabled()
            && self.layers.first().map_or(false, |l| !l.wq_int8.0.is_empty())
            && !self.conv_out_int8.0.is_empty();
        #[cfg(not(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64")))]
        let use_int8 = false;

        // Reused per-position INT8 activation scratch (grown on demand). Only
        // allocated for the INT8 encoder path.
        #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
        let (mut xq_buf, mut xs_buf): (Vec<i8>, Vec<f32>) = (Vec::new(), Vec::new());

        // Determine tokens per full chunk
        let tokens_per_chunk = {
            let w = chunk_size;
            let w1 = (w + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            (w2 + 2 - 3) / 2 + 1
        };

        let n_chunks = mel_frames.div_ceil(chunk_size);

        // Pre-calculate total tokens
        let mut total_tokens = 0;
        let mut chunk_sizes = Vec::new();
        for c in 0..n_chunks {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(mel_frames);
            let chunk_w = end - start;
            let w1 = (chunk_w + 2 - 3) / 2 + 1;
            let w2 = (w1 + 2 - 3) / 2 + 1;
            let w3 = (w2 + 2 - 3) / 2 + 1;
            total_tokens += w3;
            chunk_sizes.push((start, end, w3));
        }

        // Transformer + stem scratch buffers (reusable or fresh)
        let mut _owned_bufs;
        let bufs: &mut EncoderBuffers = match enc_bufs {
            Some(b) => {
                b.ensure(total_tokens, d_model, ffn_dim);
                b
            }
            None => {
                _owned_bufs = EncoderBuffers::new();
                _owned_bufs.ensure(total_tokens, d_model, ffn_dim);
                &mut _owned_bufs
            }
        };

        // Main sequence buffer: [total_tokens, d_model]
        let td = total_tokens * d_model;
        let mut token_offset = 0;

        // Process each chunk through Conv2D + reshape + project + sinusoidal PE
        for &(start, end, w3) in &chunk_sizes {
            let chunk_w = end - start;
            bufs.ensure_stem(chunk_w, d_model);

            // Extract chunk mel: [128, chunk_w]
            let chunk_mel = &mut bufs.chunk_mel[..128 * chunk_w];
            for m in 0..128 {
                chunk_mel[m * chunk_w..(m + 1) * chunk_w]
                    .copy_from_slice(&mel[m * mel_frames + start..m * mel_frames + end]);
            }

            // Conv2D layer 1: [1, 128, chunk_w] -> [480, h1, w1]
            let h1 = (128 + 2 - 3) / 2 + 1; // 64
            let w1 = (chunk_w + 2 - 3) / 2 + 1;
            let c1 = &mut bufs.c1[..CONV_HIDDEN * h1 * w1];
            kernels::conv2d_with_cols(
                c1,
                chunk_mel,
                &self.conv1_weight,
                Some(&self.conv1_bias),
                &mut bufs.conv_cols,
                1,
                CONV_HIDDEN,
                128,
                chunk_w,
                3,
                3,
                2,
                1,
            );
            kernels::gelu(c1, CONV_HIDDEN * h1 * w1);

            // Conv2D layer 2: [480, h1, w1] -> [480, h2, w2]
            let h2 = (h1 + 2 - 3) / 2 + 1; // 32
            let w2 = (w1 + 2 - 3) / 2 + 1;
            let c2 = &mut bufs.c2[..CONV_HIDDEN * h2 * w2];
            kernels::conv2d_with_cols(
                c2,
                c1,
                &self.conv2_weight,
                Some(&self.conv2_bias),
                &mut bufs.conv_cols,
                CONV_HIDDEN,
                CONV_HIDDEN,
                h1,
                w1,
                3,
                3,
                2,
                1,
            );
            kernels::gelu(c2, CONV_HIDDEN * h2 * w2);

            // Conv2D layer 3: [480, h2, w2] -> [480, h3, w3]
            let h3 = (h2 + 2 - 3) / 2 + 1; // 16
            let _w3_calc = (w2 + 2 - 3) / 2 + 1;
            debug_assert_eq!(_w3_calc, w3);
            let c3 = &mut bufs.c3[..CONV_HIDDEN * h3 * w3];
            kernels::conv2d_with_cols(
                c3,
                c2,
                &self.conv3_weight,
                Some(&self.conv3_bias),
                &mut bufs.conv_cols,
                CONV_HIDDEN,
                CONV_HIDDEN,
                h2,
                w2,
                3,
                3,
                2,
                1,
            );
            kernels::gelu(c3, CONV_HIDDEN * h3 * w3);

            // Reshape [480, h3, w3] -> [w3, 480*h3]
            // Loop order: ch → f → t for sequential reads from c3
            let conv_proj_dim = CONV_HIDDEN * h3;
            let reshaped = &mut bufs.reshaped[..w3 * conv_proj_dim];
            for ch in 0..CONV_HIDDEN {
                for f in 0..h3 {
                    let src_off = ch * h3 * w3 + f * w3;
                    let dst_col = ch * h3 + f;
                    for t in 0..w3 {
                        reshaped[t * conv_proj_dim + dst_col] = c3[src_off + t];
                    }
                }
            }

            // Project: [w3, 7680] -> [w3, d_model]
            let projected = &mut bufs.x[token_offset * d_model..(token_offset + w3) * d_model];
            if use_int8 {
                #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
                unsafe {
                    enc_int8_linear(
                        projected, reshaped, &self.conv_out_int8, None,
                        w3, conv_proj_dim, d_model, false, &mut xq_buf, &mut xs_buf,
                    );
                }
            } else {
                kernels::linear(
                    projected,
                    reshaped,
                    &self.conv_out_weight,
                    None,
                    w3,
                    conv_proj_dim,
                    d_model,
                );
            }

            // Add per-chunk sinusoidal PE
            let pe = &mut bufs.pe[..w3 * d_model];
            kernels::sinusoidal_pe(pe, w3, d_model);
            kernels::add_inplace(projected, pe, w3 * d_model);

            token_offset += w3;
        }

        // Build attention window boundaries
        let window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);
        let n_windows = total_tokens.div_ceil(window_token_size);
        bufs.window_starts.resize(n_windows + 1, 0);
        let window_starts = &mut bufs.window_starts[..n_windows + 1];
        for (w, ws) in window_starts.iter_mut().enumerate().take(n_windows) {
            *ws = (w * window_token_size) as i32;
        }
        window_starts[n_windows] = total_tokens as i32;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let tf = total_tokens * ffn_dim;

        for layer in &self.layers {
            // Self-attention
            kernels::layer_norm(
                &mut bufs.x_norm[..td],
                &bufs.x[..td],
                &layer.attn_norm_weight,
                &layer.attn_norm_bias,
                total_tokens,
                d_model,
                1e-5,
            );

            if use_int8 {
                #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
                unsafe {
                    enc_int8_linear(
                        &mut bufs.q[..td], &bufs.x_norm[..td], &layer.wq_int8, Some(&layer.wq_bias),
                        total_tokens, d_model, d_model, false, &mut xq_buf, &mut xs_buf,
                    );
                    enc_int8_linear(
                        &mut bufs.k[..td], &bufs.x_norm[..td], &layer.wk_int8, Some(&layer.wk_bias),
                        total_tokens, d_model, d_model, false, &mut xq_buf, &mut xs_buf,
                    );
                    enc_int8_linear(
                        &mut bufs.v[..td], &bufs.x_norm[..td], &layer.wv_int8, Some(&layer.wv_bias),
                        total_tokens, d_model, d_model, false, &mut xq_buf, &mut xs_buf,
                    );
                }
            } else {
                kernels::linear(
                    &mut bufs.q[..td],
                    &bufs.x_norm[..td],
                    &layer.wq_weight,
                    Some(&layer.wq_bias),
                    total_tokens,
                    d_model,
                    d_model,
                );
                kernels::linear(
                    &mut bufs.k[..td],
                    &bufs.x_norm[..td],
                    &layer.wk_weight,
                    Some(&layer.wk_bias),
                    total_tokens,
                    d_model,
                    d_model,
                );
                kernels::linear(
                    &mut bufs.v[..td],
                    &bufs.x_norm[..td],
                    &layer.wv_weight,
                    Some(&layer.wv_bias),
                    total_tokens,
                    d_model,
                    d_model,
                );
            }

            kernels::bidirectional_attention(
                &mut bufs.attn_out[..td],
                &bufs.q[..td],
                &bufs.k[..td],
                &bufs.v[..td],
                total_tokens,
                n_heads,
                head_dim,
                scale,
                window_starts,
                n_windows,
            );

            // Fused: x += wo_bias + attn_out @ wo_weight.T
            if use_int8 {
                #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
                unsafe {
                    enc_int8_linear(
                        &mut bufs.x[..td], &bufs.attn_out[..td], &layer.wo_int8, Some(&layer.wo_bias),
                        total_tokens, d_model, d_model, true, &mut xq_buf, &mut xs_buf,
                    );
                }
            } else {
                kernels::linear_accumulate(
                    &mut bufs.x[..td],
                    &bufs.attn_out[..td],
                    &layer.wo_weight,
                    Some(&layer.wo_bias),
                    total_tokens,
                    d_model,
                    d_model,
                );
            }

            // FFN
            kernels::layer_norm(
                &mut bufs.x_norm[..td],
                &bufs.x[..td],
                &layer.ffn_norm_weight,
                &layer.ffn_norm_bias,
                total_tokens,
                d_model,
                1e-5,
            );

            if use_int8 {
                #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
                unsafe {
                    enc_int8_linear(
                        &mut bufs.ffn_mid[..tf], &bufs.x_norm[..td], &layer.fc1_int8, Some(&layer.fc1_bias),
                        total_tokens, d_model, ffn_dim, false, &mut xq_buf, &mut xs_buf,
                    );
                }
            } else {
                kernels::linear(
                    &mut bufs.ffn_mid[..tf],
                    &bufs.x_norm[..td],
                    &layer.fc1_weight,
                    Some(&layer.fc1_bias),
                    total_tokens,
                    d_model,
                    ffn_dim,
                );
            }
            kernels::gelu(&mut bufs.ffn_mid[..tf], tf);
            // Fused: x += fc2_bias + ffn_mid @ fc2_weight.T
            if use_int8 {
                #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
                unsafe {
                    enc_int8_linear(
                        &mut bufs.x[..td], &bufs.ffn_mid[..tf], &layer.fc2_int8, Some(&layer.fc2_bias),
                        total_tokens, ffn_dim, d_model, true, &mut xq_buf, &mut xs_buf,
                    );
                }
            } else {
                kernels::linear_accumulate(
                    &mut bufs.x[..td],
                    &bufs.ffn_mid[..tf],
                    &layer.fc2_weight,
                    Some(&layer.fc2_bias),
                    total_tokens,
                    ffn_dim,
                    d_model,
                );
            }
        }

        // Final LayerNorm: use x_norm as temp, then swap into x
        kernels::layer_norm(
            &mut bufs.x_norm[..td],
            &bufs.x[..td],
            &self.ln_post_weight,
            &self.ln_post_bias,
            total_tokens,
            d_model,
            1e-5,
        );
        bufs.x[..td].copy_from_slice(&bufs.x_norm[..td]);

        // Projection: proj1 (GELU) -> proj2 (reuse activation buffers)
        if use_int8 {
            #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
            unsafe {
                enc_int8_linear(
                    &mut bufs.q[..td], &bufs.x[..td], &self.proj1_int8, Some(&self.proj1_bias),
                    total_tokens, d_model, d_model, false, &mut xq_buf, &mut xs_buf,
                );
            }
        } else {
            kernels::linear(
                &mut bufs.q[..td],
                &bufs.x[..td],
                &self.proj1_weight,
                Some(&self.proj1_bias),
                total_tokens,
                d_model,
                d_model,
            );
        }
        kernels::gelu(&mut bufs.q[..td], td);

        let mut enc_output = vec![0.0f32; total_tokens * output_dim];
        if use_int8 {
            #[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
            unsafe {
                enc_int8_linear(
                    &mut enc_output, &bufs.q[..td], &self.proj2_int8, Some(&self.proj2_bias),
                    total_tokens, d_model, output_dim, false, &mut xq_buf, &mut xs_buf,
                );
            }
        } else {
            kernels::linear(
                &mut enc_output,
                &bufs.q[..td],
                &self.proj2_weight,
                Some(&self.proj2_bias),
                total_tokens,
                d_model,
                output_dim,
            );
        }

        Some((enc_output, total_tokens))
    }
}
