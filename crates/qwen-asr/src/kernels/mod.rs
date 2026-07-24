//! BLAS/vDSP bindings, thread pool, and SIMD kernel dispatch.

pub mod generic;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "x86_64")]
pub mod avx;

const SUPERPAGE_SIZE: usize = 2 * 1024 * 1024;

/// Allocate a zeroed Vec backed by superpage-aligned memory.
/// On Apple Silicon this makes it possible for the kernel to use 2 MB pages
/// for large hot weight buffers, reducing TLB pressure during the streaming
/// weight reads of decode and encoder GEMMs. Falls back to a normal Vec on
/// failure. Shared by the decoder INT8/f32 weight prepack and the encoder
/// f32 weight prepack.
pub fn superpage_vec<T: Copy>(n: usize) -> Vec<T> {
    let size = n.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
    if size < SUPERPAGE_SIZE {
        return vec![unsafe { std::mem::zeroed() }; n];
    }
    let mut ptr = std::ptr::null_mut();
    let rc = unsafe { libc::posix_memalign(&mut ptr, SUPERPAGE_SIZE, size) };
    if rc != 0 || ptr.is_null() {
        return vec![unsafe { std::mem::zeroed() }; n];
    }
    unsafe {
        std::ptr::write_bytes(ptr, 0, size);
        Vec::from_raw_parts(ptr as *mut T, n, n)
    }
}

// BLAS extern bindings
#[cfg(all(feature = "blas", target_vendor = "apple"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

// vDSP/vForce bindings (macOS Accelerate)
#[cfg(all(feature = "vdsp", target_vendor = "apple"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn vDSP_dotpr(
        a: *const f32, a_stride: i32,
        b: *const f32, b_stride: i32,
        result: *mut f32,
        n: u64,
    );
    fn vDSP_vsmul(
        a: *const f32, a_stride: i32,
        scalar: *const f32,
        c: *mut f32, c_stride: i32,
        n: u64,
    );
    fn vDSP_vsma(
        a: *const f32, a_stride: i32,
        scalar: *const f32,
        b: *const f32, b_stride: i32,
        c: *mut f32, c_stride: i32,
        n: u64,
    );
    fn vvexpf(dst: *mut f32, src: *const f32, n: *const i32);
}

#[cfg(all(feature = "blas", not(target_vendor = "apple")))]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

#[cfg(feature = "blas")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(feature = "blas")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(feature = "blas")]
const CBLAS_TRANS: i32 = 112;

// Verbose flag
static VERBOSE: AtomicI32 = AtomicI32::new(0);

// ========================================================================
// Profiling support
// ========================================================================

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI32, Ordering};
use std::time::Instant;

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_profile(enabled: bool) {
    PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

macro_rules! define_profile_counters {
    ($($name:ident),+) => {
        pub struct ProfileCounters {
            $(pub $name: (AtomicU64, AtomicU64),)+ // (total_ns, call_count)
        }

        impl ProfileCounters {
            pub const fn new() -> Self {
                ProfileCounters {
                    $($name: (AtomicU64::new(0), AtomicU64::new(0)),)+
                }
            }
        }

        impl Default for ProfileCounters {
            fn default() -> Self {
                Self::new()
            }
        }

        impl ProfileCounters {
            pub fn reset(&self) {
                $(
                    self.$name.0.store(0, Ordering::Relaxed);
                    self.$name.1.store(0, Ordering::Relaxed);
                )+
            }

            pub fn report(&self) {
                $(
                    let ns = self.$name.0.load(Ordering::Relaxed);
                    let calls = self.$name.1.load(Ordering::Relaxed);
                    if calls > 0 {
                        let ms = ns as f64 / 1_000_000.0;
                        let avg = ms / calls as f64;
                        eprintln!("[profile] {}: {:.1}ms ({} calls, {:.2}ms avg)",
                                  stringify!($name), ms, calls, avg);
                    }
                )+
            }
        }
    }
}

define_profile_counters!(
    rms_norm, layer_norm, gelu, swiglu,
    bf16_matvec, attention_bidir, attention_causal,
    sgemm, conv2d_op, rope, add_inplace_op,
    model_load, encoder_load, decoder_load, tokenizer_load, audio_load, mel_compute
);

pub static PROF: ProfileCounters = ProfileCounters::new();

pub struct ProfileGuard {
    start: Instant,
    counter: &'static (AtomicU64, AtomicU64),
}

impl ProfileGuard {
    #[inline]
    pub fn new(counter: &'static (AtomicU64, AtomicU64)) -> Option<Self> {
        if PROFILE_ENABLED.load(Ordering::Relaxed) {
            Some(ProfileGuard { start: Instant::now(), counter })
        } else {
            None
        }
    }
}

impl Drop for ProfileGuard {
    #[inline]
    fn drop(&mut self) {
        let ns = self.start.elapsed().as_nanos() as u64;
        self.counter.0.fetch_add(ns, Ordering::Relaxed);
        self.counter.1.fetch_add(1, Ordering::Relaxed);
    }
}

// Convenience: unused ProfileTimer alias removed

pub fn profile_reset() { PROF.reset(); }
pub fn profile_report() { PROF.report(); }

pub fn set_verbose(v: i32) {
    VERBOSE.store(v, Ordering::Relaxed);
}

pub fn verbose() -> i32 {
    VERBOSE.load(Ordering::Relaxed)
}

// ========================================================================
// Thread pool + parallel region (kernels/pool.rs)
// ========================================================================

mod pool;

pub use pool::{get_default_threads, get_num_cpus, get_num_threads, set_threads};
pub(crate) use pool::parallel_for;
pub(crate) use pool::parallel_for_dynamic;
pub(crate) use pool::set_thread_override;
#[cfg(target_arch = "aarch64")]
pub(crate) use pool::{range_for, MAX_THREADS};
#[cfg(target_arch = "aarch64")]
pub(crate) use pool::parallel_region;

// ========================================================================
// Dispatch helpers - pick NEON/AVX/generic at compile time
// ========================================================================

#[inline]
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

pub fn bf16_to_f32_buf(dst: &mut [f32], src: &[u16]) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::bf16_to_f32_buf(dst, src); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::bf16_to_f32_buf(dst, src); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in 0..src.len() {
        dst[i] = bf16_to_f32(src[i]);
    }
}

/// Parallel bf16→f32 widening: splits `src`/`dst` across the persistent thread
/// pool so idle workers share the conversion instead of one core streaming the
/// whole matrix. Bit-exact vs [`bf16_to_f32_buf`] (pure element-wise widening
/// over disjoint chunks). Chunk boundaries are aligned to a multiple of 64
/// elements so each worker (except the last) runs the SIMD converter on a whole
/// vector-width-friendly span with no per-chunk tail. Falls back to the serial
/// converter below a size threshold or when single-threaded, to avoid dispatch
/// overhead on small conversions.
pub fn bf16_to_f32_buf_parallel(dst: &mut [f32], src: &[u16]) {
    // Below this many elements the dispatch/wake/join cost outweighs the win.
    const PAR_THRESHOLD: usize = 1 << 18;
    let n = src.len();
    if n < PAR_THRESHOLD || get_num_threads() <= 1 {
        bf16_to_f32_buf(dst, src);
        return;
    }

    // SAFETY: workers touch disjoint [start, end) spans of the same buffers.
    let dst_send = dst.as_mut_ptr() as usize;
    let src_send = src.as_ptr() as usize;

    // Fixed 32K-element items (multiple of 64, so each item is a whole
    // vector-width-friendly span with no per-item tail), grabbed dynamically so
    // faster cores convert more spans. Conversion is element-wise, so the result
    // is bit-identical regardless of which core converts which item.
    const ITEM: usize = 1 << 15; // 32768 elements
    let n_items = n.div_ceil(ITEM);
    parallel_for_dynamic(n_items, |item| {
        let start = item * ITEM;
        let end = (start + ITEM).min(n);
        let len = end - start;
        let dst_local = unsafe { std::slice::from_raw_parts_mut((dst_send as *mut f32).add(start), len) };
        let src_local = unsafe { std::slice::from_raw_parts((src_send as *const u16).add(start), len) };
        bf16_to_f32_buf(dst_local, src_local);
    });
}

fn bf16_matvec_fused(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    // SAFETY: Callers provide `w_bf16` with at least `out_dim * in_dim`
    // readable elements; the architecture-specific implementations above
    // rely on the same contract.
    unsafe { generic::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); }
}

fn argmax_bf16_range(x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize) -> (usize, f32) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::argmax_bf16_range(x, w_bf16, in_dim, start, end) } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::argmax_bf16_range(x, w_bf16, in_dim, start, end) } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    // SAFETY: Callers provide `w_bf16` with at least `end * in_dim` readable
    // elements; the architecture-specific implementations above use the same
    // raw-pointer contract.
    unsafe { generic::argmax_bf16_range(x, w_bf16, in_dim, start, end) }
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        let mut result = 0.0f32;
        unsafe { vDSP_dotpr(a.as_ptr(), 1, b.as_ptr(), 1, &mut result, n as u64); }
        result
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::dot_f32(a, b, n) } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::dot_f32(a, b, n) } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::dot_f32(a, b, n)
}

#[inline]
pub fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        unsafe { vDSP_vsmul(dst.as_ptr(), 1, &scale, dst.as_mut_ptr(), 1, n as u64); }
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::vec_scale_inplace(dst, scale, n); } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::vec_scale_inplace(dst, scale, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::vec_scale_inplace(dst, scale, n);
}

#[inline]
pub fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        unsafe { vDSP_vsma(src.as_ptr(), 1, &alpha, dst.as_ptr(), 1, dst.as_mut_ptr(), 1, n as u64); }
    }

    #[cfg(all(target_arch = "aarch64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { neon::vec_axpy_inplace(dst, src, alpha, n); } }

    #[cfg(all(target_arch = "x86_64", not(all(feature = "vdsp", target_vendor = "apple"))))]
    { unsafe { avx::vec_axpy_inplace(dst, src, alpha, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", all(feature = "vdsp", target_vendor = "apple"))))]
    generic::vec_axpy_inplace(dst, src, alpha, n);
}

#[inline]
pub fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::vec_scale_add(dst, src, correction, n); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::vec_scale_add(dst, src, correction, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::vec_scale_add(dst, src, correction, n);
}

// ========================================================================
// Basic Operations
// ========================================================================

pub fn add_inplace(a: &mut [f32], b: &[f32], n: usize) {
    let _pg = ProfileGuard::new(&PROF.add_inplace_op);
    for i in 0..n { a[i] += b[i]; }
}

// ========================================================================
// Matrix Operations
// ========================================================================

// ------------------------------------------------------------------------
// No-BLAS NEON pool-parallel GEMM fallbacks (R13 Android track)
//
// On Android the `blas` feature is off, so every GEMM would otherwise fall
// through a naive scalar triple loop (single-threaded, no SIMD) — ~100-800x
// slower than the AMX/BLAS reference. These helpers replace that fallback with
// NEON-vectorized, pool-parallel kernels at the sgemm-equivalent seam, so every
// caller (encoder conv/attention, decoder prefill, lm_head prefill) benefits
// with no call-site changes. Compiled ONLY for `not(feature = "blas")` +
// aarch64; other no-BLAS arches keep the scalar loops inline in each wrapper.
//
// Two shapes are covered:
//   * A·Bᵀ (dot-product style): `y[s,o] = bias[o] + dot(x[s], w[o])`, weight
//     rows contiguous — mapped to per-output-row `neon::dot_f32`. Used by
//     `linear`, `linear_accumulate`, `matmul_t`.
//   * A·B (row-major B): `c[m,n] = sum_k a[m,k]*b[k,n]` — mapped to an
//     axpy accumulation over B's contiguous rows. Used by `matmul_nn` and the
//     conv2d GEMM.
//
// Output rows are partitioned across the persistent pool with
// `parallel_for_dynamic`; small GEMMs stay single-threaded inline (R10 >=4M MAC
// / >=128 col dispatch threshold). Vectorized dots reorder float summation vs
// the old scalar loop — acceptable for no-BLAS (nothing depends on its exact
// bits; BLAS builds are untouched).
// ------------------------------------------------------------------------

/// Dispatch threshold shared by the fallbacks: below this many MACs a GEMM
/// stays single-threaded (pool wake/join cost would dominate).
#[cfg(all(not(feature = "blas"), target_arch = "aarch64"))]
const FALLBACK_MIN_MACS: usize = 1 << 22; // 4M

/// Compute output columns (weight rows) `[start, end)` for all `seq_len` rows of
/// the A·Bᵀ fallback: `y[s,o] = (accumulate ? y[s,o] : 0) + bias[o] + dot(x[s], w[o])`.
/// Each weight row `w[o]` is streamed once and reused across every activation
/// row via `neon::dot_f32`.
///
/// # Safety
/// `x`/`w`/`y` must be valid for `seq_len*in_dim`, `out_dim*in_dim`,
/// `seq_len*out_dim` elements; `bias` (if set) for `out_dim`. `[start,end)`
/// must be within `[0,out_dim)`.
#[cfg(all(not(feature = "blas"), target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_nt_rows(
    y: *mut f32, x: *const f32, w: *const f32, bias: Option<*const f32>,
    seq_len: usize, in_dim: usize, out_dim: usize,
    start: usize, end: usize, accumulate: bool,
) {
    for o in start..end {
        let w_row = std::slice::from_raw_parts(w.add(o * in_dim), in_dim);
        let bo = match bias { Some(b) => *b.add(o), None => 0.0 };
        for s in 0..seq_len {
            let x_row = std::slice::from_raw_parts(x.add(s * in_dim), in_dim);
            let d = neon::dot_f32(x_row, w_row, in_dim) + bo;
            let cell = y.add(s * out_dim + o);
            if accumulate { *cell += d; } else { *cell = d; }
        }
    }
}

/// NEON pool-parallel A·Bᵀ GEMM fallback (`linear`/`linear_accumulate`/`matmul_t`).
/// Partitions the `out_dim` weight rows across the pool; each item writes a
/// disjoint output-column range of every `y` row.
#[cfg(all(not(feature = "blas"), target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn gemm_nt_fallback(
    y: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>,
    seq_len: usize, in_dim: usize, out_dim: usize, accumulate: bool,
) {
    const MIN_COLS: usize = 128;
    let nt = get_num_threads();
    let parallel = nt > 1
        && out_dim >= MIN_COLS
        && seq_len.saturating_mul(in_dim).saturating_mul(out_dim) >= FALLBACK_MIN_MACS;
    if !parallel {
        // SAFETY: single-threaded full-range pass; slices sized by the wrapper.
        unsafe {
            gemm_nt_rows(y.as_mut_ptr(), x.as_ptr(), w.as_ptr(), bias.map(|b| b.as_ptr()),
                         seq_len, in_dim, out_dim, 0, out_dim, accumulate);
        }
        return;
    }
    let y_send = y.as_mut_ptr() as usize;
    let x_send = x.as_ptr() as usize;
    let w_send = w.as_ptr() as usize;
    let b_send = bias.map(|b| b.as_ptr() as usize);
    // Fixed 64-row output blocks grabbed dynamically so P-cores take more than
    // E-cores; boundaries depend only on out_dim, so the split is deterministic.
    const ROWS: usize = 64;
    let n_items = out_dim.div_ceil(ROWS);
    parallel_for_dynamic(n_items, |item| {
        let start = item * ROWS;
        let end = (start + ROWS).min(out_dim);
        if start >= end { return; }
        // SAFETY: items write disjoint output-column ranges [start,end).
        unsafe {
            gemm_nt_rows(y_send as *mut f32, x_send as *const f32, w_send as *const f32,
                         b_send.map(|p| p as *const f32),
                         seq_len, in_dim, out_dim, start, end, accumulate);
        }
    });
}

/// Compute rows `[start, end)` of the A·B fallback: `c[mi,:] = sum_k a[mi,k]*b[k,:]`,
/// accumulated as axpy over B's contiguous rows. Overwrites `c[mi,:]` (no read).
///
/// # Safety
/// `a`/`b`/`c` valid for `m*k`, `k*n`, `m*n` elements; `[start,end)` within `[0,m)`.
#[cfg(all(not(feature = "blas"), target_arch = "aarch64"))]
unsafe fn gemm_nn_rows(
    c: *mut f32, a: *const f32, b: *const f32, k: usize, n: usize, start: usize, end: usize,
) {
    for mi in start..end {
        let c_row = std::slice::from_raw_parts_mut(c.add(mi * n), n);
        for v in c_row.iter_mut() { *v = 0.0; }
        for ki in 0..k {
            let av = *a.add(mi * k + ki);
            let b_row = std::slice::from_raw_parts(b.add(ki * n), n);
            neon::vec_axpy_inplace(c_row, b_row, av, n);
        }
    }
}

/// NEON pool-parallel A·B GEMM fallback (`matmul_nn` / conv2d GEMM).
/// Partitions the `m` output rows across the pool.
#[cfg(all(not(feature = "blas"), target_arch = "aarch64"))]
fn gemm_nn_fallback(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    let nt = get_num_threads();
    let parallel = nt > 1
        && m >= 2
        && n >= 64
        && m.saturating_mul(k).saturating_mul(n) >= FALLBACK_MIN_MACS;
    if !parallel {
        // SAFETY: single-threaded full-range pass; slices sized by the wrapper.
        unsafe { gemm_nn_rows(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), k, n, 0, m); }
        return;
    }
    let c_send = c.as_mut_ptr() as usize;
    let a_send = a.as_ptr() as usize;
    let b_send = b.as_ptr() as usize;
    // Fixed 16-row output blocks grabbed dynamically.
    const ROWS: usize = 16;
    let n_items = m.div_ceil(ROWS);
    parallel_for_dynamic(n_items, |item| {
        let start = item * ROWS;
        let end = (start + ROWS).min(m);
        if start >= end { return; }
        // SAFETY: items write disjoint output rows [start,end) of c.
        unsafe { gemm_nn_rows(c_send as *mut f32, a_send as *const f32, b_send as *const f32, k, n, start, end); }
    });
}

// ------------------------------------------------------------------------
// INT8 decoder-prefill GEMM entry points (R13-Android, stage 1)
//
// No-BLAS (Android) build only. Route the decoder-prefill projections through
// the resident INT8 weights (half the BF16 byte stream, no f32 scratch matrix)
// instead of the bf16→f32 + NEON f32 fallback GEMM. Enabled by default when
// compiled; `QWEN_ASR_INT8_PREFILL=0` falls back to the f32 path. Desktop/BLAS
// builds never compile any of this — prefill stays on AMX f32 (R12-F2).
// ------------------------------------------------------------------------

/// Whether the INT8 decoder-prefill path is enabled at runtime. This function
/// only exists when the `int8-prefill` cargo feature is compiled in (opt-in,
/// wired into the Flutter Android build only), and in that case defaults **ON**;
/// `QWEN_ASR_INT8_PREFILL=0` is a runtime kill switch. Cached after first read,
/// matching the `QWEN_ASR_VERIFY`/`QWEN_ASR_SIDECAR` knobs.
#[cfg(all(feature = "int8-prefill", not(feature = "blas"), target_arch = "aarch64"))]
pub fn int8_prefill_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("QWEN_ASR_INT8_PREFILL")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Quantize each of `seq_len` activation rows (length `dim`) to INT8 with its
/// own absmax scale, writing into caller-provided scratch. Bit-identical, row
/// by row, to the single-token `quantize_into` (the same per-row absmax used by
/// the decode path), so the prefill GEMM's quantized inputs match the reference.
/// Shared by the INT8 decoder-prefill (stage 1) and encoder (stage 2) paths.
#[cfg(all(any(feature = "int8-prefill", feature = "int8-encoder"), not(feature = "blas"), target_arch = "aarch64"))]
pub(crate) fn quantize_rows_into(
    dst: &mut [i8], scales: &mut [f32], x: &[f32], seq_len: usize, dim: usize,
) {
    debug_assert_eq!(dst.len(), seq_len * dim);
    debug_assert_eq!(scales.len(), seq_len);
    debug_assert_eq!(x.len(), seq_len * dim);
    for p in 0..seq_len {
        let base = p * dim;
        scales[p] = quantize_into(&mut dst[base..base + dim], &x[base..base + dim]);
    }
}

/// Pool-parallel INT8 prefill matvec `y[seq × out] = x @ Wᵀ`. Partitions the
/// `out_dim` weight rows across the persistent pool in fixed 64-row blocks
/// grabbed dynamically (P-cores take more than E-cores); each block streams its
/// weight rows once across all `seq_len` positions. Small GEMMs stay
/// single-threaded (R10 dispatch threshold). `x_int8`/`x_scales` are the
/// per-position quantized inputs.
///
/// # Safety
/// `y` sized `seq_len*out_dim`; `x_int8` sized `seq_len*in_dim`; `x_scales`
/// sized `seq_len`; `w_int8`/`w_scales` cover `out_dim` rows / `in_dim` cols.
#[cfg(all(feature = "int8-prefill", not(feature = "blas"), target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_prefill_matvec(
    y: &mut [f32], x_int8: &[i8], x_scales: &[f32],
    w_int8: *const i8, w_scales: *const f32,
    in_dim: usize, out_dim: usize, seq_len: usize,
) {
    const MIN_COLS: usize = 128;
    let nt = get_num_threads();
    let parallel = nt > 1
        && out_dim >= MIN_COLS
        && seq_len.saturating_mul(in_dim).saturating_mul(out_dim) >= FALLBACK_MIN_MACS;
    if !parallel {
        neon::matvec_int8_prefill_rows(
            y.as_mut_ptr(), x_int8.as_ptr(), x_scales.as_ptr(), w_int8, w_scales,
            in_dim, out_dim, seq_len, 0, out_dim,
        );
        return;
    }
    let y_send = y.as_mut_ptr() as usize;
    let x_send = x_int8.as_ptr() as usize;
    let xs_send = x_scales.as_ptr() as usize;
    let w_send = w_int8 as usize;
    let ws_send = w_scales as usize;
    const ROWS: usize = 64;
    let n_items = out_dim.div_ceil(ROWS);
    parallel_for_dynamic(n_items, |item| {
        let start = item * ROWS;
        let end = (start + ROWS).min(out_dim);
        if start >= end { return; }
        // SAFETY: items write disjoint output-column ranges [start,end).
        unsafe {
            neon::matvec_int8_prefill_rows(
                y_send as *mut f32, x_send as *const i8, xs_send as *const f32,
                w_send as *const i8, ws_send as *const f32,
                in_dim, out_dim, seq_len, start, end,
            );
        }
    });
}

/// Pool-parallel INT8 prefill fused gate_up + SwiGLU `ffn[seq × n_rows]`.
/// Partitions the `n_rows` intermediate rows across the pool (256-row blocks);
/// each block streams its gate/up weight-row pairs once across all positions.
/// `w_int8`/`w_scales` are the resident interleaved gate_up weights.
///
/// # Safety
/// `ffn` sized `seq_len*n_rows`; `x_int8` sized `seq_len*in_dim`; `x_scales`
/// sized `seq_len`; `w_int8`/`w_scales` cover `2*n_rows` rows / `in_dim` cols.
#[cfg(all(feature = "int8-prefill", not(feature = "blas"), target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_prefill_swiglu(
    ffn: &mut [f32], x_int8: &[i8], x_scales: &[f32],
    w_int8: *const i8, w_scales: *const f32,
    in_dim: usize, n_rows: usize, seq_len: usize,
) {
    let nt = get_num_threads();
    let parallel = nt > 1
        && n_rows >= 64
        && seq_len.saturating_mul(in_dim).saturating_mul(2 * n_rows) >= FALLBACK_MIN_MACS;
    if !parallel {
        neon::swiglu_int8_prefill_rows(
            ffn.as_mut_ptr(), x_int8.as_ptr(), x_scales.as_ptr(), w_int8, w_scales,
            in_dim, n_rows, seq_len, 0, n_rows,
        );
        return;
    }
    let ffn_send = ffn.as_mut_ptr() as usize;
    let x_send = x_int8.as_ptr() as usize;
    let xs_send = x_scales.as_ptr() as usize;
    let w_send = w_int8 as usize;
    let ws_send = w_scales as usize;
    const ROWS: usize = 256;
    let n_items = n_rows.div_ceil(ROWS);
    parallel_for_dynamic(n_items, |item| {
        let start = item * ROWS;
        let end = (start + ROWS).min(n_rows);
        if start >= end { return; }
        // SAFETY: items write disjoint intermediate-row ranges [start,end).
        unsafe {
            neon::swiglu_int8_prefill_rows(
                ffn_send as *mut f32, x_send as *const i8, xs_send as *const f32,
                w_send as *const i8, ws_send as *const f32,
                in_dim, n_rows, seq_len, start, end,
            );
        }
    });
}

// ------------------------------------------------------------------------
// INT8 encoder weight-GEMM entry points (R13-Android, stage 2)
//
// No-BLAS (Android) build only. Route the encoder weight projections (conv_out,
// attention q/k/v/o, FFN fc1/fc2, proj1/proj2) through resident INT8 weights
// instead of the f32 `linear`/`linear_accumulate` GEMMs. Activation×activation
// attention GEMMs (QKᵀ, scores·V) stay f32 — no weights to quantize. Enabled by
// default when compiled; `QWEN_ASR_INT8_ENCODER=0` falls back to the f32 path.
// Desktop/BLAS builds never compile any of this. Independent of int8-prefill.
// ------------------------------------------------------------------------

/// Whether the INT8 encoder path is enabled at runtime. Only exists when the
/// `int8-encoder` cargo feature is compiled in (opt-in, wired into the Flutter
/// Android build only), and defaults **ON**; `QWEN_ASR_INT8_ENCODER=0` is a
/// runtime kill switch. Cached after first read, matching `QWEN_ASR_INT8_PREFILL`.
#[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
pub fn int8_encoder_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("QWEN_ASR_INT8_ENCODER")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Quantize an f32 weight matrix to INT8 per-row with absmax scaling. Mirrors
/// [`quantize_bf16_weights_to_int8`] (same per-row absmax → `scale = max/127`,
/// round-clamp) but reads the encoder's already-prepacked f32 weights (which are
/// the exact bf16→f32 widening), so the INT8 result equals quantizing the
/// original bf16 directly. Returns `(int8_data, per_row_scales)`.
#[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
pub fn quantize_f32_weights_to_int8(w: &[f32], out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let mut int8 = vec![0i8; out_dim * in_dim];
    let mut scales = vec![0.0f32; out_dim];
    for r in 0..out_dim {
        scales[r] = quantize_into(&mut int8[r * in_dim..(r + 1) * in_dim], &w[r * in_dim..(r + 1) * in_dim]);
    }
    (int8, scales)
}

/// Pool-parallel INT8 encoder GEMM `y[seq × out] = x @ Wᵀ (+ bias)`, optionally
/// accumulating in place (`y += …`, for the fused wo/fc2 residual adds).
/// Partitions the `out_dim` weight rows across the persistent pool in fixed
/// 64-row blocks; small GEMMs stay single-threaded (R10 dispatch threshold).
/// `x_int8`/`x_scales` are the per-position quantized inputs.
///
/// # Safety
/// `y` sized `seq_len*out_dim`; `x_int8` sized `seq_len*in_dim`; `x_scales`
/// sized `seq_len`; `w_int8`/`w_scales` cover `out_dim` rows / `in_dim` cols;
/// `bias` (if `Some`) covers `out_dim`.
#[cfg(all(feature = "int8-encoder", not(feature = "blas"), target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_encoder_matvec(
    y: &mut [f32], x_int8: &[i8], x_scales: &[f32],
    w_int8: *const i8, w_scales: *const f32, bias: Option<&[f32]>,
    in_dim: usize, out_dim: usize, seq_len: usize, accumulate: bool,
) {
    const MIN_COLS: usize = 128;
    let bias_ptr = bias.map_or(std::ptr::null(), |b| b.as_ptr());
    let nt = get_num_threads();
    let parallel = nt > 1
        && out_dim >= MIN_COLS
        && seq_len.saturating_mul(in_dim).saturating_mul(out_dim) >= FALLBACK_MIN_MACS;
    if !parallel {
        neon::matvec_int8_encoder_rows(
            y.as_mut_ptr(), x_int8.as_ptr(), x_scales.as_ptr(), w_int8, w_scales, bias_ptr,
            in_dim, out_dim, seq_len, 0, out_dim, accumulate,
        );
        return;
    }
    let y_send = y.as_mut_ptr() as usize;
    let x_send = x_int8.as_ptr() as usize;
    let xs_send = x_scales.as_ptr() as usize;
    let w_send = w_int8 as usize;
    let ws_send = w_scales as usize;
    let b_send = bias_ptr as usize;
    const ROWS: usize = 64;
    let n_items = out_dim.div_ceil(ROWS);
    parallel_for_dynamic(n_items, |item| {
        let start = item * ROWS;
        let end = (start + ROWS).min(out_dim);
        if start >= end { return; }
        // SAFETY: items write disjoint output-column ranges [start,end).
        unsafe {
            neon::matvec_int8_encoder_rows(
                y_send as *mut f32, x_send as *const i8, xs_send as *const f32,
                w_send as *const i8, ws_send as *const f32, b_send as *const f32,
                in_dim, out_dim, seq_len, start, end, accumulate,
            );
        }
    });
}

/// C = A @ B (no transpose): `A[M,K]`, `B[K,N]`, `C[M,N]`
pub fn matmul_nn(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        #[cfg(target_arch = "aarch64")]
        { gemm_nn_fallback(c, a, b, m, k, n); }

        #[cfg(not(target_arch = "aarch64"))]
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * b[ki * n + ni];
                }
                c[mi * n + ni] = sum;
            }
        }
    }
}

/// C = A @ B^T: `A[M,K]`, `B[N,K]`, `C[M,N]`
pub fn matmul_t(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), k as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        #[cfg(target_arch = "aarch64")]
        { gemm_nt_fallback(c, a, b, None, m, k, n, false); }

        #[cfg(not(target_arch = "aarch64"))]
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * b[ni * k + ki];
                }
                c[mi * n + ni] = sum;
            }
        }
    }
}

/// Pool-parallel `y[seq,out] = x @ W^T + beta*y (+ b)`: splits the output
/// columns across the persistent thread pool, one BLAS call per slice. Each
/// output element is still a single full-K dot product inside one sgemm call.
/// A lone Accelerate sgemm call runs mostly on the calling thread, leaving the
/// pool workers idle for the whole encoder/prefill GEMM phase; per-thread
/// slices let every pool thread feed the matrix hardware concurrently.
/// Returns false when the problem is too small to win over one direct call.
#[cfg(feature = "blas")]
#[allow(clippy::too_many_arguments)]
fn sgemm_nt_pooled(y: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>,
                   seq_len: usize, in_dim: usize, out_dim: usize, beta: f32) -> bool {
    let nt = get_num_threads();
    // Each slice needs enough columns for an efficient BLAS kernel, and the
    // whole product enough MACs to amortize the pool dispatch.
    const MIN_COLS: usize = 128;
    const MIN_MACS: usize = 1 << 22;
    if nt <= 1 || seq_len < 2 || out_dim < 2 * MIN_COLS
        || seq_len * in_dim * out_dim < MIN_MACS {
        return false;
    }
    let y_send = y.as_mut_ptr() as usize;
    let x_send = x.as_ptr() as usize;
    let w_send = w.as_ptr() as usize;
    let b_send = b.map(|b| b.as_ptr() as usize);
    // Fixed-size 128-column work items, grabbed dynamically so P-cores take more
    // blocks than E-cores. Item boundaries depend only on `out_dim`, never on the
    // thread count, so the column grouping (and thus the BLAS accumulation) is
    // deterministic across runs.
    let n_items = out_dim.div_ceil(MIN_COLS);
    parallel_for_dynamic(n_items, |item| {
        let start = item * MIN_COLS;
        let end = (start + MIN_COLS).min(out_dim);
        // SAFETY: items write disjoint column ranges [start, end) of y.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                seq_len as i32, (end - start) as i32, in_dim as i32,
                1.0, x_send as *const f32, in_dim as i32,
                (w_send as *const f32).add(start * in_dim), in_dim as i32,
                beta, (y_send as *mut f32).add(start), out_dim as i32,
            );
            if let Some(bp) = b_send {
                let bp = bp as *const f32;
                for s in 0..seq_len {
                    let row = (y_send as *mut f32).add(s * out_dim);
                    for o in start..end {
                        *row.add(o) += *bp.add(o);
                    }
                }
            }
        }
    });
    true
}

/// y = x @ W^T + b: `x[seq,in]`, `W[out,in]`, `b[out]`, `y[seq,out]`
pub fn linear(y: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.sgemm);
    #[cfg(feature = "blas")]
    unsafe {
        if sgemm_nt_pooled(y, x, w, b, seq_len, in_dim, out_dim, 0.0) {
            return;
        }
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            seq_len as i32, out_dim as i32, in_dim as i32,
            1.0, x.as_ptr(), in_dim as i32,
            w.as_ptr(), in_dim as i32,
            0.0, y.as_mut_ptr(), out_dim as i32,
        );
        if let Some(b) = b {
            for s in 0..seq_len {
                let row = &mut y[s * out_dim..(s + 1) * out_dim];
                // Contiguous slice iterators: no bounds checks, LLVM auto-vectorizes.
                for (v, &bv) in row.iter_mut().zip(b.iter()) {
                    *v += bv;
                }
            }
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        #[cfg(target_arch = "aarch64")]
        { gemm_nt_fallback(y, x, w, b, seq_len, in_dim, out_dim, false); }

        #[cfg(not(target_arch = "aarch64"))]
        for s in 0..seq_len {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let w_row = &w[o * in_dim..(o + 1) * in_dim];
                let mut sum = b.map_or(0.0, |b| b[o]);
                for i in 0..in_dim {
                    sum += x_row[i] * w_row[i];
                }
                y[s * out_dim + o] = sum;
            }
        }
    }
}

pub fn linear_nobias(y: &mut [f32], x: &[f32], w: &[f32], seq_len: usize, in_dim: usize, out_dim: usize) {
    linear(y, x, w, None, seq_len, in_dim, out_dim);
}

/// y += bias + x @ w.T  (accumulate into existing y, fusing residual add)
pub fn linear_accumulate(y: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.sgemm);
    #[cfg(feature = "blas")]
    unsafe {
        if sgemm_nt_pooled(y, x, w, b, seq_len, in_dim, out_dim, 1.0) {
            return;
        }
        // Add bias to y first (y already has residual)
        if let Some(b) = b {
            for s in 0..seq_len {
                let row = &mut y[s * out_dim..(s + 1) * out_dim];
                for o in 0..out_dim {
                    row[o] += b[o];
                }
            }
        }
        // y = 1.0 * x @ w.T + 1.0 * y  (accumulate matmul into y)
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            seq_len as i32, out_dim as i32, in_dim as i32,
            1.0, x.as_ptr(), in_dim as i32,
            w.as_ptr(), in_dim as i32,
            1.0, y.as_mut_ptr(), out_dim as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        #[cfg(target_arch = "aarch64")]
        { gemm_nt_fallback(y, x, w, b, seq_len, in_dim, out_dim, true); }

        #[cfg(not(target_arch = "aarch64"))]
        for s in 0..seq_len {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let w_row = &w[o * in_dim..(o + 1) * in_dim];
                let mut sum = b.map_or(0.0, |bb| bb[o]);
                for i in 0..in_dim {
                    sum += x_row[i] * w_row[i];
                }
                y[s * out_dim + o] += sum;
            }
        }
    }
}

fn bf16_to_f32_view(src: *const u16, n: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; n];
    let src_slice = unsafe { std::slice::from_raw_parts(src, n) };
    bf16_to_f32_buf(&mut buf, src_slice);
    buf
}

/// Threaded bf16 matvec
fn bf16_matvec_threaded(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim);
        return;
    }

    let y_ptr = y.as_mut_ptr();
    let x_ptr = x.as_ptr();
    let w_ptr = w_bf16;
    let bias_ptr = bias.map(|b| b.as_ptr());

    // SAFETY: Each thread writes to non-overlapping segments of y
    let y_send = y_ptr as usize;
    let x_send = x_ptr as usize;
    let w_send = w_ptr as usize;
    let bias_send = bias_ptr.map(|p| p as usize);

    // Fixed 256-row output blocks, grabbed dynamically. Each output row is a
    // full-K dot product independent of the block split, so the result is
    // bit-identical regardless of scheduling.
    const ROWS: usize = 256;
    let n_items = out_dim.div_ceil(ROWS);
    parallel_for_dynamic(n_items, |item| {
        let start = item * ROWS;
        let end = (start + ROWS).min(out_dim);
        if start >= end { return; }

        let y_local = unsafe { std::slice::from_raw_parts_mut((y_send as *mut f32).add(start), end - start) };
        let x_local = unsafe { std::slice::from_raw_parts(x_send as *const f32, in_dim) };
        let w_local = unsafe { (w_send as *const u16).add(start * in_dim) };
        let bias_local = bias_send.map(|p| unsafe { std::slice::from_raw_parts((p as *const f32).add(start), end - start) });

        bf16_matvec_fused(y_local, x_local, w_local, bias_local, in_dim, end - start);
    });
}

/// Like linear_nobias_bf16 for seq_len=1, but ADDS to the destination: `y[i] += W[i] @ x`.
/// Achieves fused residual add by passing y as its own "bias".
pub fn linear_nobias_bf16_addto(y: &mut [f32], x: &[f32], w_bf16: *const u16, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    // SAFETY: bf16_matvec_fused reads bias[i] before writing y[i], so aliasing y as bias is safe.
    let bias = unsafe { std::slice::from_raw_parts(y.as_ptr(), out_dim) };
    bf16_matvec_threaded(y, x, w_bf16, Some(bias), in_dim, out_dim);
}

pub fn linear_nobias_bf16(y: &mut [f32], x: &[f32], w_bf16: *const u16, seq_len: usize, in_dim: usize, out_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, None, in_dim, out_dim);
        return;
    }
    let w_f32 = bf16_to_f32_view(w_bf16, out_dim * in_dim);
    linear_nobias(y, x, &w_f32, seq_len, in_dim, out_dim);
}

/// Like linear_nobias_bf16 but reuses a caller-provided scratch buffer for bf16→f32 conversion.
/// # Safety
/// Caller must ensure w_bf16 points to at least out_dim * in_dim valid bf16 values.
pub unsafe fn linear_nobias_bf16_scratch(y: &mut [f32], x: &[f32], w_bf16: *const u16, seq_len: usize, in_dim: usize, out_dim: usize, scratch: &mut [f32]) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, None, in_dim, out_dim);
        return;
    }
    let n = out_dim * in_dim;
    let src = unsafe { std::slice::from_raw_parts(w_bf16, n) };
    bf16_to_f32_buf_parallel(&mut scratch[..n], src);
    linear_nobias(y, x, &scratch[..n], seq_len, in_dim, out_dim);
}

thread_local! {
    /// Per-thread gate_up scratch for [`linear_nobias_bf16_swiglu`]. The
    /// single-token BF16 decode path (x86 / no-BLAS / single-thread) calls it
    /// once per layer per decoded token, so a per-call `vec!` would put a heap
    /// allocation on the critical path of every token. Each OS thread —
    /// including `parallel_for_dynamic` workers, which run items one at a time
    /// to completion — gets its own buffer, the same isolation as today's
    /// per-call fresh allocations.
    static SWIGLU_SCRATCH_TLS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Borrow the thread-local gate_up scratch at length `n` (zeroed) and pass it
/// to `f`. Bit-identical to handing `f` a fresh `vec![0.0; n]`.
fn with_swiglu_scratch<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    SWIGLU_SCRATCH_TLS.with(|c| {
        let mut buf = c.borrow_mut();
        buf.clear();
        buf.resize(n, 0.0);
        f(&mut buf)
    })
}

/// Fused gate_up matvec + SwiGLU for single-token decode.
/// Computes: `ffn_out[j] = silu(gate[j]) * up[j]` where gate/up come from interleaved gate_up_fused matvec.
/// Keeps gate_up output in L1 cache for the SwiGLU operation.
pub fn linear_nobias_bf16_swiglu(
    ffn_out: &mut [f32],
    x: &[f32],
    gate_up_bf16: *const u16,
    in_dim: usize,
    intermediate: usize,
) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    let n_threads = get_num_threads();

    if n_threads <= 1 {
        // Single-threaded: compute gate_up, then SwiGLU inline
        with_swiglu_scratch(2 * intermediate, |gate_buf| {
            bf16_matvec_fused(gate_buf, x, gate_up_bf16, None, in_dim, 2 * intermediate);
            for j in 0..intermediate {
                let g = gate_buf[2 * j];
                let u = gate_buf[2 * j + 1];
                ffn_out[j] = g / (1.0 + (-g).exp()) * u;
            }
        });
        return;
    }

    let x_ptr = x.as_ptr() as usize;
    let w_ptr = gate_up_bf16 as usize;
    let ffn_ptr = ffn_out.as_mut_ptr() as usize;

    // Fixed 256-row intermediate blocks, grabbed dynamically. Each row is an
    // independent gate/up dot product + SwiGLU, so the result is bit-identical
    // regardless of the block split.
    const ROWS: usize = 256;
    let n_items = intermediate.div_ceil(ROWS);
    parallel_for_dynamic(n_items, |item| {
        let start = item * ROWS;
        let end = (start + ROWS).min(intermediate);
        if start >= end { return; }
        let n_rows = end - start;

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let w_local = unsafe { (w_ptr as *const u16).add(2 * start * in_dim) };

        // Compute gate_up for this chunk (thread-local scratch buffer)
        with_swiglu_scratch(2 * n_rows, |gate_up_local| {
            bf16_matvec_fused(gate_up_local, x_local, w_local, None, in_dim, 2 * n_rows);

            // Apply SwiGLU inline while data is hot in L1
            let ffn_local = unsafe { std::slice::from_raw_parts_mut((ffn_ptr as *mut f32).add(start), n_rows) };
            for j in 0..n_rows {
                let g = gate_up_local[2 * j];
                let u = gate_up_local[2 * j + 1];
                ffn_local[j] = g / (1.0 + (-g).exp()) * u;
            }
        });
    });
}

/// Compute output rows `[start, end)` of an INT8 matvec (`y = W @ x`, optional
/// fused bias) using the pre-quantized input `x_int8`. Slice entry point of the
/// fused decode region (R5-B): each worker calls it with its own row range.
/// `y_ptr`/`bias_ptr` may alias (fused residual add), in which case each thread
/// owns a disjoint row range so reads-before-writes are per-row and safe.
/// No-op when `start >= end`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)] // hot kernel entry point; params mirror the SIMD call
pub(crate) unsafe fn int8_matvec_range(
    y_ptr: *mut f32, x_int8: *const i8, x_scale: f32,
    w_int8: *const i8, w_scales: *const f32,
    bias_ptr: Option<*const f32>,
    in_dim: usize, start: usize, end: usize,
) {
    if start >= end { return; }
    let n = end - start;
    let y_local = std::slice::from_raw_parts_mut(y_ptr.add(start), n);
    let w_local = w_int8.add(start * in_dim);
    let w_scales_local = std::slice::from_raw_parts(w_scales.add(start), n);
    let bias_local = bias_ptr.map(|p| std::slice::from_raw_parts(p.add(start), n));
    neon::matvec_int8(y_local, x_int8, x_scale, w_local, w_scales_local, bias_local, in_dim, n);
}

/// Compute the `[start, end)` slice (over the concatenated `q|k|v` output rows,
/// total `q_dim + 2*kv_dim`) of the fused INT8 QKV projection. Slice entry
/// point of the fused decode region (R5-B).
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_qkv_range(
    q_ptr: *mut f32, k_ptr: *mut f32, v_ptr: *mut f32,
    x_int8: *const i8, x_scale: f32,
    wq: *const i8, wq_scales: *const f32,
    wk: *const i8, wk_scales: *const f32,
    wv: *const i8, wv_scales: *const f32,
    in_dim: usize, q_dim: usize, kv_dim: usize,
    start: usize, end: usize,
) {
    if start >= end { return; }
    let total_dim = q_dim + 2 * kv_dim;
    let q_end = q_dim;
    let k_end = q_end + kv_dim;

    // Q range
    if start < q_end {
        let s = start;
        let e = end.min(q_end);
        if s < e {
            let y = std::slice::from_raw_parts_mut(q_ptr.add(s), e - s);
            let sc = std::slice::from_raw_parts(wq_scales.add(s), e - s);
            neon::matvec_int8(y, x_int8, x_scale, wq.add(s * in_dim), sc, None, in_dim, e - s);
        }
    }
    // K range
    if start < k_end && end > q_end {
        let s = start.max(q_end) - q_end;
        let e = end.min(k_end) - q_end;
        if s < e {
            let y = std::slice::from_raw_parts_mut(k_ptr.add(s), e - s);
            let sc = std::slice::from_raw_parts(wk_scales.add(s), e - s);
            neon::matvec_int8(y, x_int8, x_scale, wk.add(s * in_dim), sc, None, in_dim, e - s);
        }
    }
    // V range
    if end > k_end {
        let s = start.max(k_end) - k_end;
        let e = end.min(total_dim) - k_end;
        if s < e {
            let y = std::slice::from_raw_parts_mut(v_ptr.add(s), e - s);
            let sc = std::slice::from_raw_parts(wv_scales.add(s), e - s);
            neon::matvec_int8(y, x_int8, x_scale, wv.add(s * in_dim), sc, None, in_dim, e - s);
        }
    }
}

/// Compute intermediate rows `[start, end)` of the fused INT8 gate_up + SwiGLU
/// projection. Slice entry point of the fused decode region (R5-B). `x_int8`
/// is the already-quantized input of length `in_dim`. `scratch` (len >=
/// `2 * (end - start)`) holds the gate_up matvec output; it is caller-provided
/// (allocated once per thread per decode step) because this kernel runs 28×
/// per decoded token inside the fused region — a per-call `vec!` would put
/// `28 × nt` heap allocations on the critical path of every token.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)] // hot kernel entry point; params mirror the SIMD call
pub(crate) unsafe fn int8_swiglu_range(
    ffn_ptr: *mut f32, x_int8: *const i8, x_scale: f32,
    w_int8: *const i8, w_scales: *const f32,
    in_dim: usize, start: usize, end: usize,
    scratch: &mut [f32],
) {
    if start >= end { return; }
    let n_rows = end - start;
    let w_local = w_int8.add(2 * start * in_dim);
    let w_scales_local = std::slice::from_raw_parts(w_scales.add(2 * start), 2 * n_rows);
    // Zero-init preserved from the original `vec![0.0; ..]`; the matvec writes
    // every element, so this only keeps the proven behavior identical.
    let gate_up_local = &mut scratch[..2 * n_rows];
    gate_up_local.fill(0.0);
    neon::matvec_int8(gate_up_local, x_int8, x_scale, w_local, w_scales_local, None, in_dim, 2 * n_rows);
    let ffn_local = std::slice::from_raw_parts_mut(ffn_ptr.add(start), n_rows);
    for j in 0..n_rows {
        let g = gate_up_local[2 * j];
        let u = gate_up_local[2 * j + 1];
        ffn_local[j] = g / (1.0 + (-g).exp()) * u;
    }
}

// ========================================================================
// Batched (lockstep) INT8 decode range wrappers (R12-E2)
//
// Slice entry points for the lockstep scheduler (stage 3): each partitions the
// SAME weight-row range `[start, end)` as the single-session kernels, but
// applies each streamed weight row to all `b` sessions before moving to the
// next row (loop order: rows outer, sessions inner). Each session's output is
// bit-identical to a standalone single-session `int8_*_range` call — see the
// exactness tests in `tests` below. `b` is a small runtime value (2–6); the
// per-session pointer arrays are stack-resident (`MAX_BATCH`).
// ========================================================================

/// Max lockstep batch size (per-session pointer arrays are stack-allocated).
#[cfg(target_arch = "aarch64")]
pub(crate) const MAX_BATCH: usize = 8;

/// Batched analogue of [`int8_matvec_range`]: rows `[start, end)` for all `b`
/// sessions. `y[bi]`/`x_int8[bi]`/`bias[bi]` are per-session base pointers.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_matvec_range_batched(
    b: usize,
    y: &[*mut f32], x_int8: &[*const i8], x_scale: &[f32],
    w_int8: *const i8, w_scales: *const f32,
    bias: Option<&[*const f32]>,
    in_dim: usize, start: usize, end: usize,
) {
    if start >= end { return; }
    let n = end - start;
    let mut y_off = [std::ptr::null_mut::<f32>(); MAX_BATCH];
    let mut bias_off = [std::ptr::null::<f32>(); MAX_BATCH];
    for bi in 0..b {
        y_off[bi] = y[bi].add(start);
    }
    if let Some(bs) = bias {
        for bi in 0..b { bias_off[bi] = bs[bi].add(start); }
    }
    let w_local = w_int8.add(start * in_dim);
    let w_scales_local = std::slice::from_raw_parts(w_scales.add(start), n);
    neon::matvec_int8_batched(
        b, &y_off[..b], &x_int8[..b], &x_scale[..b], w_local, w_scales_local,
        if bias.is_some() { Some(&bias_off[..b]) } else { None }, in_dim, n,
    );
}

/// Batched analogue of [`int8_qkv_range`]: the `[start, end)` slice over the
/// concatenated `q|k|v` rows, for all `b` sessions.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_qkv_range_batched(
    b: usize,
    q: &[*mut f32], k: &[*mut f32], v: &[*mut f32],
    x_int8: &[*const i8], x_scale: &[f32],
    wq: *const i8, wq_scales: *const f32,
    wk: *const i8, wk_scales: *const f32,
    wv: *const i8, wv_scales: *const f32,
    in_dim: usize, q_dim: usize, kv_dim: usize,
    start: usize, end: usize,
) {
    if start >= end { return; }
    let total_dim = q_dim + 2 * kv_dim;
    let q_end = q_dim;
    let k_end = q_end + kv_dim;
    // Q range
    if start < q_end {
        let s = start;
        let e = end.min(q_end);
        if s < e {
            int8_matvec_range_batched(b, q, x_int8, x_scale, wq, wq_scales, None, in_dim, s, e);
        }
    }
    // K range
    if start < k_end && end > q_end {
        let s = start.max(q_end) - q_end;
        let e = end.min(k_end) - q_end;
        if s < e {
            int8_matvec_range_batched(b, k, x_int8, x_scale, wk, wk_scales, None, in_dim, s, e);
        }
    }
    // V range
    if end > k_end {
        let s = start.max(k_end) - k_end;
        let e = end.min(total_dim) - k_end;
        if s < e {
            int8_matvec_range_batched(b, v, x_int8, x_scale, wv, wv_scales, None, in_dim, s, e);
        }
    }
}

/// Batched analogue of [`int8_swiglu_range`]: intermediate rows `[start, end)`
/// for all `b` sessions.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_swiglu_range_batched(
    b: usize,
    ffn: &[*mut f32], x_int8: &[*const i8], x_scale: &[f32],
    w_int8: *const i8, w_scales: *const f32,
    in_dim: usize, start: usize, end: usize,
) {
    if start >= end { return; }
    let n_rows = end - start;
    let mut ffn_off = [std::ptr::null_mut::<f32>(); MAX_BATCH];
    for bi in 0..b {
        ffn_off[bi] = ffn[bi].add(start);
    }
    let w_local = w_int8.add(2 * start * in_dim);
    let w_scales_local = std::slice::from_raw_parts(w_scales.add(2 * start), 2 * n_rows);
    neon::swiglu_int8_batched(
        b, &ffn_off[..b], &x_int8[..b], &x_scale[..b], w_local, w_scales_local, in_dim, n_rows,
    );
}

// ========================================================================
// 2D Convolution (im2col + BLAS sgemm)
// ========================================================================

#[allow(clippy::too_many_arguments)]
fn im2col(input: &[f32], cols: &mut [f32], c_in: usize, h_in: usize, w_in: usize,
          kh: usize, kw: usize, stride: usize, padding: usize, h_out: usize, w_out: usize) {
    let col_len = h_out * w_out;
    for ic in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let col_row = (ic * kh + ki) * kw + kj;
                for oh in 0..h_out {
                    let ih = oh * stride + ki;
                    let ih = ih as isize - padding as isize;
                    for ow in 0..w_out {
                        let iw = ow * stride + kj;
                        let iw = iw as isize - padding as isize;
                        let val = if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                            input[ic * h_in * w_in + ih as usize * w_in + iw as usize]
                        } else {
                            0.0
                        };
                        cols[col_row * col_len + oh * w_out + ow] = val;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_with_cols(out: &mut [f32], input: &[f32], weight: &[f32], bias: Option<&[f32]>,
                        cols: &mut Vec<f32>,
                        c_in: usize, c_out: usize, h_in: usize, w_in: usize,
                        kh: usize, kw: usize, stride: usize, padding: usize) {
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;
    cols.resize(patch_size * spatial_out, 0.0);
    conv2d_impl(out, input, weight, bias, cols, c_in, c_out, h_in, w_in, kh, kw, stride, padding);
}

#[allow(clippy::too_many_arguments)]
fn conv2d_impl(out: &mut [f32], input: &[f32], weight: &[f32], bias: Option<&[f32]>,
               cols: &mut [f32],
               c_in: usize, c_out: usize, h_in: usize, w_in: usize,
               kh: usize, kw: usize, stride: usize, padding: usize) {
    let _pg = ProfileGuard::new(&PROF.conv2d_op);
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;
    let cols = &mut cols[..patch_size * spatial_out];

    // Thread im2col across col_rows (each row is independent)
    let n_threads = get_num_threads();
    if n_threads > 1 && patch_size >= 16 {
        let input_ptr = input.as_ptr() as usize;
        let cols_ptr = cols.as_mut_ptr() as usize;
        // Fixed 32-row (patch-row) blocks grabbed dynamically; each col_row is an
        // independent im2col scatter.
        const PROW: usize = 32;
        let n_items = patch_size.div_ceil(PROW);
        parallel_for_dynamic(n_items, |item| {
            let start = item * PROW;
            let end = (start + PROW).min(patch_size);
            if start >= end { return; }
            for col_row in start..end {
                let ic = col_row / (kh * kw);
                let rem = col_row % (kh * kw);
                let ki = rem / kw;
                let kj = rem % kw;
                for oh in 0..h_out {
                    let ih = (oh * stride + ki) as isize - padding as isize;
                    for ow in 0..w_out {
                        let iw = (ow * stride + kj) as isize - padding as isize;
                        let val = if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                            unsafe { *(input_ptr as *const f32).add(ic * h_in * w_in + ih as usize * w_in + iw as usize) }
                        } else {
                            0.0
                        };
                        unsafe { *(cols_ptr as *mut f32).add(col_row * spatial_out + oh * w_out + ow) = val; }
                    }
                }
            }
        });
    } else {
        im2col(input, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);
    }

    // GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out]
    // Split output channels across the pool (one sgemm slice per thread) so
    // the workers that just finished im2col also share the GEMM instead of
    // idling behind a single main-thread BLAS call.
    #[cfg(feature = "blas")]
    unsafe {
        if n_threads > 1 && c_out >= 2 * n_threads
            && c_out * patch_size * spatial_out >= (1 << 22) {
            let out_send = out.as_mut_ptr() as usize;
            let w_send = weight.as_ptr() as usize;
            let cols_send = cols.as_ptr() as usize;
            let bias_send = bias.map(|b| b.as_ptr() as usize);
            // Fixed 32-channel output blocks grabbed dynamically; item boundaries
            // depend only on `c_out`, so BLAS grouping is deterministic.
            const COUT: usize = 32;
            let n_items = c_out.div_ceil(COUT);
            parallel_for_dynamic(n_items, |item| {
                let start = item * COUT;
                let end = (start + COUT).min(c_out);
                if start >= end { return; }
                // SAFETY: items write disjoint row ranges [start, end) of out.
                cblas_sgemm(
                    CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                    (end - start) as i32, spatial_out as i32, patch_size as i32,
                    1.0, (w_send as *const f32).add(start * patch_size), patch_size as i32,
                    cols_send as *const f32, spatial_out as i32,
                    0.0, (out_send as *mut f32).add(start * spatial_out), spatial_out as i32,
                );
                if let Some(bp) = bias_send {
                    let bp = bp as *const f32;
                    for oc in start..end {
                        let b = *bp.add(oc);
                        let row = (out_send as *mut f32).add(oc * spatial_out);
                        for s in 0..spatial_out {
                            *row.add(s) += b;
                        }
                    }
                }
            });
            return;
        }
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            c_out as i32, spatial_out as i32, patch_size as i32,
            1.0, weight.as_ptr(), patch_size as i32,
            cols.as_ptr(), spatial_out as i32,
            0.0, out.as_mut_ptr(), spatial_out as i32,
        );
    }

    #[cfg(not(feature = "blas"))]
    {
        // out[c_out, spatial_out] = weight[c_out, patch_size] @ cols[patch_size, spatial_out]
        #[cfg(target_arch = "aarch64")]
        { gemm_nn_fallback(out, weight, cols, c_out, patch_size, spatial_out); }

        #[cfg(not(target_arch = "aarch64"))]
        for oc in 0..c_out {
            for s in 0..spatial_out {
                let mut sum = 0.0f32;
                for p in 0..patch_size {
                    sum += weight[oc * patch_size + p] * cols[p * spatial_out + s];
                }
                out[oc * spatial_out + s] = sum;
            }
        }
    }

    if let Some(bias) = bias {
        for oc in 0..c_out {
            let b = bias[oc];
            let row = &mut out[oc * spatial_out..(oc + 1) * spatial_out];
            // Contiguous broadcast add over a slice: LLVM auto-vectorizes.
            for v in row.iter_mut() {
                *v += b;
            }
        }
    }
}

// ========================================================================
// Normalization
// ========================================================================

pub fn layer_norm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32],
                  seq_len: usize, hidden: usize, eps: f32) {
    let _pg = ProfileGuard::new(&PROF.layer_norm);
    for s in 0..seq_len {
        let x_row = &x[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::layer_norm_row(out_row, x_row, weight, bias, hidden, eps); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::layer_norm_row(out_row, x_row, weight, bias, hidden, eps); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mean: f32 = x_row.iter().sum::<f32>() / hidden as f32;

            let var: f32 = x_row.iter().map(|&v| {
                let d = v - mean;
                d * d
            }).sum::<f32>() / hidden as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..hidden {
                out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
            }
        }
    }
}

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], seq_len: usize, hidden: usize, eps: f32) {
    let _pg = ProfileGuard::new(&PROF.rms_norm);
    for s in 0..seq_len {
        let x_row = &x[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::rms_norm_row(out_row, x_row, weight, hidden, eps); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::rms_norm_row(out_row, x_row, weight, hidden, eps); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let sum_sq: f32 = x_row.iter().map(|&v| v * v).sum();
            let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                out_row[i] = x_row[i] * rms_inv * weight[i];
            }
        }
    }
}

pub fn rms_norm_per_head(x: &mut [f32], weight: &[f32], seq_len: usize, n_heads: usize, head_dim: usize, eps: f32) {
    let hidden = n_heads * head_dim;
    for s in 0..seq_len {
        for h in 0..n_heads {
            let off = s * hidden + h * head_dim;

            #[cfg(target_arch = "aarch64")]
            {
                let vec = &mut x[off..off + head_dim];
                unsafe { neon::rms_norm_inplace(vec, weight, head_dim, eps); }
                continue;
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                let vec = &mut x[off..off + head_dim];
                let sum_sq: f32 = vec.iter().map(|&v| v * v).sum();
                let rms_inv = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
                for d in 0..head_dim {
                    vec[d] = vec[d] * rms_inv * weight[d];
                }
            }
        }
    }
}

// ========================================================================
// Activation Functions
// ========================================================================

pub fn silu(x: &mut [f32], n: usize) {
    for val in x.iter_mut().take(n) {
        *val = *val / (1.0 + (-*val).exp());
    }
}

pub fn gelu(x: &mut [f32], n: usize) {
    let _pg = ProfileGuard::new(&PROF.gelu);
    let n_threads = get_num_threads();
    // Thread GELU for large buffers (encoder FFN: ~320K floats)
    if n_threads > 1 && n > 4096 {
        let x_ptr = x.as_mut_ptr() as usize;
        // Fixed 4096-element blocks grabbed dynamically; GELU is element-wise so
        // the result is bit-identical regardless of the block split.
        const ITEM: usize = 4096;
        let n_items = n.div_ceil(ITEM);
        parallel_for_dynamic(n_items, |item| {
            let start = item * ITEM;
            let end = (start + ITEM).min(n);
            if start >= end { return; }
            let x_local = unsafe { std::slice::from_raw_parts_mut((x_ptr as *mut f32).add(start), end - start) };
            #[cfg(target_arch = "aarch64")]
            unsafe { neon::gelu_inplace(x_local, end - start); }
            #[cfg(target_arch = "x86_64")]
            unsafe { avx::gelu_inplace(x_local, end - start); }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            for i in 0..(end - start) {
                let val = x_local[i];
                let x3 = val * val * val;
                let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
                x_local[i] = 0.5 * val * (1.0 + inner.tanh());
            }
        });
        return;
    }
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::gelu_inplace(x, n); } }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::gelu_inplace(x, n); } }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in 0..n {
        let val = x[i];
        let x3 = val * val * val;
        let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
        x[i] = 0.5 * val * (1.0 + inner.tanh());
    }
}

/// SwiGLU in place: `gate[i] = silu(gate[i]) * up[i]`. Reads `up` and overwrites
/// `gate` with the final activation, avoiding an extra output buffer.
pub fn swiglu_separate_inplace(gate: &mut [f32], up: &[f32], seq_len: usize, intermediate: usize) {
    let _pg = ProfileGuard::new(&PROF.swiglu);
    let total = seq_len * intermediate;
    let n_threads = get_num_threads();

    if n_threads > 1 && total > 4096 {
        let gate_ptr = gate.as_mut_ptr() as usize;
        let up_ptr = up.as_ptr() as usize;
        // Fixed 4096-element blocks grabbed dynamically; SwiGLU is element-wise so
        // the result is bit-identical regardless of the block split.
        const ITEM: usize = 4096;
        let n_items = total.div_ceil(ITEM);
        parallel_for_dynamic(n_items, |item| {
            let start = item * ITEM;
            let end = (start + ITEM).min(total);
            if start >= end { return; }
            let g = unsafe { std::slice::from_raw_parts_mut((gate_ptr as *mut f32).add(start), end - start) };
            let u = unsafe { std::slice::from_raw_parts((up_ptr as *const f32).add(start), end - start) };
            for j in 0..(end - start) {
                let gv = g[j];
                g[j] = gv / (1.0 + (-gv).exp()) * u[j];
            }
        });
        return;
    }

    for j in 0..total {
        let gv = gate[j];
        gate[j] = gv / (1.0 + (-gv).exp()) * up[j];
    }
}

pub fn softmax(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for val in row.iter_mut().take(cols) {
            *val -= max_val;
        }

        #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
        {
            let n = cols as i32;
            unsafe { vvexpf(row.as_mut_ptr(), row.as_ptr(), &n); }
        }
        #[cfg(not(all(feature = "vdsp", target_vendor = "apple")))]
        {
            #[cfg(target_arch = "aarch64")]
            { unsafe { neon::exp_inplace(row); } }

            #[cfg(target_arch = "x86_64")]
            { unsafe { avx::exp_inplace(row); } }

            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            for c in 0..cols {
                row[c] = row[c].exp();
            }
        }

        let mut sum = 0.0f32;
        for val in row.iter().take(cols) {
            sum += val;
        }
        let inv_sum = 1.0 / sum;
        for val in row.iter_mut().take(cols) {
            *val *= inv_sum;
        }
    }
}

// ========================================================================
// Attention Operations
// ========================================================================

#[allow(clippy::too_many_arguments)]
fn bidirectional_attention_heads(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
                                  n_heads: usize, head_dim: usize, scale: f32,
                                  window_starts: &[i32], n_windows: usize,
                                  head_start: usize, head_end: usize) {
    let hidden = n_heads * head_dim;

    for h in head_start..head_end {
        for w in 0..n_windows {
            let ws = window_starts[w] as usize;
            let we = window_starts[w + 1] as usize;

            for i in ws..we {
                let q_off = i * hidden + h * head_dim;
                let q_row = &q[q_off..q_off + head_dim];
                let o_row = &mut out[i * hidden + h * head_dim..i * hidden + h * head_dim + head_dim];

                let mut max_score = -1e30f32;
                let mut sum_exp = 0.0f32;
                for val in o_row.iter_mut().take(head_dim) { *val = 0.0; }

                for j in ws..we {
                    let k_off = j * hidden + h * head_dim;
                    let v_off = j * hidden + h * head_dim;
                    let k_row = &k[k_off..k_off + head_dim];
                    let v_row = &v[v_off..v_off + head_dim];

                    let score = dot_f32(q_row, k_row, head_dim) * scale;

                    if score > max_score {
                        let correction = (max_score - score).exp();
                        sum_exp = sum_exp * correction + 1.0;
                        vec_scale_add(o_row, v_row, correction, head_dim);
                        max_score = score;
                    } else {
                        let wt = (score - max_score).exp();
                        sum_exp += wt;
                        vec_axpy_inplace(o_row, v_row, wt, head_dim);
                    }
                }

                if sum_exp > 0.0 {
                    let inv_sum = 1.0 / sum_exp;
                    vec_scale_inplace(o_row, inv_sum, head_dim);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bidirectional_attention(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
                               seq: usize, n_heads: usize, head_dim: usize, scale: f32,
                               window_starts: &[i32], n_windows: usize) {
    let _pg = ProfileGuard::new(&PROF.attention_bidir);
    let n_threads = get_num_threads();
    let hidden = n_heads * head_dim;

    if n_threads > 1 && n_heads >= 2 {
        let out_ptr = out.as_mut_ptr() as usize;
        let q_ptr = q.as_ptr() as usize;
        let k_ptr = k.as_ptr() as usize;
        let v_ptr = v.as_ptr() as usize;
        let ws_ptr = window_starts.as_ptr() as usize;

        // One work item per attention head, grabbed dynamically. Each head is an
        // independent softmax over the same K/V, so per-head results are
        // bit-identical regardless of which core runs which head.
        parallel_for_dynamic(n_heads, |h| {
            let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, seq * hidden) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, seq * hidden) };
            let k_local = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, seq * hidden) };
            let v_local = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, seq * hidden) };
            let ws_local = unsafe { std::slice::from_raw_parts(ws_ptr as *const i32, n_windows + 1) };

            bidirectional_attention_heads(out_local, q_local, k_local, v_local,
                                         n_heads, head_dim, scale,
                                         ws_local, n_windows, h, h + 1);
        });
        return;
    }

    bidirectional_attention_heads(out, q, k, v, n_heads, head_dim, scale,
                                 window_starts, n_windows, 0, n_heads);
}

/// Two-pass causal attention using BLAS sgemm with head-contiguous KV cache.
/// K/V layout: `[head][pos][head_dim]` — each head's data is contiguous across positions.
///
/// Single-token (seq_q=1): online softmax with NEON dot products — avoids BLAS overhead,
/// scores allocation, and fuses all 3 passes into a single scan over KV positions.
///
/// Multi-token (seq_q>1): 3-pass BLAS sgemm approach.
/// One query row of the GQA-paired online-softmax attention scan. Iterates by
/// KV-head group so each K/V row is loaded once and shared by all
/// `heads_per_kv` query heads of that group. Each query head's per-`j`
/// operation sequence is identical to an unpaired per-head scan — only the
/// interleaving between heads changes — so results are bit-identical. Writes
/// heads `[head_start, head_end)` of `out_row`; `q_row`/`out_row` hold one
/// token's `n_heads * head_dim` floats. No BLAS dependency (dot_f32 /
/// vec_scale_add / vec_axpy_inplace only); shared by the blas single-token
/// path and the non-blas fallback.
#[allow(clippy::too_many_arguments)]
#[inline]
fn paired_attention_row(out_row: &mut [f32], q_row: &[f32],
                        k_base: *const f32, v_base: *const f32,
                        head_stride: usize, k_end: usize,
                        heads_per_kv: usize, head_dim: usize, scale: f32,
                        head_start: usize, head_end: usize) {
    // Per-head online-softmax state, indexed within the current group.
    // heads_per_kv is bounded by the model's head count; keep on stack.
    const MAX_GROUP: usize = 32;
    debug_assert!(heads_per_kv <= MAX_GROUP);
    let mut max_score = [-1e30f32; MAX_GROUP];
    let mut sum_exp = [0.0f32; MAX_GROUP];

    let mut h = head_start;
    while h < head_end {
        let kv_h = h / heads_per_kv;
        // Query heads of this KV group, clamped to [head_start, head_end).
        let group_start = h;
        let group_end = ((kv_h + 1) * heads_per_kv).min(head_end);
        let group_size = group_end - group_start;

        let o_span = &mut out_row[group_start * head_dim..group_end * head_dim];

        for g in 0..group_size {
            max_score[g] = -1e30f32;
            sum_exp[g] = 0.0f32;
        }
        for val in o_span.iter_mut() { *val = 0.0; }

        let k_head = unsafe { k_base.add(kv_h * head_stride) };
        let v_head = unsafe { v_base.add(kv_h * head_stride) };

        // Single pass over KV positions; each row loaded once for the group.
        for j in 0..k_end {
            let k_row = unsafe { std::slice::from_raw_parts(k_head.add(j * head_dim), head_dim) };
            let v_row = unsafe { std::slice::from_raw_parts(v_head.add(j * head_dim), head_dim) };

            for g in 0..group_size {
                let q_off = (group_start + g) * head_dim;
                let q_head = &q_row[q_off..q_off + head_dim];
                let o_row = &mut o_span[g * head_dim..g * head_dim + head_dim];

                let score = dot_f32(q_head, k_row, head_dim) * scale;

                if score > max_score[g] {
                    let correction = (max_score[g] - score).exp();
                    sum_exp[g] = sum_exp[g] * correction + 1.0;
                    vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score[g] = score;
                } else {
                    let wt = (score - max_score[g]).exp();
                    sum_exp[g] += wt;
                    vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }
        }

        for g in 0..group_size {
            if sum_exp[g] > 0.0 {
                let inv_sum = 1.0 / sum_exp[g];
                let o_row = &mut o_span[g * head_dim..g * head_dim + head_dim];
                vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }

        h = group_end;
    }
}

#[cfg(feature = "blas")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_attention_heads(out: &mut [f32], q: &[f32],
                           k_base: *const f32, v_base: *const f32,
                           head_stride: usize,
                           seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                           head_dim: usize, scale: f32, q_offset: usize,
                           head_start: usize, head_end: usize) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;

    // Single-token path: GQA-paired online softmax without allocation or BLAS.
    if seq_q == 1 {
        let k_end = (q_offset + 1).min(seq_k);
        paired_attention_row(out, q, k_base, v_base, head_stride, k_end,
                             heads_per_kv, head_dim, scale, head_start, head_end);
        return;
    }

    // Multi-token path: batched per-head GEMMs.
    // Per head: S[seq_q, seq_k] = scale * Q_h @ K_h^T, then causal-masked
    // row softmax, then O[seq_q, head_dim] = S @ V_h. This replaces the
    // 2*seq_q tiny (N=1) BLAS calls per head with two real GEMMs.
    let mut scores = vec![0.0f32; seq_q * seq_k];

    for h in head_start..head_end {
        let kv_h = h / heads_per_kv;
        let k_head = unsafe { k_base.add(kv_h * head_stride) };
        let v_head = unsafe { v_base.add(kv_h * head_stride) };

        // S = scale * Q_h[seq_q, head_dim] @ K_h[seq_k, head_dim]^T.
        // Q_h rows are strided by q_hidden inside `q`; K_h is contiguous.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                seq_q as i32, seq_k as i32, head_dim as i32,
                scale,
                q.as_ptr().add(h * head_dim), q_hidden as i32,
                k_head, head_dim as i32,
                0.0,
                scores.as_mut_ptr(), seq_k as i32,
            );
        }

        // Causal-masked row softmax: query i attends keys 0..=(q_offset+i).
        for i in 0..seq_q {
            let k_end = (q_offset + i + 1).min(seq_k);
            let row = &mut scores[i * seq_k..i * seq_k + seq_k];
            if k_end == 0 {
                for v in row.iter_mut() { *v = 0.0; }
                continue;
            }

            let mut max_s = row[0];
            for &s in &row[1..k_end] { if s > max_s { max_s = s; } }
            for s in &mut row[..k_end] { *s -= max_s; }

            #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
            {
                let n = k_end as i32;
                unsafe { vvexpf(row.as_mut_ptr(), row.as_ptr(), &n); }
            }
            #[cfg(not(all(feature = "vdsp", target_vendor = "apple")))]
            {
                for s in &mut row[..k_end] { *s = s.exp(); }
            }

            let mut sum_exp = 0.0f32;
            for &s in &row[..k_end] { sum_exp += s; }
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for s in &mut row[..k_end] { *s *= inv; }
            }
            // Zero the masked (future) keys so the O = S @ V GEMM ignores them.
            for s in &mut row[k_end..seq_k] { *s = 0.0; }
        }

        // O[seq_q, head_dim] = S[seq_q, seq_k] @ V_h[seq_k, head_dim].
        // O rows are strided by q_hidden inside `out`.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                seq_q as i32, head_dim as i32, seq_k as i32,
                1.0,
                scores.as_ptr(), seq_k as i32,
                v_head, head_dim as i32,
                0.0,
                out.as_mut_ptr().add(h * head_dim), q_hidden as i32,
            );
        }
    }
}

/// Fallback: online softmax causal attention (no BLAS), head-contiguous KV
/// layout. Runs the GQA-paired scan row by row for any `seq_q`.
#[cfg(not(feature = "blas"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_attention_heads(out: &mut [f32], q: &[f32],
                           k_base: *const f32, v_base: *const f32,
                           head_stride: usize,
                           seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                           head_dim: usize, scale: f32, q_offset: usize,
                           head_start: usize, head_end: usize) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;

    for i in 0..seq_q {
        let global_pos = q_offset + i;
        let k_end = (global_pos + 1).min(seq_k);
        let row_base = i * q_hidden;
        paired_attention_row(&mut out[row_base..row_base + q_hidden],
                             &q[row_base..row_base + q_hidden],
                             k_base, v_base, head_stride, k_end,
                             heads_per_kv, head_dim, scale, head_start, head_end);
    }
}

/// Partition attention work for one worker. Single-token (`seq_q == 1`) splits
/// by KV-head **group** so a GQA-paired group is never split across threads
/// (each K/V row is loaded once per group); multi-token splits by query head.
/// Returns this worker's `[head_start, head_end)`, or `None` if empty. Shared
/// by the threaded public kernel and the fused decode region.
#[inline]
pub(crate) fn attn_head_range(tid: usize, nt: usize, seq_q: usize, n_heads: usize, n_kv_heads: usize)
    -> Option<(usize, usize)>
{
    if seq_q == 1 && n_heads.is_multiple_of(n_kv_heads) {
        let heads_per_kv = n_heads / n_kv_heads;
        let chunk = n_kv_heads.div_ceil(nt);
        let g0 = tid * chunk;
        let g1 = (g0 + chunk).min(n_kv_heads);
        if g0 >= g1 { return None; }
        Some((g0 * heads_per_kv, g1 * heads_per_kv))
    } else {
        let chunk = n_heads.div_ceil(nt);
        let h0 = tid * chunk;
        let h1 = (h0 + chunk).min(n_heads);
        if h0 >= h1 { return None; }
        Some((h0, h1))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn causal_attention(out: &mut [f32], q: &[f32],
                         k_base: *const f32, v_base: *const f32,
                         head_stride: usize,
                         seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                         head_dim: usize, scale: f32, q_offset: usize) {
    let _pg = ProfileGuard::new(&PROF.attention_causal);
    let n_threads = get_num_threads();
    if n_threads > 1 && n_heads >= 2 {
        let out_ptr = out.as_mut_ptr() as usize;
        let q_ptr = q.as_ptr() as usize;
        let k_ptr = k_base as usize;
        let v_ptr = v_base as usize;
        let q_hidden = n_heads * head_dim;

        parallel_for(|tid, nt| {
            let (h0, h1) = match attn_head_range(tid, nt, seq_q, n_heads, n_kv_heads) {
                Some(r) => r,
                None => return,
            };

            let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, seq_q * q_hidden) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, seq_q * q_hidden) };

            causal_attention_heads(out_local, q_local,
                                   k_ptr as *const f32, v_ptr as *const f32,
                                   head_stride,
                                   seq_q, seq_k, n_heads, n_kv_heads,
                                   head_dim, scale, q_offset, h0, h1);
        });
        return;
    }

    causal_attention_heads(out, q, k_base, v_base, head_stride,
                            seq_q, seq_k, n_heads, n_kv_heads,
                            head_dim, scale, q_offset, 0, n_heads);
}

// ========================================================================
// Position Embeddings
// ========================================================================

pub fn sinusoidal_pe(pe: &mut [f32], n_pos: usize, d_model: usize) {
    let half = d_model / 2;
    let log_timescale = (10000.0f32).ln() / (half - 1) as f32;

    for p in 0..n_pos {
        let row = &mut pe[p * d_model..(p + 1) * d_model];
        for d in 0..half {
            let inv_timescale = (-(d as f32) * log_timescale).exp();
            let angle = p as f32 * inv_timescale;
            row[d] = angle.sin();
            row[half + d] = angle.cos();
        }
    }
}

pub fn apply_rope_neox(x: &mut [f32], cos_vals: &[f32], sin_vals: &[f32],
                        seq: usize, n_heads: usize, head_dim: usize) {
    let _pg = ProfileGuard::new(&PROF.rope);
    let half = head_dim / 2;
    let hidden = n_heads * head_dim;

    for s in 0..seq {
        let c = &cos_vals[s * head_dim..];
        let sn = &sin_vals[s * head_dim..];

        for h in 0..n_heads {
            let base = s * hidden + h * head_dim;
            let vec = &mut x[base..base + head_dim];

            #[cfg(target_arch = "aarch64")]
            {
                let mut d = 0usize;
                while d + 4 <= half {
                    unsafe {
                        use core::arch::aarch64::*;
                        let x1 = vld1q_f32(vec.as_ptr().add(d));
                        let x2 = vld1q_f32(vec.as_ptr().add(half + d));
                        let cv = vld1q_f32(c.as_ptr().add(d));
                        let sv = vld1q_f32(sn.as_ptr().add(d));
                        // vec[d] = x1*cos - x2*sin
                        let new1 = vfmsq_f32(vmulq_f32(x1, cv), x2, sv);
                        // vec[half+d] = x2*cos + x1*sin (cos[half+d]==cos[d])
                        let new2 = vfmaq_f32(vmulq_f32(x2, cv), x1, sv);
                        vst1q_f32(vec.as_mut_ptr().add(d), new1);
                        vst1q_f32(vec.as_mut_ptr().add(half + d), new2);
                    }
                    d += 4;
                }
                while d < half {
                    let x1 = vec[d];
                    let x2 = vec[half + d];
                    vec[d]        = x1 * c[d] - x2 * sn[d];
                    vec[half + d] = x2 * c[d] + x1 * sn[d];
                    d += 1;
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                for d in 0..half {
                    let x1 = vec[d];
                    let x2 = vec[half + d];
                    vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                    vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
                }
            }
        }
    }
}

/// Streaming argmax: finds argmax(W_bf16 @ x) without materializing full logits.
/// Quantize x (f32) to int8 with absmax scaling. Returns (x_int8, scale).
pub fn quantize_f32_to_int8(x: &[f32]) -> (Vec<i8>, f32) {
    let mut int8 = vec![0i8; x.len()];
    let scale = quantize_into(&mut int8, x);
    (int8, scale)
}

thread_local! {
    /// Per-thread quantization scratch for [`with_quantized_int8`]. The INT8
    /// lm_head argmax runs once per decoded token, so a per-call `vec!` would
    /// put a heap allocation on the critical path of every token; each OS
    /// thread gets its own buffer, the same isolation as a fresh allocation.
    static QUANT_TLS: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Quantize x and pass (buffer, scale) to f, reusing thread-local storage.
/// Bit-identical to [`quantize_f32_to_int8`] — same absmax scaling math, only
/// the destination buffer is reused instead of freshly allocated.
pub fn with_quantized_int8<R>(x: &[f32], f: impl FnOnce(&[i8], f32) -> R) -> R {
    QUANT_TLS.with(|c| {
        let mut v = c.borrow_mut();
        v.clear();
        v.resize(x.len(), 0i8);
        let scale = quantize_into(&mut v, x);
        f(&v, scale)
    })
}

#[cfg(target_arch = "aarch64")]
thread_local! {
    /// Per-thread pool of quantization buffers for
    /// [`with_quantized_int8_batch`]. The batched INT8 lm_head argmax scores
    /// all `b` session inputs against the weights once per decoded token; the
    /// `b` quantized buffers must stay alive simultaneously, so one reused
    /// buffer per session (indexed by session, not by call) gives the same
    /// isolation as today's fresh per-session `vec!`s.
    static QUANT_BATCH_TLS: std::cell::RefCell<Vec<Vec<i8>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Quantize every session input in `xs` into reused thread-local buffers and
/// pass (buffers, scales) to `f`. Bit-identical to mapping
/// [`quantize_f32_to_int8`] over `xs` — same absmax scaling math, only the
/// destination buffers are reused instead of freshly allocated per token.
#[cfg(target_arch = "aarch64")]
pub fn with_quantized_int8_batch<R>(xs: &[&[f32]], f: impl FnOnce(&[Vec<i8>], &[f32]) -> R) -> R {
    QUANT_BATCH_TLS.with(|c| {
        let mut bufs = c.borrow_mut();
        bufs.resize_with(xs.len(), Vec::new);
        let mut scales = Vec::with_capacity(xs.len());
        for (buf, x) in bufs.iter_mut().zip(xs.iter()) {
            buf.clear();
            buf.resize(x.len(), 0i8);
            scales.push(quantize_into(buf, x));
        }
        f(&bufs, &scales)
    })
}

/// Quantize `x` into a caller-provided `dst` buffer with absmax scaling and
/// return the scale. Bit-identical to [`quantize_f32_to_int8`] but writes into
/// reusable storage (used by the fused decode region to avoid per-stage
/// allocation). `dst.len()` must equal `x.len()`.
pub fn quantize_into(dst: &mut [i8], x: &[f32]) -> f32 {
    debug_assert_eq!(dst.len(), x.len());
    let mut max_abs = 0.0f32;
    for &v in x { max_abs = max_abs.max(v.abs()); }
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 127.0 / max_abs.max(1e-10);
    for (d, &v) in dst.iter_mut().zip(x.iter()) {
        *d = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Quantize BF16 weights to INT8 per-row. Returns (int8_data, per_row_scales).
///
/// # Safety
/// `w_bf16` must point to at least `out_dim * in_dim` readable `u16` (BF16)
/// values that stay valid for the duration of the call.
pub unsafe fn quantize_bf16_weights_to_int8(w_bf16: *const u16, out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    #[cfg(target_arch = "aarch64")]
    unsafe { neon::quantize_bf16_to_int8(w_bf16, out_dim, in_dim) }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut int8_data = vec![0i8; out_dim * in_dim];
        let mut scales = vec![0.0f32; out_dim];
        let src = unsafe { std::slice::from_raw_parts(w_bf16, out_dim * in_dim) };
        for row in 0..out_dim {
            let mut max_abs = 0.0f32;
            for k in 0..in_dim {
                let v = f32::from_bits((src[row * in_dim + k] as u32) << 16).abs();
                if v > max_abs { max_abs = v; }
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 127.0 / max_abs.max(1e-10);
            scales[row] = scale;
            for k in 0..in_dim {
                let v = f32::from_bits((src[row * in_dim + k] as u32) << 16);
                int8_data[row * in_dim + k] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        (int8_data, scales)
    }
}

/// INT8 threaded argmax: find argmax(x @ W.T) using INT8 quantized weights.
pub fn argmax_matvec_int8(x: &[f32], w_int8: &[i8], w_scales: &[f32], in_dim: usize, out_dim: usize) -> usize {
    with_quantized_int8(x, |x_int8, x_scale| {
    let n_threads = get_num_threads();
    #[cfg(target_arch = "aarch64")]
    {
        if n_threads <= 1 {
            let (best, _) = unsafe {
                neon::argmax_int8_range(x_int8.as_ptr(), x_scale, w_int8.as_ptr(), w_scales, in_dim, 0, out_dim)
            };
            return best;
        }

        let mut best_indices = [0usize; MAX_THREADS];
        let mut best_vals = [-1e30f32; MAX_THREADS];

        let x_int8_ptr = x_int8.as_ptr() as usize;
        let w_int8_ptr = w_int8.as_ptr() as usize;
        let w_scales_ptr = w_scales.as_ptr() as usize;
        let bi_ptr = best_indices.as_mut_ptr() as usize;
        let bv_ptr = best_vals.as_mut_ptr() as usize;

        parallel_for(|tid, nt| {
            let chunk = out_dim.div_ceil(nt);
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end {
                unsafe {
                    *(bv_ptr as *mut f32).add(tid) = -1e30;
                    *(bi_ptr as *mut usize).add(tid) = 0;
                }
                return;
            }

            let w_scales_local = unsafe { std::slice::from_raw_parts(w_scales_ptr as *const f32, out_dim) };
            let (best, best_val) = unsafe {
                neon::argmax_int8_range(x_int8_ptr as *const i8, x_scale, w_int8_ptr as *const i8, w_scales_local, in_dim, start, end)
            };
            unsafe {
                *(bi_ptr as *mut usize).add(tid) = best;
                *(bv_ptr as *mut f32).add(tid) = best_val;
            }
        });

        let mut best = best_indices[0];
        let mut best_val = best_vals[0];
        for i in 1..n_threads {
            if best_vals[i] > best_val {
                best_val = best_vals[i];
                best = best_indices[i];
            }
        }
        best
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Fallback: use f32 computation
        let _ = (x, w_int8, w_scales, in_dim, out_dim, n_threads, x_int8, x_scale);
        unimplemented!("INT8 argmax only implemented for aarch64")
    }
    })
}

/// Batched INT8 lm_head argmax (R12-E2): stream the (~155 MB) lm_head weights
/// ONCE and score all `b` session hidden states against them, returning each
/// session's argmax token. Bit-identical to `b` independent
/// [`argmax_matvec_int8`] calls when `in_dim` is a multiple of 16 (the lm_head
/// case, in_dim = dec_hidden = 1024) — the amortization lever for the lockstep
/// decode's per-token vocabulary scoring. Row partitioning + strict-`>` reduce
/// give the same index-stable tie-break as the single-session kernel.
#[cfg(target_arch = "aarch64")]
pub fn argmax_matvec_int8_batched(
    xs: &[&[f32]], w_int8: &[i8], w_scales: &[f32], in_dim: usize, out_dim: usize,
) -> Vec<usize> {
    let b = xs.len();
    with_quantized_int8_batch(xs, |x_int8_bufs, x_scales| {
    let x_ptrs: Vec<*const i8> = x_int8_bufs.iter().map(|v| v.as_ptr()).collect();
    let n_threads = get_num_threads();

    if n_threads <= 1 {
        let mut best = vec![0usize; b];
        let mut best_val = vec![-1e30f32; b];
        unsafe {
            neon::argmax_int8_batched(
                b, &mut best, &mut best_val, &x_ptrs, x_scales,
                w_int8.as_ptr(), w_scales, in_dim, 0, out_dim,
            );
        }
        return best;
    }

    // Per-thread × per-session best, laid out row-major [tid * b + bi].
    let mut best_all = vec![0usize; n_threads * b];
    let mut best_val_all = vec![-1e30f32; n_threads * b];
    let x_ptrs_addr = x_ptrs.as_ptr() as usize;
    let x_scales_addr = x_scales.as_ptr() as usize;
    let w_int8_ptr = w_int8.as_ptr() as usize;
    let w_scales_ptr = w_scales.as_ptr() as usize;
    let bi_ptr = best_all.as_mut_ptr() as usize;
    let bv_ptr = best_val_all.as_mut_ptr() as usize;

    parallel_for(|tid, nt| {
        let chunk = out_dim.div_ceil(nt);
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        let base = tid * b;
        let best = unsafe { std::slice::from_raw_parts_mut((bi_ptr as *mut usize).add(base), b) };
        let best_val = unsafe { std::slice::from_raw_parts_mut((bv_ptr as *mut f32).add(base), b) };
        if start >= end { return; }
        let x_ptrs = unsafe { std::slice::from_raw_parts(x_ptrs_addr as *const *const i8, b) };
        let x_scales = unsafe { std::slice::from_raw_parts(x_scales_addr as *const f32, b) };
        let w_scales = unsafe { std::slice::from_raw_parts(w_scales_ptr as *const f32, out_dim) };
        unsafe {
            neon::argmax_int8_batched(
                b, best, best_val, x_ptrs, x_scales,
                w_int8_ptr as *const i8, w_scales, in_dim, start, end,
            );
        }
    });

    // Reduce per session: threads own increasing contiguous row ranges, so a
    // strict-`>` reduce keeps the lowest-index winner on ties.
    let mut best = vec![0usize; b];
    let mut best_val = vec![-1e30f32; b];
    for tid in 0..n_threads {
        for bi in 0..b {
            let v = best_val_all[tid * b + bi];
            if v > best_val[bi] {
                best_val[bi] = v;
                best[bi] = best_all[tid * b + bi];
            }
        }
    }
    best
    })
}

pub fn argmax_matvec_bf16(x: &[f32], w_bf16: *const u16, in_dim: usize, out_dim: usize) -> usize {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        let (best, _) = argmax_bf16_range(x, w_bf16, in_dim, 0, out_dim);
        return best;
    }

    let mut best_indices = vec![0usize; n_threads];
    let mut best_vals = vec![-1e30f32; n_threads];

    let x_ptr = x.as_ptr() as usize;
    let w_ptr = w_bf16 as usize;
    let bi_ptr = best_indices.as_mut_ptr() as usize;
    let bv_ptr = best_vals.as_mut_ptr() as usize;

    parallel_for(|tid, nt| {
        let chunk = out_dim.div_ceil(nt);
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end {
            unsafe {
                *(bv_ptr as *mut f32).add(tid) = -1e30;
                *(bi_ptr as *mut usize).add(tid) = 0;
            }
            return;
        }

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let (best, best_val) = argmax_bf16_range(x_local, w_ptr as *const u16, in_dim, start, end);
        unsafe {
            *(bi_ptr as *mut usize).add(tid) = best;
            *(bv_ptr as *mut f32).add(tid) = best_val;
        }
    });

    let mut best = best_indices[0];
    let mut best_val = best_vals[0];
    for i in 1..n_threads {
        if best_vals[i] > best_val {
            best_val = best_vals[i];
            best = best_indices[i];
        }
    }
    best
}

// ========================================================================
// Batched decode kernel exactness tests (R12-E2)
//
// Assert every batched kernel's per-session output is BIT-IDENTICAL (exact f32
// equality, not tolerance) to the corresponding single-session kernel, across
// odd sizes / tails / partial row ranges / batch sizes. This is the correctness
// argument for lockstep decode: per-session token sequences cannot depend on
// batch composition if the kernels are exact.
// ========================================================================
#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // Deterministic LCG helpers so tests are reproducible without deps.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self { Rng(seed) }
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn i8(&mut self) -> i8 { ((self.next_u32() % 255) as i32 - 127) as i8 }
        fn f32_pm1(&mut self) -> f32 { (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0 }
    }

    fn gen_w(rng: &mut Rng, rows: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>) {
        let w: Vec<i8> = (0..rows * in_dim).map(|_| rng.i8()).collect();
        let s: Vec<f32> = (0..rows).map(|_| 0.001 + rng.f32_pm1().abs() * 0.02).collect();
        (w, s)
    }

    // Per-session quantized inputs (realistic: quantize f32 activations).
    fn gen_inputs(rng: &mut Rng, b: usize, in_dim: usize) -> (Vec<Vec<i8>>, Vec<f32>) {
        let mut q = Vec::new();
        let mut sc = Vec::new();
        for _ in 0..b {
            let x: Vec<f32> = (0..in_dim).map(|_| rng.f32_pm1() * 3.0).collect();
            let (xi, s) = quantize_f32_to_int8(&x);
            q.push(xi);
            sc.push(s);
        }
        (q, sc)
    }

    #[test]
    fn batched_matvec_matches_single() {
        let mut rng = Rng::new(0xC0FFEE);
        for &in_dim in &[16usize, 32, 48, 64, 17, 33, 1024] {
            for &out_dim in &[1usize, 2, 7, 16, 31] {
                for &b in &[1usize, 2, 3, 5, 6] {
                    for use_bias in [false, true] {
                        let (w, ws) = gen_w(&mut rng, out_dim, in_dim);
                        let (xi, xs) = gen_inputs(&mut rng, b, in_dim);
                        let bias: Vec<Vec<f32>> = (0..b)
                            .map(|_| (0..out_dim).map(|_| rng.f32_pm1()).collect())
                            .collect();
                        // partial range: middle slice + full
                        for &(s, e) in &[(0usize, out_dim), (0, out_dim / 2), (out_dim / 3, out_dim)] {
                            if s >= e { continue; }
                            let xi_ptrs: Vec<*const i8> = xi.iter().map(|v| v.as_ptr()).collect();
                            let bias_ptrs: Vec<*const f32> = bias.iter().map(|v| v.as_ptr()).collect();

                            // Reference: single-session int8_matvec_range per session.
                            let mut y_ref: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; out_dim]).collect();
                            for j in 0..b {
                                unsafe {
                                    int8_matvec_range(
                                        y_ref[j].as_mut_ptr(), xi[j].as_ptr(), xs[j],
                                        w.as_ptr(), ws.as_ptr(),
                                        if use_bias { Some(bias[j].as_ptr()) } else { None },
                                        in_dim, s, e,
                                    );
                                }
                            }
                            // Batched.
                            let mut y_bat: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; out_dim]).collect();
                            let y_ptrs: Vec<*mut f32> = y_bat.iter_mut().map(|v| v.as_mut_ptr()).collect();
                            unsafe {
                                int8_matvec_range_batched(
                                    b, &y_ptrs, &xi_ptrs, &xs, w.as_ptr(), ws.as_ptr(),
                                    if use_bias { Some(&bias_ptrs) } else { None },
                                    in_dim, s, e,
                                );
                            }
                            for j in 0..b {
                                assert_eq!(y_ref[j], y_bat[j],
                                    "matvec mismatch in_dim={in_dim} out_dim={out_dim} b={b} bias={use_bias} range={s}..{e} sess={j}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn batched_swiglu_matches_single() {
        let mut rng = Rng::new(0x5019_1000);
        for &in_dim in &[16usize, 32, 48, 17, 1024] {
            for &n_rows in &[1usize, 3, 8, 15] {
                for &b in &[1usize, 2, 4, 6] {
                    // gate_up has 2*n_rows weight rows (interleaved gate/up).
                    let (w, ws) = gen_w(&mut rng, 2 * n_rows, in_dim);
                    let (xi, xs) = gen_inputs(&mut rng, b, in_dim);
                    for &(s, e) in &[(0usize, n_rows), (0, n_rows / 2), (n_rows / 3, n_rows)] {
                        if s >= e { continue; }
                        let xi_ptrs: Vec<*const i8> = xi.iter().map(|v| v.as_ptr()).collect();
                        let mut ff_ref: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; n_rows]).collect();
                        let mut scratch = vec![0.0f32; 2 * n_rows];
                        for j in 0..b {
                            unsafe {
                                int8_swiglu_range(
                                    ff_ref[j].as_mut_ptr(), xi[j].as_ptr(), xs[j],
                                    w.as_ptr(), ws.as_ptr(), in_dim, s, e,
                                    &mut scratch,
                                );
                            }
                        }
                        let mut ff_bat: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; n_rows]).collect();
                        let ff_ptrs: Vec<*mut f32> = ff_bat.iter_mut().map(|v| v.as_mut_ptr()).collect();
                        unsafe {
                            int8_swiglu_range_batched(
                                b, &ff_ptrs, &xi_ptrs, &xs, w.as_ptr(), ws.as_ptr(), in_dim, s, e,
                            );
                        }
                        for j in 0..b {
                            assert_eq!(ff_ref[j], ff_bat[j],
                                "swiglu mismatch in_dim={in_dim} n_rows={n_rows} b={b} range={s}..{e} sess={j}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn batched_qkv_matches_single() {
        let mut rng = Rng::new(0xA11CE);
        let in_dim = 1024usize;
        for &(q_dim, kv_dim) in &[(2048usize, 1024usize), (32, 16), (48, 16)] {
            for &b in &[1usize, 2, 4, 6] {
                let total = q_dim + 2 * kv_dim;
                let (wq, wqs) = gen_w(&mut rng, q_dim, in_dim);
                let (wk, wks) = gen_w(&mut rng, kv_dim, in_dim);
                let (wv, wvs) = gen_w(&mut rng, kv_dim, in_dim);
                let (xi, xs) = gen_inputs(&mut rng, b, in_dim);
                let xi_ptrs: Vec<*const i8> = xi.iter().map(|v| v.as_ptr()).collect();
                for &(s, e) in &[(0usize, total), (0, q_dim + 3), (q_dim, k_end(q_dim, kv_dim) + 2), (k_end(q_dim, kv_dim), total)] {
                    if s >= e { continue; }
                    let mut q_ref: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; q_dim]).collect();
                    let mut k_ref: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; kv_dim]).collect();
                    let mut v_ref: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; kv_dim]).collect();
                    for j in 0..b {
                        unsafe {
                            int8_qkv_range(
                                q_ref[j].as_mut_ptr(), k_ref[j].as_mut_ptr(), v_ref[j].as_mut_ptr(),
                                xi[j].as_ptr(), xs[j],
                                wq.as_ptr(), wqs.as_ptr(), wk.as_ptr(), wks.as_ptr(),
                                wv.as_ptr(), wvs.as_ptr(), in_dim, q_dim, kv_dim, s, e,
                            );
                        }
                    }
                    let mut q_bat: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; q_dim]).collect();
                    let mut k_bat: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; kv_dim]).collect();
                    let mut v_bat: Vec<Vec<f32>> = (0..b).map(|_| vec![0.0f32; kv_dim]).collect();
                    let qp: Vec<*mut f32> = q_bat.iter_mut().map(|v| v.as_mut_ptr()).collect();
                    let kp: Vec<*mut f32> = k_bat.iter_mut().map(|v| v.as_mut_ptr()).collect();
                    let vp: Vec<*mut f32> = v_bat.iter_mut().map(|v| v.as_mut_ptr()).collect();
                    unsafe {
                        int8_qkv_range_batched(
                            b, &qp, &kp, &vp, &xi_ptrs, &xs,
                            wq.as_ptr(), wqs.as_ptr(), wk.as_ptr(), wks.as_ptr(),
                            wv.as_ptr(), wvs.as_ptr(), in_dim, q_dim, kv_dim, s, e,
                        );
                    }
                    for j in 0..b {
                        assert_eq!(q_ref[j], q_bat[j], "qkv q mismatch qd={q_dim} kvd={kv_dim} b={b} range={s}..{e} sess={j}");
                        assert_eq!(k_ref[j], k_bat[j], "qkv k mismatch qd={q_dim} kvd={kv_dim} b={b} range={s}..{e} sess={j}");
                        assert_eq!(v_ref[j], v_bat[j], "qkv v mismatch qd={q_dim} kvd={kv_dim} b={b} range={s}..{e} sess={j}");
                    }
                }
            }
        }
    }

    fn k_end(q_dim: usize, kv_dim: usize) -> usize { q_dim + kv_dim }

    #[test]
    fn batched_argmax_matches_single() {
        // lm_head is always tail-free (in_dim = dec_hidden = 1024), so restrict
        // exactness to multiple-of-16 in_dim where the combined-tail form is
        // byte-identical to argmax_int8_range.
        let mut rng = Rng::new(0xF00D5);
        for &in_dim in &[16usize, 32, 1024] {
            for &out_dim in &[1usize, 4, 17, 128, 257] {
                for &b in &[1usize, 2, 3, 5] {
                    let (w, ws) = gen_w(&mut rng, out_dim, in_dim);
                    let (xi, xs) = gen_inputs(&mut rng, b, in_dim);
                    let xi_ptrs: Vec<*const i8> = xi.iter().map(|v| v.as_ptr()).collect();

                    // Reference: neon single-session argmax over the full range.
                    let mut ref_best = vec![0usize; b];
                    for j in 0..b {
                        let (best, _) = unsafe {
                            neon::argmax_int8_range(xi[j].as_ptr(), xs[j], w.as_ptr(), &ws, in_dim, 0, out_dim)
                        };
                        ref_best[j] = best;
                    }

                    // Batched neon core (single range).
                    let mut best = vec![0usize; b];
                    let mut best_val = vec![-1e30f32; b];
                    unsafe {
                        neon::argmax_int8_batched(b, &mut best, &mut best_val, &xi_ptrs, &xs, w.as_ptr(), &ws, in_dim, 0, out_dim);
                    }
                    assert_eq!(ref_best, best, "argmax core mismatch in_dim={in_dim} out_dim={out_dim} b={b}");
                }
            }
        }
    }

    #[test]
    fn batched_argmax_toplevel_matches_single() {
        // Exercise the threaded top-level entry against argmax_matvec_int8.
        // Pin to 1 thread to avoid racing the global pool with parallel tests.
        let saved = get_num_threads();
        set_threads(1);
        let mut rng = Rng::new(0x1234ABCD);
        let in_dim = 1024usize;
        for &out_dim in &[128usize, 1000, 4096] {
            for &b in &[1usize, 2, 4] {
                let (w, ws) = gen_w(&mut rng, out_dim, in_dim);
                let xs_f: Vec<Vec<f32>> = (0..b)
                    .map(|_| (0..in_dim).map(|_| rng.f32_pm1() * 3.0).collect())
                    .collect();
                let refs: Vec<usize> = xs_f.iter()
                    .map(|x| argmax_matvec_int8(x, &w, &ws, in_dim, out_dim))
                    .collect();
                let xs_ref: Vec<&[f32]> = xs_f.iter().map(|v| v.as_slice()).collect();
                let bat = argmax_matvec_int8_batched(&xs_ref, &w, &ws, in_dim, out_dim);
                assert_eq!(refs, bat, "argmax toplevel mismatch out_dim={out_dim} b={b}");
            }
        }
        set_threads(saved);
    }

    // ====================================================================
    // R13-Android INT8 decoder-prefill exactness (no-BLAS build only).
    //
    // Each Y[position][out_row] of the multi-row prefill GEMM must be BIT-
    // IDENTICAL (exact f32 equality) to the trusted single-token
    // `int8_matvec_range` / `int8_swiglu_range` applied to that position's
    // quantized activation. Integer SDOT sums are order-exact and the float
    // combine is byte-for-byte the same, so exact equality is achievable and
    // required. Exercised across odd in/out dims + tails, and across both the
    // single-threaded and pool-parallel dispatch paths.
    // ====================================================================

    #[cfg(all(feature = "int8-prefill", not(feature = "blas")))]
    fn prefill_matvec_check(seq_len: usize, in_dim: usize, out_dim: usize, seed: u64) {
        let mut rng = Rng::new(seed);
        let (w, ws) = gen_w(&mut rng, out_dim, in_dim);
        let (xi, xs) = gen_inputs(&mut rng, seq_len, in_dim);
        // Flatten per-position quantized inputs into a contiguous [seq*in] buffer.
        let mut xq = vec![0i8; seq_len * in_dim];
        for p in 0..seq_len {
            xq[p * in_dim..(p + 1) * in_dim].copy_from_slice(&xi[p]);
        }
        // Reference: single-token int8_matvec_range for each position.
        let mut y_ref = vec![0.0f32; seq_len * out_dim];
        for p in 0..seq_len {
            unsafe {
                int8_matvec_range(
                    y_ref[p * out_dim..].as_mut_ptr(), xi[p].as_ptr(), xs[p],
                    w.as_ptr(), ws.as_ptr(), None, in_dim, 0, out_dim,
                );
            }
        }
        // Prefill GEMM.
        let mut y = vec![0.0f32; seq_len * out_dim];
        unsafe {
            int8_prefill_matvec(&mut y, &xq, &xs, w.as_ptr(), ws.as_ptr(), in_dim, out_dim, seq_len);
        }
        assert_eq!(y, y_ref, "prefill matvec mismatch seq={seq_len} in={in_dim} out={out_dim}");
    }

    /// The pool-parallel dispatch just calls `matvec_int8_prefill_rows` over a
    /// partition of the `out_dim` rows; assert that any disjoint row split writes
    /// the same `Y` as the full-range pass (so the dynamic 64-row split is exact).
    /// Runs on the caller thread only — the global pool is a singleton unsafe to
    /// dispatch concurrently from multiple test threads, so tests never invoke it.
    #[cfg(all(feature = "int8-prefill", not(feature = "blas")))]
    fn prefill_matvec_split_check(seq_len: usize, in_dim: usize, out_dim: usize, seed: u64) {
        let mut rng = Rng::new(seed);
        let (w, ws) = gen_w(&mut rng, out_dim, in_dim);
        let (xi, xs) = gen_inputs(&mut rng, seq_len, in_dim);
        let mut xq = vec![0i8; seq_len * in_dim];
        for p in 0..seq_len {
            xq[p * in_dim..(p + 1) * in_dim].copy_from_slice(&xi[p]);
        }
        let mut y_full = vec![0.0f32; seq_len * out_dim];
        let mut y_split = vec![0.0f32; seq_len * out_dim];
        unsafe {
            neon::matvec_int8_prefill_rows(
                y_full.as_mut_ptr(), xq.as_ptr(), xs.as_ptr(), w.as_ptr(), ws.as_ptr(),
                in_dim, out_dim, seq_len, 0, out_dim,
            );
            let mut start = 0;
            while start < out_dim {
                let end = (start + 64).min(out_dim);
                neon::matvec_int8_prefill_rows(
                    y_split.as_mut_ptr(), xq.as_ptr(), xs.as_ptr(), w.as_ptr(), ws.as_ptr(),
                    in_dim, out_dim, seq_len, start, end,
                );
                start = end;
            }
        }
        assert_eq!(y_full, y_split, "prefill matvec split seq={seq_len} in={in_dim} out={out_dim}");
    }

    #[cfg(all(feature = "int8-prefill", not(feature = "blas")))]
    #[test]
    fn int8_prefill_matvec_matches_single() {
        // Serial full-range exactness vs int8_matvec_range, across odd sizes.
        let mut seed = 0x0DEF_ACED_u64;
        for &in_dim in &[16usize, 32, 48, 17, 33, 1024] {
            for &out_dim in &[1usize, 2, 7, 16, 31, 128] {
                for &seq_len in &[1usize, 2, 5, 8, 33] {
                    seed = seed.wrapping_add(0x9E37_79B9);
                    prefill_matvec_check(seq_len, in_dim, out_dim, seed);
                }
            }
        }
        // Row-split equivalence (models the dynamic 64-row pool partition).
        for &(seq_len, in_dim, out_dim) in
            &[(64usize, 1024usize, 2048usize), (48, 1024, 1024), (96, 3072, 1024), (5, 64, 200)]
        {
            prefill_matvec_split_check(seq_len, in_dim, out_dim, 0xD00D_5EED ^ (seq_len as u64));
        }
    }

    #[cfg(all(feature = "int8-prefill", not(feature = "blas")))]
    fn prefill_swiglu_check(seq_len: usize, in_dim: usize, n_rows: usize, seed: u64) {
        let mut rng = Rng::new(seed);
        // gate_up: 2*n_rows interleaved gate/up weight rows.
        let (w, ws) = gen_w(&mut rng, 2 * n_rows, in_dim);
        let (xi, xs) = gen_inputs(&mut rng, seq_len, in_dim);
        let mut xq = vec![0i8; seq_len * in_dim];
        for p in 0..seq_len {
            xq[p * in_dim..(p + 1) * in_dim].copy_from_slice(&xi[p]);
        }
        let mut ff_ref = vec![0.0f32; seq_len * n_rows];
        let mut scratch = vec![0.0f32; 2 * n_rows];
        for p in 0..seq_len {
            unsafe {
                int8_swiglu_range(
                    ff_ref[p * n_rows..].as_mut_ptr(), xi[p].as_ptr(), xs[p],
                    w.as_ptr(), ws.as_ptr(), in_dim, 0, n_rows,
                    &mut scratch,
                );
            }
        }
        let mut ff = vec![0.0f32; seq_len * n_rows];
        unsafe {
            int8_prefill_swiglu(&mut ff, &xq, &xs, w.as_ptr(), ws.as_ptr(), in_dim, n_rows, seq_len);
        }
        assert_eq!(ff, ff_ref, "prefill swiglu mismatch seq={seq_len} in={in_dim} n_rows={n_rows}");
    }

    /// Row-split equivalence for the fused swiglu prefill (256-row pool blocks).
    #[cfg(all(feature = "int8-prefill", not(feature = "blas")))]
    fn prefill_swiglu_split_check(seq_len: usize, in_dim: usize, n_rows: usize, seed: u64) {
        let mut rng = Rng::new(seed);
        let (w, ws) = gen_w(&mut rng, 2 * n_rows, in_dim);
        let (xi, xs) = gen_inputs(&mut rng, seq_len, in_dim);
        let mut xq = vec![0i8; seq_len * in_dim];
        for p in 0..seq_len {
            xq[p * in_dim..(p + 1) * in_dim].copy_from_slice(&xi[p]);
        }
        let mut ff_full = vec![0.0f32; seq_len * n_rows];
        let mut ff_split = vec![0.0f32; seq_len * n_rows];
        unsafe {
            neon::swiglu_int8_prefill_rows(
                ff_full.as_mut_ptr(), xq.as_ptr(), xs.as_ptr(), w.as_ptr(), ws.as_ptr(),
                in_dim, n_rows, seq_len, 0, n_rows,
            );
            let mut start = 0;
            while start < n_rows {
                let end = (start + 256).min(n_rows);
                neon::swiglu_int8_prefill_rows(
                    ff_split.as_mut_ptr(), xq.as_ptr(), xs.as_ptr(), w.as_ptr(), ws.as_ptr(),
                    in_dim, n_rows, seq_len, start, end,
                );
                start = end;
            }
        }
        assert_eq!(ff_full, ff_split, "prefill swiglu split seq={seq_len} in={in_dim} n_rows={n_rows}");
    }

    #[cfg(all(feature = "int8-prefill", not(feature = "blas")))]
    #[test]
    fn int8_prefill_swiglu_matches_single() {
        let mut seed = 0x5019_D00D_u64;
        for &in_dim in &[16usize, 32, 48, 17, 1024] {
            for &n_rows in &[1usize, 3, 8, 15, 64] {
                for &seq_len in &[1usize, 2, 5, 8, 33] {
                    seed = seed.wrapping_add(0x9E37_79B9);
                    prefill_swiglu_check(seq_len, in_dim, n_rows, seed);
                }
            }
        }
        for &(seq_len, in_dim, n_rows) in
            &[(64usize, 1024usize, 3072usize), (48, 1024, 512), (5, 64, 300)]
        {
            prefill_swiglu_split_check(seq_len, in_dim, n_rows, 0xBEEF_F00D ^ (seq_len as u64));
        }
    }

    // ====================================================================
    // R13-Android INT8 encoder weight-GEMM exactness (no-BLAS build only).
    //
    // Each Y[position][out_col] of the batched encoder GEMM must be BIT-
    // IDENTICAL (exact f32 equality) to the trusted single-vector `matvec_int8`
    // (with the same optional bias) applied to that position's quantized
    // activation — and, for the accumulate variant, added onto the same prior
    // Y. Integer SDOT sums are order-exact and the float bias/accumulate is
    // byte-for-byte the same, so exact equality is achievable and required.
    // Exercised across odd in/out dims + tails, bias on/off, accumulate on/off,
    // and across both the single-threaded and pool-row-split dispatch paths.
    // ====================================================================

    #[cfg(all(feature = "int8-encoder", not(feature = "blas")))]
    fn gen_bias(rng: &mut Rng, out_dim: usize) -> Vec<f32> {
        (0..out_dim).map(|_| rng.f32_pm1() * 0.5).collect()
    }

    #[cfg(all(feature = "int8-encoder", not(feature = "blas")))]
    fn encoder_matvec_check(
        seq_len: usize, in_dim: usize, out_dim: usize, with_bias: bool, accumulate: bool, seed: u64,
    ) {
        let mut rng = Rng::new(seed);
        let (w, ws) = gen_w(&mut rng, out_dim, in_dim);
        let (xi, xs) = gen_inputs(&mut rng, seq_len, in_dim);
        let bias = gen_bias(&mut rng, out_dim);
        let bias_opt: Option<&[f32]> = if with_bias { Some(&bias) } else { None };
        // Flatten per-position quantized inputs into a contiguous [seq*in] buffer.
        let mut xq = vec![0i8; seq_len * in_dim];
        for p in 0..seq_len {
            xq[p * in_dim..(p + 1) * in_dim].copy_from_slice(&xi[p]);
        }
        // Prior Y contents (nonzero so the accumulate path is actually exercised).
        let mut y_init = vec![0.0f32; seq_len * out_dim];
        for v in y_init.iter_mut() { *v = rng.f32_pm1() * 4.0; }

        // Reference: single-vector matvec_int8 per position, then f32 combine.
        let mut y_ref = y_init.clone();
        for p in 0..seq_len {
            let mut row = vec![0.0f32; out_dim];
            unsafe {
                neon::matvec_int8(
                    &mut row, xi[p].as_ptr(), xs[p], w.as_ptr(), &ws, bias_opt, in_dim, out_dim,
                );
            }
            for o in 0..out_dim {
                if accumulate { y_ref[p * out_dim + o] += row[o]; }
                else { y_ref[p * out_dim + o] = row[o]; }
            }
        }
        // Batched encoder GEMM.
        let mut y = y_init.clone();
        unsafe {
            int8_encoder_matvec(
                &mut y, &xq, &xs, w.as_ptr(), ws.as_ptr(), bias_opt,
                in_dim, out_dim, seq_len, accumulate,
            );
        }
        assert_eq!(
            y, y_ref,
            "encoder matvec mismatch seq={seq_len} in={in_dim} out={out_dim} bias={with_bias} acc={accumulate}"
        );
    }

    /// Pool-parallel dispatch calls `matvec_int8_encoder_rows` over a partition
    /// of the `out_dim` columns; assert any disjoint split writes the same `Y`
    /// as the full-range pass (models the dynamic 64-row pool partition).
    #[cfg(all(feature = "int8-encoder", not(feature = "blas")))]
    fn encoder_matvec_split_check(
        seq_len: usize, in_dim: usize, out_dim: usize, accumulate: bool, seed: u64,
    ) {
        let mut rng = Rng::new(seed);
        let (w, ws) = gen_w(&mut rng, out_dim, in_dim);
        let (xi, xs) = gen_inputs(&mut rng, seq_len, in_dim);
        let bias = gen_bias(&mut rng, out_dim);
        let mut xq = vec![0i8; seq_len * in_dim];
        for p in 0..seq_len {
            xq[p * in_dim..(p + 1) * in_dim].copy_from_slice(&xi[p]);
        }
        let mut y_init = vec![0.0f32; seq_len * out_dim];
        for v in y_init.iter_mut() { *v = rng.f32_pm1() * 4.0; }
        let mut y_full = y_init.clone();
        let mut y_split = y_init.clone();
        unsafe {
            neon::matvec_int8_encoder_rows(
                y_full.as_mut_ptr(), xq.as_ptr(), xs.as_ptr(), w.as_ptr(), ws.as_ptr(), bias.as_ptr(),
                in_dim, out_dim, seq_len, 0, out_dim, accumulate,
            );
            let mut start = 0;
            while start < out_dim {
                let end = (start + 64).min(out_dim);
                neon::matvec_int8_encoder_rows(
                    y_split.as_mut_ptr(), xq.as_ptr(), xs.as_ptr(), w.as_ptr(), ws.as_ptr(), bias.as_ptr(),
                    in_dim, out_dim, seq_len, start, end, accumulate,
                );
                start = end;
            }
        }
        assert_eq!(
            y_full, y_split,
            "encoder matvec split seq={seq_len} in={in_dim} out={out_dim} acc={accumulate}"
        );
    }

    #[cfg(all(feature = "int8-encoder", not(feature = "blas")))]
    #[test]
    fn int8_encoder_matvec_matches_single() {
        // Serial full-range exactness vs matvec_int8, across odd sizes, bias
        // on/off, accumulate on/off.
        let mut seed = 0x0EDC_0DE5_u64;
        for &in_dim in &[16usize, 32, 48, 17, 33, 896] {
            for &out_dim in &[1usize, 2, 7, 16, 31, 128] {
                for &seq_len in &[1usize, 2, 5, 8, 33] {
                    for &with_bias in &[false, true] {
                        for &accumulate in &[false, true] {
                            seed = seed.wrapping_add(0x9E37_79B9);
                            encoder_matvec_check(seq_len, in_dim, out_dim, with_bias, accumulate, seed);
                        }
                    }
                }
            }
        }
        // Row-split equivalence (models the dynamic 64-row pool partition), with
        // real encoder shapes (conv_out 7680→896, q/k/v/o 896→896, fc1 896→3584,
        // fc2 3584→896, proj2 896→1024).
        for &(seq_len, in_dim, out_dim) in &[
            (48usize, 896usize, 896usize),
            (64, 7680, 896),
            (48, 896, 3584),
            (64, 3584, 896),
            (33, 896, 1024),
            (5, 64, 200),
        ] {
            for &accumulate in &[false, true] {
                encoder_matvec_split_check(seq_len, in_dim, out_dim, accumulate, 0xC0DE_5EED ^ (seq_len as u64));
            }
        }
    }
}
