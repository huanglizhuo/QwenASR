//! BLAS/vDSP bindings, thread pool, and SIMD kernel dispatch.

pub mod generic;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "x86_64")]
pub mod avx;

use std::thread;

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
#[link(name = "openblas")]
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
pub static mut VERBOSE: i32 = 0;

// ========================================================================
// Profiling support
// ========================================================================

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_profile(enabled: bool) {
    PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn is_profiling() -> bool {
    PROFILE_ENABLED.load(Ordering::Relaxed)
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
    rms_norm, layer_norm, silu, gelu, swiglu,
    bf16_matvec, bf16_to_f32_conv, attention_bidir, attention_causal,
    linear_f32, softmax_op, conv2d_op, rope
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
    unsafe { VERBOSE = v; }
}

pub fn verbose() -> i32 {
    unsafe { VERBOSE }
}

// ========================================================================
// Thread Pool (persistent, mutex+condvar, matches C approach)
// ========================================================================

use std::sync::{Mutex, Condvar, Arc, Once};

const MAX_THREADS: usize = 16;

struct ThreadPool {
    state: Mutex<ThreadPoolState>,
    work_cv: Condvar,
    done_cv: Condvar,
}

struct ThreadPoolState {
    n_threads: usize,
    shutdown: bool,
    generation: u64,
    n_done: usize,
    fn_ptr: Option<*const ()>,  // type-erased function pointer
    fn_call: Option<fn(*const (), usize, usize)>, // trampoline
}

unsafe impl Send for ThreadPoolState {}
unsafe impl Sync for ThreadPoolState {}

#[allow(static_mut_refs)]
static mut THREAD_POOL: Option<Arc<ThreadPool>> = None;
static POOL_INIT: Once = Once::new();

#[allow(static_mut_refs)]
fn get_pool() -> &'static Arc<ThreadPool> {
    POOL_INIT.call_once(|| {
        unsafe {
            THREAD_POOL = Some(Arc::new(ThreadPool {
                state: Mutex::new(ThreadPoolState {
                    n_threads: 1,
                    shutdown: false,
                    generation: 0,
                    n_done: 0,
                    fn_ptr: None,
                    fn_call: None,
                }),
                work_cv: Condvar::new(),
                done_cv: Condvar::new(),
            }));
        }
    });
    unsafe { THREAD_POOL.as_ref().unwrap() }
}

fn pool_worker(pool: Arc<ThreadPool>, tid: usize) {
    let mut last_gen: u64 = 0;
    loop {
        let (fn_ptr, fn_call, n_threads);
        {
            let mut state = match pool.state.lock() {
                Ok(s) => s,
                Err(p) => p.into_inner(), // recover from poisoned mutex
            };
            while !state.shutdown && state.generation == last_gen {
                state = match pool.work_cv.wait(state) {
                    Ok(s) => s,
                    Err(p) => p.into_inner(),
                };
            }
            if state.shutdown {
                return;
            }
            last_gen = state.generation;
            fn_ptr = match state.fn_ptr {
                Some(p) => p,
                None => continue, // spurious wake or cleared — retry
            };
            fn_call = match state.fn_call {
                Some(f) => f,
                None => continue,
            };
            n_threads = state.n_threads;
        }

        // Execute work
        fn_call(fn_ptr, tid, n_threads);

        // Signal done
        {
            let mut state = match pool.state.lock() {
                Ok(s) => s,
                Err(p) => p.into_inner(),
            };
            state.n_done += 1;
            pool.done_cv.notify_one();
        }
    }
}

static mut SPAWNED_THREADS: usize = 0;

fn ensure_workers(pool: &Arc<ThreadPool>, n_threads: usize) {
    unsafe {
        if SPAWNED_THREADS >= n_threads - 1 {
            return;
        }
        let start = SPAWNED_THREADS + 1;
        for tid in start..n_threads {
            let p = pool.clone();
            thread::Builder::new()
                .name(format!("qwen-worker-{}", tid))
                .spawn(move || pool_worker(p, tid))
                .unwrap();
        }
        SPAWNED_THREADS = n_threads - 1;
    }
}

static mut THREAD_POOL_THREADS: usize = 1;

pub fn set_threads(n: usize) {
    let n = n.max(1).min(MAX_THREADS);
    unsafe { THREAD_POOL_THREADS = n; }
    if n > 1 {
        let pool = get_pool();
        ensure_workers(pool, n);
        match pool.state.lock() {
            Ok(mut s) => s.n_threads = n,
            Err(p) => p.into_inner().n_threads = n,
        }
    }
    if verbose() >= 2 {
        eprintln!("Thread pool: {} threads", n);
    }
}

pub fn get_num_threads() -> usize {
    unsafe { THREAD_POOL_THREADS }
}

pub fn get_num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Run a closure in parallel using the persistent thread pool.
/// The closure takes (thread_id, n_threads).
fn parallel_for<F: Fn(usize, usize) + Send + Sync>(f: F) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        f(0, 1);
        return;
    }

    let pool = get_pool();

    // Trampoline: cast *const () back to &F and call it
    fn trampoline<F: Fn(usize, usize) + Send + Sync>(ptr: *const (), tid: usize, nt: usize) {
        let f = unsafe { &*(ptr as *const F) };
        f(tid, nt);
    }

    {
        let mut state = match pool.state.lock() {
            Ok(s) => s,
            Err(p) => p.into_inner(),
        };
        state.fn_ptr = Some(&f as *const F as *const ());
        state.fn_call = Some(trampoline::<F>);
        state.n_done = 0;
        state.generation += 1;
        pool.work_cv.notify_all();
    }

    // Main thread does tid=0
    f(0, n_threads);

    // Wait for workers
    {
        let mut state = match pool.state.lock() {
            Ok(s) => s,
            Err(p) => p.into_inner(),
        };
        while state.n_done < n_threads - 1 {
            state = match pool.done_cv.wait(state) {
                Ok(s) => s,
                Err(p) => p.into_inner(),
            };
        }
        state.fn_ptr = None;
        state.fn_call = None;
    }
}

// ========================================================================
// Dispatch helpers - pick NEON/AVX/generic at compile time
// ========================================================================

#[inline]
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

pub fn bf16_to_f32_buf(dst: &mut [f32], src: &[u16]) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::bf16_to_f32_buf(dst, src); } return; }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::bf16_to_f32_buf(dst, src); } return; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in 0..src.len() {
        dst[i] = bf16_to_f32(src[i]);
    }
}

fn bf16_matvec_fused(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); } return; }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim); } return; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::bf16_matvec_fused(y, x, w_bf16, bias, in_dim, out_dim);
}

fn argmax_bf16_range(x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize) -> (usize, f32) {
    #[cfg(target_arch = "aarch64")]
    { return unsafe { neon::argmax_bf16_range(x, w_bf16, in_dim, start, end) }; }

    #[cfg(target_arch = "x86_64")]
    { return unsafe { avx::argmax_bf16_range(x, w_bf16, in_dim, start, end) }; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::argmax_bf16_range(x, w_bf16, in_dim, start, end)
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        let mut result = 0.0f32;
        unsafe { vDSP_dotpr(a.as_ptr(), 1, b.as_ptr(), 1, &mut result, n as u64); }
        return result;
    }

    #[cfg(target_arch = "aarch64")]
    { return unsafe { neon::dot_f32(a, b, n) }; }

    #[cfg(target_arch = "x86_64")]
    { return unsafe { avx::dot_f32(a, b, n) }; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::dot_f32(a, b, n)
}

#[inline]
pub fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        unsafe { vDSP_vsmul(dst.as_ptr(), 1, &scale, dst.as_mut_ptr(), 1, n as u64); }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::vec_scale_inplace(dst, scale, n); } return; }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::vec_scale_inplace(dst, scale, n); } return; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::vec_scale_inplace(dst, scale, n);
}

#[inline]
pub fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    #[cfg(all(feature = "vdsp", target_vendor = "apple"))]
    {
        unsafe { vDSP_vsma(src.as_ptr(), 1, &alpha, dst.as_ptr(), 1, dst.as_mut_ptr(), 1, n as u64); }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::vec_axpy_inplace(dst, src, alpha, n); } return; }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::vec_axpy_inplace(dst, src, alpha, n); } return; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::vec_axpy_inplace(dst, src, alpha, n);
}

#[inline]
pub fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::vec_scale_add(dst, src, correction, n); } return; }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::vec_scale_add(dst, src, correction, n); } return; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    generic::vec_scale_add(dst, src, correction, n);
}

// ========================================================================
// Basic Operations
// ========================================================================

pub fn add_inplace(a: &mut [f32], b: &[f32], n: usize) {
    for i in 0..n { a[i] += b[i]; }
}

pub fn scale(x: &mut [f32], s: f32, n: usize) {
    for i in 0..n { x[i] *= s; }
}

pub fn copy(dst: &mut [f32], src: &[f32], n: usize) {
    dst[..n].copy_from_slice(&src[..n]);
}

// ========================================================================
// Matrix Operations
// ========================================================================

/// C = A @ B^T: A[M,K], B[N,K], C[M,N]
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
        return;
    }

    #[cfg(not(feature = "blas"))]
    {
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

/// y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out]
pub fn linear(y: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    #[cfg(feature = "blas")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
            seq_len as i32, out_dim as i32, in_dim as i32,
            1.0, x.as_ptr(), in_dim as i32,
            w.as_ptr(), in_dim as i32,
            0.0, y.as_mut_ptr(), out_dim as i32,
        );
        if let Some(b) = b {
            for s in 0..seq_len {
                for o in 0..out_dim {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
        return;
    }

    #[cfg(not(feature = "blas"))]
    {
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

// BF16 scratch buffer for bf16->f32 conversion
thread_local! {
    static BF16_SCRATCH: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
}

fn bf16_get_scratch(n: usize) -> Vec<f32> {
    vec![0.0f32; n]
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

    parallel_for(|tid, nt| {
        let chunk = (out_dim + nt - 1) / nt;
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }

        let y_local = unsafe { std::slice::from_raw_parts_mut((y_send as *mut f32).add(start), end - start) };
        let x_local = unsafe { std::slice::from_raw_parts(x_send as *const f32, in_dim) };
        let w_local = unsafe { (w_send as *const u16).add(start * in_dim) };
        let bias_local = bias_send.map(|p| unsafe { std::slice::from_raw_parts((p as *const f32).add(start), end - start) });

        bf16_matvec_fused(y_local, x_local, w_local, bias_local, in_dim, end - start);
    });
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
pub fn linear_nobias_bf16_scratch(y: &mut [f32], x: &[f32], w_bf16: *const u16, seq_len: usize, in_dim: usize, out_dim: usize, scratch: &mut [f32]) {
    let _pg = ProfileGuard::new(&PROF.bf16_matvec);
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, None, in_dim, out_dim);
        return;
    }
    let n = out_dim * in_dim;
    let src = unsafe { std::slice::from_raw_parts(w_bf16, n) };
    bf16_to_f32_buf(&mut scratch[..n], src);
    linear_nobias(y, x, &scratch[..n], seq_len, in_dim, out_dim);
}

pub fn linear_bf16(y: &mut [f32], x: &[f32], w_bf16: *const u16, b: Option<&[f32]>, seq_len: usize, in_dim: usize, out_dim: usize) {
    if seq_len == 1 {
        bf16_matvec_threaded(y, x, w_bf16, b, in_dim, out_dim);
        return;
    }
    let w_f32 = bf16_to_f32_view(w_bf16, out_dim * in_dim);
    linear(y, x, &w_f32, b, seq_len, in_dim, out_dim);
}

/// Fused Q/K/V matvec for single-token decode
pub fn linear_nobias_bf16_qkv(
    q: &mut [f32], k: &mut [f32], v: &mut [f32], x: &[f32],
    wq: *const u16, wk: *const u16, wv: *const u16,
    in_dim: usize, q_dim: usize, kv_dim: usize,
) {
    let n_threads = get_num_threads();
    if n_threads <= 1 {
        bf16_matvec_fused(q, x, wq, None, in_dim, q_dim);
        bf16_matvec_fused(k, x, wk, None, in_dim, kv_dim);
        bf16_matvec_fused(v, x, wv, None, in_dim, kv_dim);
        return;
    }

    let total_dim = q_dim + 2 * kv_dim;
    let q_ptr = q.as_mut_ptr() as usize;
    let k_ptr = k.as_mut_ptr() as usize;
    let v_ptr = v.as_mut_ptr() as usize;
    let x_ptr = x.as_ptr() as usize;
    let wq_ptr = wq as usize;
    let wk_ptr = wk as usize;
    let wv_ptr = wv as usize;

    parallel_for(|tid, nt| {
        let chunk = (total_dim + nt - 1) / nt;
        let start = tid * chunk;
        let end = (start + chunk).min(total_dim);
        if start >= end { return; }

        let x_local = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, in_dim) };
        let q_end = q_dim;
        let k_end = q_end + kv_dim;

        // Q range
        if start < q_end {
            let s = start;
            let e = end.min(q_end);
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((q_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wq_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }

        // K range
        if end > q_end && start < k_end {
            let s = if start > q_end { start - q_end } else { 0 };
            let e_abs = end.min(k_end);
            let e = e_abs - q_end;
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((k_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wk_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }

        // V range
        if end > k_end {
            let s = if start > k_end { start - k_end } else { 0 };
            let e_abs = end.min(total_dim);
            let e = e_abs - k_end;
            if s < e {
                let y_local = unsafe { std::slice::from_raw_parts_mut((v_ptr as *mut f32).add(s), e - s) };
                let w_local = unsafe { (wv_ptr as *const u16).add(s * in_dim) };
                bf16_matvec_fused(y_local, x_local, w_local, None, in_dim, e - s);
            }
        }
    });
}

pub fn matmul_t_bf16(c: &mut [f32], a: &[f32], b_bf16: *const u16, m: usize, k: usize, n: usize) {
    if m == 1 {
        bf16_matvec_threaded(c, a, b_bf16, None, k, n);
    } else {
        let b_f32 = bf16_to_f32_view(b_bf16, n * k);
        matmul_t(c, a, &b_f32, m, k, n);
    }
}

// ========================================================================
// 2D Convolution (im2col + BLAS sgemm)
// ========================================================================

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

pub fn conv2d(out: &mut [f32], input: &[f32], weight: &[f32], bias: Option<&[f32]>,
              c_in: usize, c_out: usize, h_in: usize, w_in: usize,
              kh: usize, kw: usize, stride: usize, padding: usize) {
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;

    let mut cols = vec![0.0f32; patch_size * spatial_out];
    im2col(input, &mut cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    // GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out]
    #[cfg(feature = "blas")]
    unsafe {
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
            for s in 0..spatial_out {
                out[oc * spatial_out + s] += b;
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
            let vec = &mut x[off..off + head_dim];

            let sum_sq: f32 = vec.iter().map(|&v| v * v).sum();
            let rms_inv = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();

            for d in 0..head_dim {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
        }
    }
}

// ========================================================================
// Activation Functions
// ========================================================================

pub fn silu(x: &mut [f32], n: usize) {
    for i in 0..n {
        let val = x[i];
        x[i] = val / (1.0 + (-val).exp());
    }
}

pub fn gelu(x: &mut [f32], n: usize) {
    let _pg = ProfileGuard::new(&PROF.gelu);
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon::gelu_inplace(x, n); } return; }

    #[cfg(target_arch = "x86_64")]
    { unsafe { avx::gelu_inplace(x, n); } return; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in 0..n {
        let val = x[i];
        let x3 = val * val * val;
        let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
        x[i] = 0.5 * val * (1.0 + inner.tanh());
    }
}

pub fn swiglu_multiply(out: &mut [f32], gate_up: &[f32], seq_len: usize, intermediate: usize) {
    let _pg = ProfileGuard::new(&PROF.swiglu);
    for s in 0..seq_len {
        let gu = &gate_up[s * 2 * intermediate..s * 2 * intermediate + 2 * intermediate];
        let o = &mut out[s * intermediate..(s + 1) * intermediate];

        #[cfg(target_arch = "aarch64")]
        { unsafe { neon::swiglu_interleaved(o, gu, intermediate); } continue; }

        #[cfg(target_arch = "x86_64")]
        { unsafe { avx::swiglu_interleaved(o, gu, intermediate); } continue; }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        for j in 0..intermediate {
            let g = gu[2 * j];
            let u = gu[2 * j + 1];
            let g_silu = g / (1.0 + (-g).exp());
            o[j] = g_silu * u;
        }
    }
}

/// SwiGLU in-place: gate = silu(gate) * up, where gate and up are separate slices.
pub fn swiglu_multiply_inplace(gate: &mut [f32], up: &[f32]) {
    for j in 0..gate.len() {
        let g = gate[j];
        let g_silu = g / (1.0 + (-g).exp());
        gate[j] = g_silu * up[j];
    }
}

pub fn softmax(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for c in 0..cols {
            row[c] -= max_val;
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
        for c in 0..cols {
            sum += row[c];
        }
        let inv_sum = 1.0 / sum;
        for c in 0..cols {
            row[c] *= inv_sum;
        }
    }
}

// ========================================================================
// Attention Operations
// ========================================================================

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
                for d in 0..head_dim { o_row[d] = 0.0; }

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

        parallel_for(|tid, nt| {
            let chunk = (n_heads + nt - 1) / nt;
            let h0 = tid * chunk;
            let h1 = (h0 + chunk).min(n_heads);
            if h0 >= h1 { return; }

            let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, seq * hidden) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, seq * hidden) };
            let k_local = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, seq * hidden) };
            let v_local = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, seq * hidden) };
            let ws_local = unsafe { std::slice::from_raw_parts(ws_ptr as *const i32, n_windows + 1) };

            bidirectional_attention_heads(out_local, q_local, k_local, v_local,
                                         n_heads, head_dim, scale,
                                         ws_local, n_windows, h0, h1);
        });
        return;
    }

    bidirectional_attention_heads(out, q, k, v, n_heads, head_dim, scale,
                                 window_starts, n_windows, 0, n_heads);
}

fn causal_attention_heads(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
                           seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                           head_dim: usize, scale: f32, q_offset: usize,
                           head_start: usize, head_end: usize) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;
    let kv_hidden = n_kv_heads * head_dim;

    for h in head_start..head_end {
        let kv_h = h / heads_per_kv;

        for i in 0..seq_q {
            let q_off = i * q_hidden + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];
            let o_row = &mut out[i * q_hidden + h * head_dim..i * q_hidden + h * head_dim + head_dim];
            let global_pos = q_offset + i;
            let k_end = (global_pos + 1).min(seq_k);

            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            for d in 0..head_dim { o_row[d] = 0.0; }

            for j in 0..k_end {
                let k_off = j * kv_hidden + kv_h * head_dim;
                let v_off = j * kv_hidden + kv_h * head_dim;
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

pub fn causal_attention(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
                         seq_q: usize, seq_k: usize, n_heads: usize, n_kv_heads: usize,
                         head_dim: usize, scale: f32, q_offset: usize) {
    let _pg = ProfileGuard::new(&PROF.attention_causal);
    let n_threads = get_num_threads();
    if n_threads > 1 && n_heads >= 2 {
        let out_ptr = out.as_mut_ptr() as usize;
        let q_ptr = q.as_ptr() as usize;
        let k_ptr = k.as_ptr() as usize;
        let v_ptr = v.as_ptr() as usize;
        let q_hidden = n_heads * head_dim;
        let kv_hidden = n_kv_heads * head_dim;

        parallel_for(|tid, nt| {
            let chunk = (n_heads + nt - 1) / nt;
            let h0 = tid * chunk;
            let h1 = (h0 + chunk).min(n_heads);
            if h0 >= h1 { return; }

            let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, seq_q * q_hidden) };
            let q_local = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, seq_q * q_hidden) };
            let k_local = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, seq_k * kv_hidden) };
            let v_local = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, seq_k * kv_hidden) };

            causal_attention_heads(out_local, q_local, k_local, v_local,
                                   seq_q, seq_k, n_heads, n_kv_heads,
                                   head_dim, scale, q_offset, h0, h1);
        });
        return;
    }

    causal_attention_heads(out, q, k, v, seq_q, seq_k, n_heads, n_kv_heads,
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

pub fn compute_rope_neox(cos_out: &mut [f32], sin_out: &mut [f32], positions: &[i32],
                          seq: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;

    for s in 0..seq {
        let pos = positions[s] as f32;
        for d in 0..half {
            let freq = 1.0 / theta.powf((2 * d) as f32 / head_dim as f32);
            let angle = pos * freq;
            let c = angle.cos();
            let sn = angle.sin();
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

pub fn apply_rope_neox(x: &mut [f32], cos_vals: &[f32], sin_vals: &[f32],
                        seq: usize, n_heads: usize, head_dim: usize) {
    let half = head_dim / 2;
    let hidden = n_heads * head_dim;

    for s in 0..seq {
        let c = &cos_vals[s * head_dim..];
        let sn = &sin_vals[s * head_dim..];

        for h in 0..n_heads {
            let vec = &mut x[s * hidden + h * head_dim..s * hidden + (h + 1) * head_dim];

            for d in 0..half {
                let x1 = vec[d];
                let x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
        }
    }
}

/// Streaming argmax: finds argmax(W_bf16 @ x) without materializing full logits.
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
        let chunk = (out_dim + nt - 1) / nt;
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
