//! SME + NEON hybrid GEMM for the decoder prefill on Apple Silicon with SME
//! (R16-1).
//!
//! Replaces the prefill's bf16→f32 scratch widening + pool-sliced Accelerate
//! `cblas_sgemm` for `Y[m][n] = X[m][k] · W[n][k]ᵀ` with bf16 weights. A few
//! pool threads drive the SME unit through a hand-written FMOPA kernel from the
//! front of the column range while the others run a NEON kernel from the back;
//! the weights are widened exactly while being packed into the kernels'
//! layout, which replaces the old widening pass.
//!
//! Bit-identical to the replaced path by construction: both kernels
//! accumulate every output with one fused multiply-add chain in increasing k
//! starting from +0 — the order Accelerate's multi-row sgemm uses on this
//! hardware for every slicing (verified bitwise on the engine's shapes, see
//! `docs/research/experiments.md` R16-1; its 1-row sgemm sums differently, so
//! only the pooled multi-row calls come here).
//!
//! Used only when the CPU reports FEAT_SME with the 512-bit streaming vector
//! length the kernel is written for. `QWEN_ASR_SME=0` disables it.

use core::arch::aarch64::*;
use std::arch::global_asm;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use super::parallel_region;
use super::pool::RegionBarrier;

/// One SME work item: rows `[0, m)` × columns `[0, n)` of Y. `a` and `b` are
/// 32-row / 32-column panels (per k, 32 contiguous values; `k * 128` bytes
/// between panels). Field offsets are read by the asm.
#[repr(C)]
struct SmeTask {
    a: *const f32,
    b: *const f32,
    /// Y at row 0 and the item's first column.
    y: *mut f32,
    /// Bytes between rows of Y.
    ldy: usize,
    k: usize,
    m: usize,
    n: usize,
}

// FMOPA kernel. For each 32-row block of A (outer) and each 32-column panel of
// B (inner), ZA tiles za0..za3 hold the 32x32 output block (za0: rows 0-15 x
// cols 0-15, za1: rows 0-15 x cols 16-31, za2/za3: rows 16-31). Each k step
// loads two 16-row vectors of A and two 16-column vectors of B (predicated to
// the item's columns) and issues four FMOPA, each a fused multiply-add per
// element. Rows beyond m are zero padding and never stored. Requires a
// 64-byte streaming vector length (checked before use).
global_asm!(
    ".arch_extension sme",
    ".text",
    ".p2align 4",
    ".private_extern _qasr_sme_gemm_f32",
    "_qasr_sme_gemm_f32:",
    "stp x19, x20, [sp, #-96]!",
    "stp x21, x22, [sp, #16]",
    // smstart zeroes the vector registers, whose low halves d8-d15 are callee-saved.
    "stp d8, d9, [sp, #32]",
    "stp d10, d11, [sp, #48]",
    "stp d12, d13, [sp, #64]",
    "stp d14, d15, [sp, #80]",
    "ldp x1, x3, [x0]",      // a, b
    "ldp x6, x7, [x0, #16]", // y, ldy
    "ldp x8, x9, [x0, #32]", // k, m (rows left)
    "ldr x10, [x0, #48]",    // n
    "lsl x2, x8, #7",        // panel stride = k * 128 bytes
    "smstart",
    "ptrue p0.s",
    "mov x13, #16",
    // ---- row-block loop: x1 = A block, x6 = Y row block, x9 = rows left
    "10:",
    "cmp x9, #16",
    "csel x21, x9, x13, lt", // rows in za0/za1 = min(rows left, 16)
    "subs x22, x9, #16",
    "csel x22, xzr, x22, lt",
    "cmp x22, #16",
    "csel x22, x22, x13, lt", // rows in za2/za3 = min(max(rows left - 16, 0), 16)
    "mov x14, x3",            // B panel
    "mov x15, x6",            // Y panel
    "mov x16, x10",           // columns left
    // ---- panel loop
    "20:",
    "whilelt p2.s, xzr, x16", // columns 0-15 of the panel that belong to the item
    "whilelt p3.s, x13, x16", // columns 16-31
    "zero {{za}}",
    "mov x17, x1",
    "mov x19, x14",
    "mov x20, x8",
    ".p2align 4",
    "50:",
    "ld1w {{z0.s}}, p0/z, [x17]",
    "ld1w {{z1.s}}, p0/z, [x17, #1, mul vl]",
    "ld1w {{z2.s}}, p2/z, [x19]",
    "ld1w {{z3.s}}, p3/z, [x19, #1, mul vl]",
    "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
    "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
    "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
    "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
    "add x17, x17, #128",
    "add x19, x19, #128",
    "subs x20, x20, #1",
    "b.ne 50b",
    // ---- store the valid rows (columns via the predicates)
    "mov w12, #0",
    "mov x11, x15",
    "60:",
    "cmp w12, w21",
    "b.ge 61f",
    "st1w {{za0h.s[w12, 0]}}, p2, [x11]",
    "add x4, x11, #64",
    "st1w {{za1h.s[w12, 0]}}, p3, [x4]",
    "add x11, x11, x7",
    "add w12, w12, #1",
    "b 60b",
    "61:",
    "mov w12, #0",
    "add x11, x15, x7, lsl #4",
    "62:",
    "cmp w12, w22",
    "b.ge 70f",
    "st1w {{za2h.s[w12, 0]}}, p2, [x11]",
    "add x4, x11, #64",
    "st1w {{za3h.s[w12, 0]}}, p3, [x4]",
    "add x11, x11, x7",
    "add w12, w12, #1",
    "b 62b",
    // ---- next panel, then next row block
    "70:",
    "add x14, x14, x2",
    "add x15, x15, #128",
    "subs x16, x16, #32",
    "b.gt 20b",
    "add x1, x1, x2",
    "add x6, x6, x7, lsl #5",
    "subs x9, x9, #32",
    "b.gt 10b",
    "smstop",
    "ldp d14, d15, [sp, #80]",
    "ldp d12, d13, [sp, #64]",
    "ldp d10, d11, [sp, #48]",
    "ldp d8, d9, [sp, #32]",
    "ldp x21, x22, [sp, #16]",
    "ldp x19, x20, [sp], #96",
    "ret",
    "",
    ".p2align 2",
    ".private_extern _qasr_sme_svl_bytes",
    "_qasr_sme_svl_bytes:",
    "rdsvl x0, #1",
    "ret",
);

extern "C" {
    fn qasr_sme_gemm_f32(task: *const SmeTask);
    /// Streaming vector length in bytes. Only valid on CPUs with FEAT_SME.
    fn qasr_sme_svl_bytes() -> usize;
}

fn sysctl_is_one(name: &[u8]) -> bool {
    debug_assert_eq!(name.last(), Some(&0));
    let mut out: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut out as *mut i32 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    rc == 0 && out == 1
}

/// Whether the hybrid GEMM is available and enabled (cached).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var("QWEN_ASR_SME").is_ok_and(|v| v.trim() == "0") {
            return false;
        }
        // RDSVL is only defined with FEAT_SME, so check that first.
        sysctl_is_one(b"hw.optional.arm.FEAT_SME\0") && unsafe { qasr_sme_svl_bytes() } == 64
    })
}

/// Pool threads that drive the SME kernel; the rest run NEON. M5 Pro sweep
/// (experiments.md R16-1): 2..6 within noise, 4 best on the prefill shapes.
const SME_THREADS: usize = 4;

/// Largest `m * k` (activation elements) handled here: 8 MB of packed rows,
/// which stays in an E-cluster L2. Every column item re-reads all packed rows,
/// so beyond that the long single-pass prefill becomes memory-bound and was
/// 4-7% slower than Accelerate under background load.
pub(crate) const MAX_A_ELEMS: usize = 1 << 21;

/// 16-column slots an SME thread claims at a time (two 32-column panels).
const SME_ITEM_SLOTS: usize = 4;
/// NEON threads stop claiming once this few slots remain: an SME thread does
/// several slots in the time a NEON thread does one, so the tail goes to SME
/// instead of waiting on a NEON straggler.
const NEON_RESERVE_SLOTS: usize = 8;

thread_local! {
    /// Packed X / W panels, owned by the dispatching thread and reused.
    static SCRATCH: RefCell<(Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// `Y[m][n] = X[m][k] · W[n][k]ᵀ` on the kernel pool, with `W` row-major bf16
/// bits (`n % 16 == 0`, `m >= 2`, `m * k <= MAX_A_ELEMS` for speed).
pub(crate) fn gemm_bf16(y: &mut [f32], x: &[f32], w: *const u16, m: usize, k: usize, n: usize) {
    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let (a_buf, b_buf) = &mut *scratch;
        let job = Job::new(y, x, w, m, k, n, a_buf, b_buf, SME_THREADS);
        parallel_region(|barrier, tid, nt| job.run(barrier, tid, nt));
    });
}

struct Job {
    y: *mut f32,
    x: *const f32,
    w: *const u16,
    m: usize,
    k: usize,
    n: usize,
    a_panels: *mut f32,
    b_panels: *mut f32,
    a_blocks: usize,
    b_count: usize,
    sme_threads: usize,
    pack_next: AtomicUsize,
    /// Unclaimed 16-column slots `lo << 32 | hi`: SME claims from `lo`, NEON from `hi`.
    slots: AtomicU64,
}

// The raw pointers are only dereferenced on disjoint regions (packing items,
// claimed column slots), published across threads by the region barrier and
// the region join.
unsafe impl Sync for Job {}

impl Job {
    #[allow(clippy::too_many_arguments)]
    fn new(
        y: &mut [f32],
        x: &[f32],
        w: *const u16,
        m: usize,
        k: usize,
        n: usize,
        a_buf: &mut Vec<f32>,
        b_buf: &mut Vec<f32>,
        sme_threads: usize,
    ) -> Self {
        assert!(m >= 1 && k >= 1 && n >= 16 && n.is_multiple_of(16));
        assert!(y.len() >= m * n && x.len() >= m * k);
        let a_blocks = m.div_ceil(32);
        let b_count = n.div_ceil(32);
        if a_buf.len() < a_blocks * k * 32 {
            a_buf.resize(a_blocks * k * 32, 0.0);
        }
        if b_buf.len() < b_count * k * 32 {
            b_buf.resize(b_count * k * 32, 0.0);
        }
        Job {
            y: y.as_mut_ptr(),
            x: x.as_ptr(),
            w,
            m,
            k,
            n,
            a_panels: a_buf.as_mut_ptr(),
            b_panels: b_buf.as_mut_ptr(),
            a_blocks,
            b_count,
            sme_threads,
            pack_next: AtomicUsize::new(0),
            slots: AtomicU64::new((n / 16) as u64),
        }
    }

    /// Per-thread body: pack X and W into panels, barrier, claim column slots.
    fn run(&self, barrier: &RegionBarrier, tid: usize, nt: usize) {
        let k = self.k;
        loop {
            let i = self.pack_next.fetch_add(1, Ordering::Relaxed);
            if i >= self.a_blocks + self.b_count {
                break;
            }
            unsafe {
                if i < self.a_blocks {
                    let rows = (self.m - i * 32).min(32);
                    pack_block(
                        self.a_panels.add(i * k * 32),
                        self.x.add(i * 32 * k),
                        k,
                        rows,
                    );
                } else {
                    let p = i - self.a_blocks;
                    let rows = (self.n - p * 32).min(32);
                    pack_block(
                        self.b_panels.add(p * k * 32),
                        self.w.add(p * 32 * k),
                        k,
                        rows,
                    );
                }
            }
        }
        barrier.wait();

        let sme = self.sme_threads.min(nt);
        if tid < sme {
            self.sme_claims();
        } else {
            self.neon_claims(if sme == 0 { 0 } else { NEON_RESERVE_SLOTS });
        }
    }

    fn sme_claims(&self) {
        loop {
            let cur = self.slots.load(Ordering::Relaxed);
            let (lo, hi) = ((cur >> 32) as usize, (cur & 0xffff_ffff) as usize);
            if lo >= hi {
                return;
            }
            let end = (lo + SME_ITEM_SLOTS).min(hi);
            let next = ((end as u64) << 32) | hi as u64;
            if self
                .slots
                .compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                // `lo` starts at 0 and moves in even steps, so the item starts
                // on a 32-column panel; an odd `end` leaves a half panel.
                let col0 = lo * 16;
                let task = SmeTask {
                    a: self.a_panels,
                    b: unsafe { self.b_panels.add((col0 / 32) * self.k * 32) },
                    y: unsafe { self.y.add(col0) },
                    ldy: self.n * 4,
                    k: self.k,
                    m: self.m,
                    n: (end - lo) * 16,
                };
                unsafe { qasr_sme_gemm_f32(&task) };
            }
        }
    }

    fn neon_claims(&self, reserve: usize) {
        loop {
            let cur = self.slots.load(Ordering::Relaxed);
            let (lo, hi) = ((cur >> 32) as usize, (cur & 0xffff_ffff) as usize);
            if hi <= lo + reserve {
                return;
            }
            let next = ((lo as u64) << 32) | (hi - 1) as u64;
            if self
                .slots
                .compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                let col0 = (hi - 1) * 16;
                unsafe {
                    let bp = self.b_panels.add((col0 / 32) * self.k * 32 + col0 % 32);
                    neon_kernel(self.y.add(col0), self.n, self.a_panels, bp, self.k, self.m);
                }
            }
        }
    }
}

/// Source element of a packed matrix: f32, or bf16 bits widened exactly.
trait PackSrc: Copy {
    unsafe fn load4(p: *const Self) -> float32x4_t;
    fn to_f32(self) -> f32;
}
impl PackSrc for f32 {
    #[inline(always)]
    unsafe fn load4(p: *const f32) -> float32x4_t {
        vld1q_f32(p)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}
impl PackSrc for u16 {
    #[inline(always)]
    unsafe fn load4(p: *const u16) -> float32x4_t {
        vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(p)))
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from_bits((self as u32) << 16)
    }
}

/// Pack `rows` (<= 32) rows of a row-major `[rows][k]` matrix into one 32-row
/// panel, `dst[kk * 32 + r] = src[r][kk]`, zeroing rows `rows..32`.
unsafe fn pack_block<T: PackSrc>(dst: *mut f32, src: *const T, k: usize, rows: usize) {
    let k4 = k & !3;
    let mut r0 = 0;
    while r0 < 32 {
        if r0 + 4 <= rows {
            let (s0, s1, s2, s3) = (
                src.add(r0 * k),
                src.add((r0 + 1) * k),
                src.add((r0 + 2) * k),
                src.add((r0 + 3) * k),
            );
            let mut kk = 0;
            while kk < k4 {
                // 4x4 transpose: rows r0..r0+4 at kk..kk+4 -> four k rows of the panel.
                let (v0, v1) = (T::load4(s0.add(kk)), T::load4(s1.add(kk)));
                let (v2, v3) = (T::load4(s2.add(kk)), T::load4(s3.add(kk)));
                let (t0, t1) = (vtrn1q_f32(v0, v1), vtrn2q_f32(v0, v1));
                let (t2, t3) = (vtrn1q_f32(v2, v3), vtrn2q_f32(v2, v3));
                let pair = |lo: bool, a: float32x4_t, b: float32x4_t| {
                    let (a, b) = (vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b));
                    vreinterpretq_f32_f64(if lo {
                        vtrn1q_f64(a, b)
                    } else {
                        vtrn2q_f64(a, b)
                    })
                };
                let d = dst.add(kk * 32 + r0);
                vst1q_f32(d, pair(true, t0, t2));
                vst1q_f32(d.add(32), pair(true, t1, t3));
                vst1q_f32(d.add(64), pair(false, t0, t2));
                vst1q_f32(d.add(96), pair(false, t1, t3));
                kk += 4;
            }
            while kk < k {
                for r in 0..4 {
                    *dst.add(kk * 32 + r0 + r) = (*src.add((r0 + r) * k + kk)).to_f32();
                }
                kk += 1;
            }
        } else {
            for r in r0..r0 + 4 {
                for kk in 0..k {
                    *dst.add(kk * 32 + r) = if r < rows {
                        (*src.add(r * k + kk)).to_f32()
                    } else {
                        0.0
                    };
                }
            }
        }
        r0 += 4;
    }
}

/// NEON kernel for one 16-column slot: `Y[i][0..16] = Σ_k A[i][k] B[k][0..16]`
/// for `i < m`, one fused multiply-add per k in increasing k from +0 (the
/// same arithmetic as the FMOPA kernel). `a_panels` is A in 32-row panels;
/// `bp` points at the slot's first column in a 32-column panel of B.
#[inline(never)]
unsafe fn neon_kernel(
    y: *mut f32,
    ldy: usize,
    a_panels: *const f32,
    bp: *const f32,
    k: usize,
    m: usize,
) {
    let z = vdupq_n_f32(0.0);
    let mut i0 = 0;
    while i0 < m {
        let ap = a_panels.add((i0 / 32) * k * 32 + i0 % 32);
        let (mut c00, mut c01, mut c02, mut c03) = (z, z, z, z);
        let (mut c10, mut c11, mut c12, mut c13) = (z, z, z, z);
        let (mut c20, mut c21, mut c22, mut c23) = (z, z, z, z);
        let (mut c30, mut c31, mut c32, mut c33) = (z, z, z, z);
        for kk in 0..k {
            let a = vld1q_f32(ap.add(kk * 32));
            let b = bp.add(kk * 32);
            let (b0, b1, b2, b3) = (
                vld1q_f32(b),
                vld1q_f32(b.add(4)),
                vld1q_f32(b.add(8)),
                vld1q_f32(b.add(12)),
            );
            c00 = vfmaq_laneq_f32::<0>(c00, b0, a);
            c01 = vfmaq_laneq_f32::<0>(c01, b1, a);
            c02 = vfmaq_laneq_f32::<0>(c02, b2, a);
            c03 = vfmaq_laneq_f32::<0>(c03, b3, a);
            c10 = vfmaq_laneq_f32::<1>(c10, b0, a);
            c11 = vfmaq_laneq_f32::<1>(c11, b1, a);
            c12 = vfmaq_laneq_f32::<1>(c12, b2, a);
            c13 = vfmaq_laneq_f32::<1>(c13, b3, a);
            c20 = vfmaq_laneq_f32::<2>(c20, b0, a);
            c21 = vfmaq_laneq_f32::<2>(c21, b1, a);
            c22 = vfmaq_laneq_f32::<2>(c22, b2, a);
            c23 = vfmaq_laneq_f32::<2>(c23, b3, a);
            c30 = vfmaq_laneq_f32::<3>(c30, b0, a);
            c31 = vfmaq_laneq_f32::<3>(c31, b1, a);
            c32 = vfmaq_laneq_f32::<3>(c32, b2, a);
            c33 = vfmaq_laneq_f32::<3>(c33, b3, a);
        }
        let out = [
            [c00, c01, c02, c03],
            [c10, c11, c12, c13],
            [c20, c21, c22, c23],
            [c30, c31, c32, c33],
        ];
        for (r, vs) in out.iter().enumerate().take(m - i0) {
            for (j, &v) in vs.iter().enumerate() {
                vst1q_f32(y.add((i0 + r) * ldy + 4 * j), v);
            }
        }
        i0 += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn f32(&mut self, scale: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        }
    }

    fn to_bf16(v: f32) -> u16 {
        let u = v.to_bits();
        ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
    }

    /// One job on `nt` scoped threads (not the global pool, so tests can run
    /// concurrently), `sme` of them on the SME kernel.
    #[allow(clippy::too_many_arguments)]
    fn run_job(
        y: &mut [f32],
        x: &[f32],
        w: &[u16],
        m: usize,
        k: usize,
        n: usize,
        nt: usize,
        sme: usize,
    ) {
        let (mut ab, mut bb) = (Vec::new(), Vec::new());
        let job = Job::new(y, x, w.as_ptr(), m, k, n, &mut ab, &mut bb, sme);
        let barrier = RegionBarrier::new(nt);
        std::thread::scope(|s| {
            for tid in 0..nt {
                let (job, barrier) = (&job, &barrier);
                s.spawn(move || job.run(barrier, tid, nt));
            }
        });
    }

    fn inputs(m: usize, k: usize, n: usize, seed: u64) -> (Vec<f32>, Vec<u16>) {
        let mut rng = Rng(seed);
        let x: Vec<f32> = (0..m * k).map(|_| rng.f32(2.0)).collect();
        let w: Vec<u16> = (0..n * k).map(|_| to_bf16(rng.f32(0.05))).collect();
        (x, w)
    }

    #[test]
    fn hybrid_kernels_match_sequential_fma_reference() {
        // (m, k, n): partial row blocks / 4-row tiles, odd k, half panels.
        for &(m, k, n) in &[
            (2, 1, 16),
            (3, 7, 16),
            (17, 33, 48),
            (32, 64, 32),
            (33, 64, 96),
            (50, 130, 80),
        ] {
            let (x, w) = inputs(m, k, n, (m * 1000 + k * 10 + n) as u64);
            // Reference: one fused multiply-add per k in increasing k from +0.
            let mut want = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc = x[i * k + kk].mul_add(w[j * k + kk].to_f32(), acc);
                    }
                    want[i * n + j] = acc;
                }
            }
            // NEON only, then SME only and mixed where SME exists.
            let mut configs = vec![(3usize, 0usize)];
            if enabled() {
                configs.extend([(1, 1), (2, 2), (4, 1), (6, 2)]);
            }
            for (nt, sme) in configs {
                let mut got = vec![f32::NAN; m * n];
                run_job(&mut got, &x, &w, m, k, n, nt, sme);
                let bad = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(bad, 0, "m={m} k={k} n={n} nt={nt} sme={sme}");
            }
        }
    }

    /// The exactness argument rests on Accelerate's multi-row sgemm summing in
    /// increasing k with one fused chain per output. Pin it on real prefill
    /// shapes against the path the hybrid replaces (widen + 128-column slices).
    #[test]
    fn hybrid_matches_accelerate_on_prefill_shapes() {
        if !enabled() {
            return;
        }
        for &(m, k, n) in &[(383, 1024, 1024), (64, 3072, 1024), (7, 1024, 3072)] {
            let (x, w) = inputs(m, k, n, 7);
            let wf: Vec<f32> = w.iter().map(|&h| h.to_f32()).collect();
            let mut want = vec![0.0f32; m * n];
            for s in (0..n).step_by(128) {
                let e = (s + 128).min(n);
                unsafe {
                    super::super::cblas_sgemm(
                        101,
                        111,
                        112,
                        m as i32,
                        (e - s) as i32,
                        k as i32,
                        1.0,
                        x.as_ptr(),
                        k as i32,
                        wf.as_ptr().add(s * k),
                        k as i32,
                        0.0,
                        want.as_mut_ptr().add(s),
                        n as i32,
                    );
                }
            }
            let mut got = vec![f32::NAN; m * n];
            run_job(&mut got, &x, &w, m, k, n, 6, 2);
            let bad = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                bad, 0,
                "m={m} k={k} n={n}: {bad} outputs differ from Accelerate"
            );
        }
    }
}
