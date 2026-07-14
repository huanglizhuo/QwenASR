//! Persistent thread pool (mutex+condvar, matching the C reference approach)
//! and the barrier-synchronized parallel region used by the fused decode loop.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;

use super::verbose;

pub(crate) const MAX_THREADS: usize = 16;

struct ThreadPool {
    // Mutex+condvar only used as slow-path fallback when spin-wait misses
    state: Mutex<bool>, // shutdown flag only
    work_cv: Condvar,
    // All dispatch data is lock-free via atomics
    gen_atomic: AtomicU64,
    done_atomic: AtomicUsize,
    fn_ptr_atomic: AtomicUsize,
    fn_call_atomic: AtomicUsize,
    n_threads_atomic: AtomicUsize,
}

static THREAD_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

fn get_pool() -> &'static Arc<ThreadPool> {
    THREAD_POOL.get_or_init(|| {
        Arc::new(ThreadPool {
            state: Mutex::new(false),
            work_cv: Condvar::new(),
            gen_atomic: AtomicU64::new(0),
            done_atomic: AtomicUsize::new(0),
            fn_ptr_atomic: AtomicUsize::new(0),
            fn_call_atomic: AtomicUsize::new(0),
            n_threads_atomic: AtomicUsize::new(1),
        })
    })
}

fn pool_worker(pool: Arc<ThreadPool>, tid: usize) {
    // Start from the CURRENT generation: a worker spawned after dispatches
    // have already happened (set_threads growing the pool mid-process, e.g.
    // in tests) must not "see" the last long-completed dispatch as new work
    // and replay its dead closure frame.
    let mut last_gen: u64 = pool.gen_atomic.load(Ordering::Acquire);
    loop {
        // Fast path: spin briefly on atomic generation counter
        let mut found = false;
        for _ in 0..512 {
            let gen = pool.gen_atomic.load(Ordering::Acquire);
            if gen != last_gen {
                last_gen = gen;
                found = true;
                break;
            }
            core::hint::spin_loop();
        }

        if !found {
            // Slow path: condvar wait (mutex only protects shutdown flag)
            let mut shutdown = match pool.state.lock() {
                Ok(s) => s,
                Err(p) => p.into_inner(),
            };
            while !*shutdown && pool.gen_atomic.load(Ordering::Relaxed) == last_gen {
                shutdown = match pool.work_cv.wait(shutdown) {
                    Ok(s) => s,
                    Err(p) => p.into_inner(),
                };
            }
            if *shutdown {
                return;
            }
            last_gen = pool.gen_atomic.load(Ordering::Acquire);
        }

        // Read dispatch data from atomics (ordered by gen_atomic Acquire)
        let fn_ptr = pool.fn_ptr_atomic.load(Ordering::Relaxed) as *const ();
        let fn_call: fn(*const (), usize, usize) = unsafe {
            core::mem::transmute(pool.fn_call_atomic.load(Ordering::Relaxed))
        };
        let n_threads = pool.n_threads_atomic.load(Ordering::Relaxed);

        fn_call(fn_ptr, tid, n_threads);
        pool.done_atomic.fetch_add(1, Ordering::Release);
    }
}

static SPAWNED_THREADS: AtomicUsize = AtomicUsize::new(0);

fn ensure_workers(pool: &Arc<ThreadPool>, n_threads: usize) {
    let spawned = SPAWNED_THREADS.load(Ordering::Relaxed);
    if spawned >= n_threads - 1 {
        return;
    }
    let start = spawned + 1;
    for tid in start..n_threads {
        let p = pool.clone();
        thread::Builder::new()
            .name(format!("qwen-worker-{}", tid))
            .spawn(move || pool_worker(p, tid))
            .expect("failed to spawn worker thread");
    }
    SPAWNED_THREADS.store(n_threads - 1, Ordering::Relaxed);
}

static THREAD_POOL_THREADS: AtomicUsize = AtomicUsize::new(1);

pub fn set_threads(n: usize) {
    let n = n.clamp(1, MAX_THREADS);
    THREAD_POOL_THREADS.store(n, Ordering::Relaxed);
    if n > 1 {
        let pool = get_pool();
        ensure_workers(pool, n);
    }
    if verbose() >= 2 {
        eprintln!("Thread pool: {} threads", n);
    }
}

thread_local! {
    /// Per-thread override of the effective kernel thread count. `0` means "no
    /// override" (fall back to the global pool count). Set to `1` by the
    /// parallel-segment decode workers so each concurrent segment runs its
    /// kernels single-threaded — `parallel_for`/`parallel_region` then take the
    /// inline `nt == 1` path and never touch the shared global pool, so `K`
    /// segment workers occupy `K` cores without contending on the one pool's
    /// dispatch/barrier machinery (the in-process analogue of L2's independent
    /// processes).
    static THREAD_OVERRIDE: Cell<usize> = const { Cell::new(0) };
}

/// Override the effective kernel thread count for the *current* thread only.
/// Pass `0` to clear. Internal knob for the parallel-segment path; not part of
/// any public API.
pub(crate) fn set_thread_override(n: usize) {
    THREAD_OVERRIDE.with(|c| c.set(n));
}

pub fn get_num_threads() -> usize {
    let ov = THREAD_OVERRIDE.with(|c| c.get());
    if ov != 0 {
        return ov;
    }
    THREAD_POOL_THREADS.load(Ordering::Relaxed)
}

pub fn get_num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(target_os = "macos")]
fn sysctl_uint(name: &[u8]) -> Option<usize> {
    debug_assert!(name.last() == Some(&0), "sysctl name must be NUL-terminated");
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
    if rc == 0 && out > 0 {
        Some(out as usize)
    } else {
        None
    }
}

/// Default kernel thread count when the user passes no `-t N`.
///
/// On Apple Silicon this reads the P-core count (`hw.perflevel0.physicalcpu`)
/// and E-core count (`hw.perflevel1.physicalcpu`) and returns
/// `P + min(E, P + (E - P) / 2)` (integer division): all performance cores plus
/// a bounded slice of the efficiency cores. When `E <= P` this reduces exactly
/// to the older `P + min(E, P)` (= `P + E`), so machines where we have no sweep
/// data — and where the old formula already used all/most cores — are left
/// untouched. When `E > P` it adds P E-cores plus half of the surplus:
/// e.g. M5 Pro 5P/10E → `5 + min(10, 5 + (10 - 5) / 2)` = `5 + min(10, 7)` = 12.
/// The result never exceeds `P + E` (each branch caps the E-core term at `E`).
///
/// Historically this returned P-cores only, on the assumption that efficiency
/// cores always hurt (extra dispatch + memory-bus contention). Round 10 made
/// the multi-token encoder/prefill GEMM phase pool-parallel (`sgemm_nt_pooled`),
/// so `P + min(E, P) = 10` threads then beat the 5-P-core default. Round 11's
/// dynamic work-stealing chunks (`parallel_for_dynamic`) went further: extra
/// E-cores now steal work proportionally instead of straggling on a fixed even
/// slice, which made *more* E-cores profitable. An M5 Pro sweep (t5..t15) put
/// the optimum at 12 threads (t12 −2.8% 3-mode avg vs the t10 default, WER
/// unchanged); t14+ oversubscribes the P+E cores against the process's
/// auxiliary/OS threads and regresses. Hence the new formula lands on 12 here.
/// NOTE: this 12-thread optimum is validated on M5 Pro (5P/10E) only; the shape
/// of the curve (best just below all-cores, cliff at t14+) is the machine-
/// specific part.
///
/// Falls back to the total CPU count on non-macOS or when the perflevel
/// sysctls are unavailable (e.g. Intel Macs), matching the previous behavior.
/// Clamped to [`MAX_THREADS`].
pub fn get_default_threads() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(p) = sysctl_uint(b"hw.perflevel0.physicalcpu\0") {
            let e = sysctl_uint(b"hw.perflevel1.physicalcpu\0").unwrap_or(0);
            return default_threads_formula(p, e).clamp(1, MAX_THREADS);
        }
    }
    get_num_cpus().clamp(1, MAX_THREADS)
}

/// `P + min(E, P + (E - P) / 2)` (integer division), computed with an explicit
/// branch so the `E - P` subtraction can never underflow the unsigned type.
///   E <= P  -> extra = E            -> P + E   (== old `P + min(E, P)`)
///   E >  P  -> extra = P + (E-P)/2  (always <= E, so result <= P + E)
#[inline]
fn default_threads_formula(p: usize, e: usize) -> usize {
    let extra = if e <= p { e } else { (p + (e - p) / 2).min(e) };
    p + extra
}

/// Run a closure in parallel using the persistent thread pool.
/// The closure takes (thread_id, n_threads).
pub(crate) fn parallel_for<F: Fn(usize, usize) + Send + Sync>(f: F) {
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

    // Publish dispatch data via atomics (Relaxed OK: gen_atomic Release provides ordering)
    pool.done_atomic.store(0, Ordering::Relaxed);
    pool.fn_ptr_atomic.store(&f as *const F as *const () as usize, Ordering::Relaxed);
    pool.fn_call_atomic.store(trampoline::<F> as usize, Ordering::Relaxed);
    pool.n_threads_atomic.store(n_threads, Ordering::Relaxed);
    // Release: ensures all stores above are visible to workers that Acquire gen_atomic
    pool.gen_atomic.fetch_add(1, Ordering::Release);

    // Wake workers that fell through to condvar wait
    // Lock scope is minimal: just notify, no data to write
    {
        let _guard = match pool.state.lock() {
            Ok(s) => s,
            Err(p) => p.into_inner(),
        };
        pool.work_cv.notify_all();
    }

    // Main thread does tid=0
    f(0, n_threads);

    // Wait for workers: spin on atomic done counter
    let expected = n_threads - 1;
    loop {
        if pool.done_atomic.load(Ordering::Acquire) >= expected {
            break;
        }
        core::hint::spin_loop();
    }
}

/// Dynamic (work-stealing) parallel loop over `n_items` fixed-size work items.
/// Every participating pool thread repeatedly grabs the next item index from a
/// shared atomic counter and runs `f(item)`, so faster cores (P-cores) process
/// more items than slower ones (E-cores) instead of each thread owning a fixed
/// even slice and the whole op stalling on the slowest slice. On a heterogeneous
/// P/E-core machine this removes the E-core straggler tax that a static even
/// split pays on every parallel op.
///
/// Work items are FIXED-SIZE and their boundaries do NOT depend on the thread
/// count or on which thread runs which item, so any deterministic per-item
/// computation yields results independent of scheduling. Callers pick the item
/// granularity so there are several items per thread (to allow stealing to
/// balance) while each item is still large enough to amortize one atomic RMW.
///
/// The counter lives on the caller's stack and is captured by the dispatched
/// closure — the same borrow pattern as [`parallel_for`]'s trampoline.
pub(crate) fn parallel_for_dynamic<F: Fn(usize) + Send + Sync>(n_items: usize, f: F) {
    let n_threads = get_num_threads();
    if n_threads <= 1 || n_items <= 1 {
        for i in 0..n_items {
            f(i);
        }
        return;
    }
    let counter = AtomicUsize::new(0);
    parallel_for(|_tid, _nt| loop {
        let i = counter.fetch_add(1, Ordering::Relaxed);
        if i >= n_items {
            break;
        }
        f(i);
    });
}

/// Cache-line-padded wrapper to keep two atomics on separate cache lines and
/// avoid false sharing between the arrival counter and the generation counter.
#[cfg(target_arch = "aarch64")]
#[repr(align(64))]
struct CachePadded<T>(T);

/// Spin barrier for use inside a [`parallel_region`]. Generation-counter based:
/// every participant increments `arrived`; the last arriver resets `arrived`
/// and bumps `generation` (Release), which the spinning participants observe
/// (Acquire). Reusable across many stages within one region. Correct for
/// `nt == 1` (immediate no-op) and for participants whose stage slice was empty
/// (they still call `wait` the same number of times as everyone else).
#[cfg(target_arch = "aarch64")]
pub(crate) struct RegionBarrier {
    arrived: CachePadded<AtomicUsize>,
    generation: CachePadded<AtomicUsize>,
    nt: usize,
}

#[cfg(target_arch = "aarch64")]
impl RegionBarrier {
    #[inline]
    fn new(nt: usize) -> Self {
        RegionBarrier {
            arrived: CachePadded(AtomicUsize::new(0)),
            generation: CachePadded(AtomicUsize::new(0)),
            nt,
        }
    }

    /// Block until all `nt` participants have called `wait`.
    #[inline]
    pub(crate) fn wait(&self) {
        if self.nt <= 1 {
            return;
        }
        let gen = self.generation.0.load(Ordering::Acquire);
        // AcqRel so the arrival RMWs form a release sequence: the last arriver
        // acquires every earlier participant's writes (e.g. tid-0 serial glue),
        // and its `generation` Release then publishes them to all waiters.
        let count = self.arrived.0.fetch_add(1, Ordering::AcqRel) + 1;
        if count == self.nt {
            // Last arriver: reset the counter, then open the gate.
            self.arrived.0.store(0, Ordering::Relaxed);
            self.generation.0.fetch_add(1, Ordering::Release);
        } else {
            while self.generation.0.load(Ordering::Acquire) == gen {
                core::hint::spin_loop();
            }
        }
    }
}

/// Run a closure once on every worker (plus the calling thread) in a single
/// thread-pool dispatch, keeping the workers resident for the whole closure.
/// The closure receives a shared [`RegionBarrier`] plus `(tid, nt)` and uses
/// `barrier.wait()` to synchronize between dependent stages — so a multi-stage
/// pipeline runs under one dispatch/wake/join cycle instead of one per stage.
#[cfg(target_arch = "aarch64")]
pub(crate) fn parallel_region<F: Fn(&RegionBarrier, usize, usize) + Send + Sync>(f: F) {
    let n_threads = get_num_threads();
    let barrier = RegionBarrier::new(n_threads);
    // Reuse the existing dispatch machinery: one dispatch, and the closure body
    // (with its internal barriers) is what keeps workers spinning across stages.
    parallel_for(|tid, nt| f(&barrier, tid, nt));
}

/// Even split of `total` items across `nt` workers; returns this worker's
/// `[start, end)` range (possibly empty).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn range_for(tid: usize, nt: usize, total: usize) -> (usize, usize) {
    let chunk = total.div_ceil(nt);
    let start = (tid * chunk).min(total);
    let end = (start + chunk).min(total);
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::default_threads_formula;

    #[test]
    fn default_threads_formula_matches_spec() {
        // (a) M5 Pro 5P/10E lands on 12.
        assert_eq!(default_threads_formula(5, 10), 12);
        // (b) E <= P reduces to the old `P + min(E, P)` = `P + E`.
        for (p, e) in [(4, 0), (4, 1), (4, 4), (8, 3), (10, 10), (6, 6)] {
            assert_eq!(default_threads_formula(p, e), p + e, "E<=P: {p}P/{e}E");
        }
        // (c) Never exceeds P + E.
        for p in 1..=16 {
            for e in 0..=24 {
                assert!(default_threads_formula(p, e) <= p + e, "{p}P/{e}E");
            }
        }
        // Spot checks of the E > P branch: P + P + (E-P)/2.
        assert_eq!(default_threads_formula(4, 8), 4 + 6); // 4 + min(8, 4+2)
        assert_eq!(default_threads_formula(2, 8), 2 + 5); // 2 + min(8, 2+3)
        assert_eq!(default_threads_formula(5, 11), 5 + 8); // 5 + min(11, 5+3)
    }
}
