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
    let mut last_gen: u64 = 0;
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
/// and E-core count (`hw.perflevel1.physicalcpu`) and returns `P + min(E, P)`:
/// all performance cores plus up to an equal number of efficiency cores.
///
/// Historically this returned P-cores only, on the assumption that efficiency
/// cores always hurt (extra dispatch + memory-bus contention). That is no
/// longer true: once the multi-token encoder/prefill GEMM phase became
/// pool-parallel (round-10 `sgemm_nt_pooled`), the extra cores have real GEMM
/// work to do, and on M5 Pro (5P/10E) `P + min(E, P) = 10` threads beats the
/// 5-P-core default across offline/segmented/streaming with WER unchanged.
/// Capping the E-core contribution at `P` keeps us short of the over-subscribed
/// regime (all cores) that regressed in the same sweep.
///
/// Falls back to the total CPU count on non-macOS or when the perflevel
/// sysctls are unavailable (e.g. Intel Macs), matching the previous behavior.
/// Clamped to [`MAX_THREADS`].
pub fn get_default_threads() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(p) = sysctl_uint(b"hw.perflevel0.physicalcpu\0") {
            let e = sysctl_uint(b"hw.perflevel1.physicalcpu\0").unwrap_or(0);
            return (p + e.min(p)).clamp(1, MAX_THREADS);
        }
    }
    get_num_cpus().clamp(1, MAX_THREADS)
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
