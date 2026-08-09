/* Does this machine's BLAS tolerate concurrent cblas_sgemm calls?
 *
 * Standalone: no model, no Rust, no qwen-asr source. It reproduces the two
 * arrangements at the encoder conv-stem's real shape and times both.
 *
 *   A) ONE thread issues one sgemm(M=480, N=800, K=4320).
 *      This is what antirez/qwen-asr does (qwen_asr_kernels.c, qwen_conv2d):
 *      serial im2col, then a single full-width BLAS call. OpenBLAS threads it
 *      internally.
 *
 *   B) A pool of `nproc` workers (capped at 16, like kernels::get_default_threads)
 *      pulls 15 slices of sgemm(M=32, N=800, K=4320) off a shared counter, over
 *      disjoint output rows of the same C, sharing one B. This is what QwenASR
 *      did before PR #51. Concurrency is bounded by the pool, not by the slice
 *      count; one thread per slice would just oversubscribe a small box and
 *      measure the scheduler instead.
 *
 * Build:
 *   Linux:  gcc -O2 blas_concurrency_repro.c -o repro -lopenblas -lpthread
 *   macOS:  clang -O2 blas_concurrency_repro.c -o repro -framework Accelerate
 *
 * Run:  ./repro
 *
 * WHAT THIS DOES NOT SHOW. It was written to isolate the #47 hang outside our
 * codebase, and it fails at that: on the very runner where the in-repo
 * `gemm_pooling_bench` conv-sliced config hangs in 4 of 5 runs, config B here
 * completes every time at a mild 1.24x. It also does not separate vendors --
 * 1.24x on EPYC/OpenBLAS vs 1.40x on a 3-core M1 VM, while the real kernel
 * hangs on one and gets *faster* on the other.
 *
 * The likely reason is that this uses plain pthread_create/join, whereas the
 * real dispatch goes through a persistent pool whose join spins on an atomic
 * with no yield (kernels/pool.rs). Concurrent BLAS entry may need that spinning
 * to wedge. So treat "B is only mildly slower" here as NOT exonerating a BLAS.
 *
 * WHAT IT IS GOOD FOR: column A, one thread issuing one full-width sgemm, is a
 * stable measure of how fast this BLAS is at the shape that dominates the
 * encoder. Reference values, 29 reps: 69 ms on an M5 Pro, 107 ms on a 3-core
 * M1 VM, 649 ms on a 4-core EPYC 7763 with libopenblas0-pthread 0.3.26. A
 * machine far off those numbers has a slow BLAS build, independent of any
 * concurrency question -- which is the open half of #47.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* Encoder conv layer 2 with enc_chunk_size=100, CONV_HIDDEN=480:
 * C[480,800] = W[480,4320] @ cols[4320,800]. cols is ~13.2 MB and shared. */
#define M_TOTAL 480
#define N_DIM    800
#define K_DIM   4320
#define BLOCK     32                     /* output rows per slice */
#define NSLICE  (M_TOTAL / BLOCK)        /* 15 concurrent BLAS calls */
#define REPS      29                     /* chunks in a 28.2 s clip */

static float *W, *B, *C;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void fill(float *p, size_t n, unsigned s) {
    for (size_t i = 0; i < n; i++) { s = s * 1103515245u + 12345u; p[i] = (float)((s >> 16) & 0xff) / 255.0f - 0.5f; }
}

/* One full-width call, exactly like antirez/qwen-asr's qwen_conv2d. */
static void run_single(void) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M_TOTAL, N_DIM, K_DIM, 1.0f,
                W, K_DIM, B, N_DIM, 0.0f, C, N_DIM);
}

/* Mirror QwenASR's `parallel_for_dynamic`: a fixed pool of `n_workers` threads
 * pulls slice indices off a shared counter. Concurrency is bounded by the pool
 * size, NOT by NSLICE -- spawning one thread per slice would oversubscribe a
 * small machine 5:1 and measure scheduler thrash instead of BLAS reentrancy. */
static int n_workers;
static int next_slice;
static pthread_mutex_t slice_lock = PTHREAD_MUTEX_INITIALIZER;

static void *slice_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&slice_lock);
        int i = next_slice++;
        pthread_mutex_unlock(&slice_lock);
        if (i >= NSLICE) return NULL;
        int start = i * BLOCK;
        /* Disjoint output rows; every slice re-reads the whole shared B. */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    BLOCK, N_DIM, K_DIM, 1.0f,
                    W + (size_t)start * K_DIM, K_DIM, B, N_DIM,
                    0.0f, C + (size_t)start * N_DIM, N_DIM);
    }
}

/* Up to n_workers threads inside BLAS at once, like QwenASR before PR #51. */
static void run_sliced(void) {
    pthread_t th[NSLICE];
    next_slice = 0;
    for (long i = 0; i < n_workers; i++) pthread_create(&th[i], NULL, slice_worker, (void *)i);
    for (int i = 0; i < n_workers; i++) pthread_join(th[i], NULL);
}

int main(void) {
    /* Match the pool QwenASR would use: online cores, capped at MAX_THREADS(16)
     * like kernels::get_default_threads. Override with QASR_REPRO_THREADS. */
    n_workers = (int)sysconf(_SC_NPROCESSORS_ONLN);
    const char *env = getenv("QASR_REPRO_THREADS");
    if (env && *env) n_workers = atoi(env);
    if (n_workers < 1) n_workers = 1;
    if (n_workers > 16) n_workers = 16;
    if (n_workers > NSLICE) n_workers = NSLICE;

    W = malloc(sizeof(float) * (size_t)M_TOTAL * K_DIM);
    B = malloc(sizeof(float) * (size_t)K_DIM * N_DIM);
    C = malloc(sizeof(float) * (size_t)M_TOTAL * N_DIM);
    if (!W || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill(W, (size_t)M_TOTAL * K_DIM, 1);
    fill(B, (size_t)K_DIM * N_DIM, 2);

    printf("shape C[%d,%d] = W[%d,%d] @ B[%d,%d], B = %.1f MB, %d reps\n",
           M_TOTAL, N_DIM, M_TOTAL, K_DIM, K_DIM, N_DIM,
           (double)K_DIM * N_DIM * 4 / (1024 * 1024), REPS);
    printf("A = 1 thread, one full call         (antirez/qwen-asr)\n");
    printf("B = %d workers over %d slices of %d rows  (QwenASR before #51)\n\n",
           n_workers, NSLICE, BLOCK);

    run_single();                       /* warm up */
    double t0 = now_ms();
    for (int i = 0; i < REPS; i++) run_single();
    double single_ms = now_ms() - t0;
    printf("A  single call : %8.1f ms\n", single_ms);
    fflush(stdout);

    printf("B  %2d workers  : ", n_workers);
    fflush(stdout);                     /* flush before the call that may hang */
    t0 = now_ms();
    for (int i = 0; i < REPS; i++) run_sliced();
    double sliced_ms = now_ms() - t0;
    printf("%8.1f ms\n\n", sliced_ms);

    /* Deliberately not labelled pass/fail: this ratio does not reliably
     * separate a BLAS that wedges from one that does not. See the header. */
    printf("ratio B/A = %.2fx\n", sliced_ms / single_ms);
    printf("\nCompare column A against the reference values in the header;\n"
           "the ratio is informational only and does not exonerate a BLAS.\n");
    free(W); free(B); free(C);
    return 0;
}
