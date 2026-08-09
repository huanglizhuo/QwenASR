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
 *   B) 15 threads each issue sgemm(M=32, N=800, K=4320) over disjoint output
 *      rows of the same C, sharing one B. This is what QwenASR did before
 *      PR #51 -- and what stock OpenBLAS does not support: its thread server
 *      needs USE_LOCKING=1 for concurrent entry.
 *
 * Build:
 *   Linux:  gcc -O2 blas_concurrency_repro.c -o repro -lopenblas -lpthread
 *   macOS:  clang -O2 blas_concurrency_repro.c -o repro -framework Accelerate
 *
 * Run:  ./repro
 *
 * If B hangs or is far slower than A, this machine's BLAS is not safe to call
 * concurrently and PR #51's gating is the right fix. If B matches A, the BLAS
 * is fine and QwenASR's slowness on this machine has another cause.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

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

static void *slice_worker(void *arg) {
    long id = (long)arg;
    int start = (int)id * BLOCK;
    /* Disjoint output rows; every slice re-reads the whole shared B. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                BLOCK, N_DIM, K_DIM, 1.0f,
                W + (size_t)start * K_DIM, K_DIM, B, N_DIM,
                0.0f, C + (size_t)start * N_DIM, N_DIM);
    return NULL;
}

/* NSLICE threads inside BLAS at once, like QwenASR before PR #51. */
static void run_sliced(void) {
    pthread_t th[NSLICE];
    for (long i = 0; i < NSLICE; i++) pthread_create(&th[i], NULL, slice_worker, (void *)i);
    for (int i = 0; i < NSLICE; i++) pthread_join(th[i], NULL);
}

int main(void) {
    W = malloc(sizeof(float) * (size_t)M_TOTAL * K_DIM);
    B = malloc(sizeof(float) * (size_t)K_DIM * N_DIM);
    C = malloc(sizeof(float) * (size_t)M_TOTAL * N_DIM);
    if (!W || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill(W, (size_t)M_TOTAL * K_DIM, 1);
    fill(B, (size_t)K_DIM * N_DIM, 2);

    printf("shape C[%d,%d] = W[%d,%d] @ B[%d,%d], B = %.1f MB, %d reps\n",
           M_TOTAL, N_DIM, M_TOTAL, K_DIM, K_DIM, N_DIM,
           (double)K_DIM * N_DIM * 4 / (1024 * 1024), REPS);
    printf("A = 1 thread, one full call   (antirez/qwen-asr)\n");
    printf("B = %d threads, %d rows each   (QwenASR before #51)\n\n", NSLICE, BLOCK);

    run_single();                       /* warm up */
    double t0 = now_ms();
    for (int i = 0; i < REPS; i++) run_single();
    double single_ms = now_ms() - t0;
    printf("A  single call : %8.1f ms\n", single_ms);
    fflush(stdout);

    printf("B  %2d threads  : ", NSLICE);
    fflush(stdout);                     /* flush before the call that may hang */
    t0 = now_ms();
    for (int i = 0; i < REPS; i++) run_sliced();
    double sliced_ms = now_ms() - t0;
    printf("%8.1f ms\n\n", sliced_ms);

    printf("ratio B/A = %.2fx %s\n", sliced_ms / single_ms,
           sliced_ms > single_ms * 1.25 ? "  <-- concurrent entry is harmful here"
                                        : "  (no penalty on this BLAS)");
    free(W); free(B); free(C);
    return 0;
}
