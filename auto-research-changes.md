# Performance Commit Notes

This file summarizes the major performance-oriented commits on `autoresearch/perf-opt-1`.

Notes:
- Benchmark numbers below are taken from `results.tsv` where a matching experiment row exists.
- `offline_time_ms` and `offline_rtf` are the most complete long-run metrics in this branch history.
- `b383a8f` is bookkeeping only; the rest are implementation changes.

## b383a8f - update result

- Scope: updates `results.tsv`.
- What changed: recorded later experiment outcomes.
- Why it helps: it does not improve runtime directly; it preserves the optimization history and the keep/revert decisions for later work.

## c0de131 - experiment 59: thread decoder prefill SwiGLU multiply

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: `swiglu_multiply()` was parallelized for large prefill buffers by splitting work across sequence rows.
- Why it improves performance: decoder prefill applies SwiGLU over large `[seq_len x intermediate]` buffers. That work is embarrassingly parallel, so spreading rows across worker threads reduces wall-clock time and keeps more CPU cores busy.
- Recorded result: experiment `59`, `1373ms` offline, `20.54x` realtime, status `kept`.

## 76c36f2 - experiment 56: thread im2col in conv2d + add profiling counters

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Added profiling counters for major kernels.
  - Parallelized the `im2col` packing step in `conv2d()`.
- Why it improves performance: the BLAS GEMM in convolution was already fast, but the data rearrangement step before GEMM was still serial. Threading `im2col` cuts preprocessing time for encoder convolutions. The added profiling made it easier to verify that `conv2d_op` was still a hotspot.
- Recorded result: experiment `56`, `1388ms` offline, `20.32x` realtime, status `kept`.

## 940f88d - experiment 53: thread GELU for large encoder FFN buffers

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: `gelu()` was threaded for large buffers, especially encoder FFN activations.
- Why it improves performance: encoder FFN layers apply GELU over large contiguous arrays. Once buffers are large enough, activation math becomes CPU time worth parallelizing, and the threading overhead is amortized.
- Recorded result: experiment `53`, `1468ms` offline, `19.21x` realtime, status `kept`.

## 7de7b4b - experiment 44: NEON-accelerated rms_norm_per_head for Q/K head norms

- Scope:
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/neon.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/neon.rs)
- What changed: added a NEON implementation for in-place per-head RMS normalization used on decoder Q and K vectors.
- Why it improves performance: this path runs on every decoder layer and every generated token. SIMD reduces scalar reduction and scale/multiply overhead, and per-head normalization is small enough that lowering instruction count matters.
- Recorded result: experiment `44`, `1504ms` offline, `18.75x` realtime, status `kept`.

## 6bfe117 - experiment 40: INT8 quantize all decoder attention weights (QKV + O-proj)

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Quantized decoder attention weights to INT8 with per-row scales.
  - Added INT8 kernels for Q/K/V projection and O-projection.
  - Switched single-token decode attention projection path to INT8.
- Why it improves performance: decoder attention matvecs are bandwidth-heavy and run every token. INT8 cuts weight bandwidth roughly 4x versus FP32 and significantly versus BF16-to-F32 conversion paths, improving cache fit and throughput.
- Recorded result: experiment `40`, `1565ms` offline, `17.98x` realtime, status `kept`.

## 1b57ac2 - experiment 39: INT8 quantized decoder FFN (gate_up + down projections)

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/neon.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/neon.rs)
- What changed:
  - Quantized decoder MLP weights to INT8.
  - Added NEON-backed INT8 matvec and INT8 SwiGLU support.
  - Moved gate/up and down projection work in single-token decode onto the INT8 path.
- Why it improves performance: decoder FFN projections dominate token generation cost. INT8 reduces memory traffic and lets NEON dot-product instructions handle more math per byte fetched.
- Recorded result: experiment `39`, `1650ms` offline, `17.10x` realtime, status `kept`.

## 4b698b4 - experiment 38: INT8 quantized argmax for vocabulary projection

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
  - [crates/qwen-asr/src/kernels/neon.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/neon.rs)
- What changed:
  - Quantized `lm_head` weights to INT8.
  - Added streaming argmax kernels that search `argmax(W @ x)` without materializing full logits in float.
- Why it improves performance: final vocabulary projection is large and memory-bound. INT8 lowers bandwidth and avoids building a full logits tensor just to select the max token.
- Recorded result: experiment `38`, `1813ms` offline, `15.56x` realtime, status `kept`.

## bed522a - experiment 34: fuse residual add into encoder sgemm (linear_accumulate)

- Scope:
  - [crates/qwen-asr/src/encoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/encoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Added `linear_accumulate()`.
  - Changed encoder residual branches so projection outputs are accumulated directly into the residual buffer.
- Why it improves performance: it removes separate post-matmul add passes over large encoder tensors and lets BLAS accumulate directly into the destination, saving memory traffic.
- Recorded result: experiment `34`, `1858ms` offline, `15.17x` realtime, status `kept`.

## 5e3d92f - experiment 24: lock-free thread pool fast path

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: reworked the thread-pool dispatch path so normal work scheduling uses atomics instead of locking, keeping mutex/condvar only as a slow path.
- Why it improves performance: tiny kernels and matvec slices were paying synchronization overhead. A lock-free fast path reduces wakeup and dispatch cost, which matters when many short parallel regions run during inference.
- Recorded result: experiment `24`, `1775ms` offline, `15.89x` realtime, status `kept`.

## 1090847 - experiment 23: hybrid spin-wait thread pool

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: workers now spin briefly looking for new work before falling back to condvar sleep.
- Why it improves performance: inference launches many back-to-back jobs. Short spinning avoids kernel sleep/wakeup latency when the next job arrives quickly, while still allowing sleep for longer idle periods.
- Recorded result: experiment `23`, `1845ms` offline, `15.28x` realtime, status `kept`.

## 2233b28 - experiment: default to performance cores only on Apple Silicon

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: default thread selection was biased toward performance cores on Apple Silicon.
- Why it improves performance: this workload is latency-sensitive and compute-heavy. Restricting execution to P-cores avoids slower E-core participation, which can reduce overall throughput because the parallel phases often wait for the slowest worker.
- Recorded result: experiment `20`, `1945ms` offline, `14.50x` realtime, status `kept`.

## 146df5c - experiment: fuse residual add into O-projection and down-projection matvec

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Added matvec helpers that add directly into an existing destination.
  - Switched decoder O-projection and FFN down-projection to fused residual-add forms.
- Why it improves performance: it removes two extra vector-add passes per decoder layer per token and keeps the destination hot in cache while projection results are produced.
- Recorded result: experiment `16`, `2130ms` offline, `13.24x` realtime, status `kept`.

## 9db81dc - experiment: NEON-vectorized RoPE (apply_rope_neox)

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: replaced scalar RoPE rotation math with NEON vector code for pairs of sub-vectors.
- Why it improves performance: RoPE is applied to Q and K on every decoder layer. SIMD executes the pairwise rotate-and-mix operations more efficiently and reduces scalar loop overhead.
- Recorded result: experiment `15`, `2140ms` offline, `13.18x` realtime, status `kept`.

## 2687065 - experiment: online softmax for single-token causal attention

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: replaced the single-token causal attention path with an online softmax scan that combines score tracking, normalization, and weighted value accumulation in one pass.
- Why it improves performance: for `seq_q = 1`, BLAS launches and temporary score buffers cost more than the math itself. The online formulation avoids allocation, avoids a separate softmax pass, and scans KV once.
- Recorded result: experiment `14`, `2166ms` offline, `13.02x` realtime, status `kept`.

## 80baa6f - experiment: vectorized softmax in causal attention via vDSP vvexpf

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: switched exponentiation in softmax-heavy attention code to Apple Accelerate `vvexpf`.
- Why it improves performance: exponentiation is one of the more expensive scalar operations in softmax. `vvexpf` batches that work inside a tuned vector math library.
- Recorded result: experiment `11`, `2167ms` offline, `12.99x` realtime, status `kept`.

## bd96813 - experiment: fuse gate_up matvec + SwiGLU in single-token decode

- Scope:
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Added a fused kernel that computes gate/up projection and immediately applies SwiGLU.
  - Replaced separate gate/up materialization plus activation with one tighter path.
- Why it improves performance: it reduces intermediate buffer traffic and keeps gate/up values hot in L1 cache instead of writing and rereading a larger temporary.
- Recorded result: experiment `10`, `2231ms` offline, `12.62x` realtime, status `kept`.

## 33864f8 - experiment: batched BLAS sgemm for mel spectrogram computation

- Scope:
  - [crates/qwen-asr/src/audio.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/audio.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Reworked mel spectrogram generation to batch all frames together.
  - Used matrix multiplication for DFT cosine/sine passes and mel filter-bank application.
- Why it improves performance: the old approach repeated lots of small per-frame work. Batching turns the problem into larger dense GEMMs that Accelerate handles efficiently, improving cache use and reducing interpreter-like loop overhead in Rust.
- Recorded result: experiment `9`, `2272ms` offline, `12.40x` realtime, status `kept`.

## 70db51f - experiment: head-contiguous KV cache layout for cache-friendly attention

- Scope:
  - [crates/qwen-asr/src/context.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/context.rs)
  - [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
  - [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed:
  - Changed KV cache layout to `[layer][head][pos][head_dim]`.
  - Added helpers for head-stride addressing and updated attention kernels to consume the new layout.
- Why it improves performance: causal attention walks one head across many positions. Making each head’s history contiguous improves spatial locality and reduces cache misses during KV scans.
- Recorded result: experiment `8`, `2501ms` offline, `11.26x` realtime, status `kept`.

## 1d423b5 - experiment: NEON-accelerated token embedding + eliminate final norm allocation

- Scope: [crates/qwen-asr/src/decoder.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/decoder.rs)
- What changed:
  - Switched token embedding conversion to a NEON-backed BF16-to-F32 path.
  - Reused an existing buffer for the decoder’s final RMS norm instead of allocating a fresh vector.
- Why it improves performance: token embedding lookup happens every generated token, and final normalization is also on the decode hot path. Faster BF16 conversion plus removing heap allocation trims recurring per-token overhead.
- Recorded result: experiment `6`, `2841ms` offline, `9.91x` realtime, status `kept`.

## 89c7283 - experiment: use BLAS sgemm for causal attention score/V computation

- Scope: [crates/qwen-asr/src/kernels/mod.rs](/Users/lizhuo/owork/q-asr/crates/qwen-asr/src/kernels/mod.rs)
- What changed: added BLAS-based matrix multiplication for the multi-token causal attention path, covering both score computation and value accumulation.
- Why it improves performance: for multi-token attention, the workload is dense enough that BLAS beats scalar loops. Offloading score and value matmuls to Accelerate reduces per-element Rust overhead and uses highly tuned kernels.
- Recorded result: experiment `2`, `2577ms` offline, `10.93x` realtime, status `kept`.

## Overall pattern

The biggest wins in this branch came from four themes:

- Moving decoder hot paths from BF16/FP32 to INT8.
- Fusing residual adds and activation steps to cut memory traffic.
- Using Accelerate BLAS/vDSP for dense linear algebra and vector math.
- Making thread scheduling and SIMD kernels cheaper on Apple Silicon.
