# QwenASR Performance Optimizations

This document catalogs the performance optimizations implemented in the pure-Rust QwenASR CPU inference engine. Current HEAD (`d241af9b`) reaches 50× realtime on Apple M5 Pro for full-transcript offline transcription of the 28.2 s benchmark sample (563 ms median inference) — the fastest implementation in the cross-comparison, ahead of both MLX GPU baselines.

## 1. Memory Traffic & Allocation Reduction

- **INT8 Quantization for Decoder**: Decoder attention weights (`QKV`, `O`-projection), the FFN weights (interleaved `gate_up` + `down`), and `lm_head` are quantized to INT8 with per-row scales at load time, cutting single-token decode weight traffic ~4-8× below FP32. Implemented via NEON SDOT INT8 matvec/argmax kernels. (R11-I additionally group-quantized the FFN weights to INT4 for a ~20% further traffic cut, but the full-set WER gate later measured this as a +10.2% relative corpus-WER regression on dev-clean and it was reverted in R12-B5; see `experiments.md`.)
- **Reusable Workspaces**: Eliminated transient heap allocations in hot paths.
  - **Encoder**: `EncoderBuffers` persists scratch spaces for `chunk_mel`, convolution variables, and `im2col`. The main activation buffer (`x`) and `window_starts` metadata are reused across calls.
  - **Decoder**: `DecoderBuffers` provides pre-allocated scratch for BF16-to-F32 conversions, removing ~140 allocations per prefill pass.
  - **Transcription**: Embedding assembly buffers are reused instead of being reallocated per chunk.
  - **Decode hot path (R14-A1)**: The per-token INT8 quantization buffer is `thread_local` and fused SwiGLU kernels take caller scratch, eliminating all per-token heap traffic in the decode loop.
  - **Mel front-end (R14-A4b)**: Spectrogram scratch (windowing, DFT, power) is `thread_local` and reused across chunks.
- **Static Weight Prepacking**: Multi-token decoder prefill weights are preconverted from BF16 to F32 at load time and stored in a reusable matrix. This skips repetitive conversions across streaming chunks or segmented prefills.

## 2. Kernel Fusion & Cache Locality

- **Fused Residual Adds**: Replaced separate `y = y + x` loops with `linear_accumulate` and `linear_nobias_bf16_addto`. Matvec/GEMM operations accumulate directly into the destination residual buffer, saving read/write passes.
- **Fused Matvec + SwiGLU**: A fused kernel computes the `gate_up` projection and applies the `SwiGLU` activation in one pass, keeping intermediate values in L1 cache.
- **Head-Contiguous KV Cache**: Cache layout is `[layer][head][pos][head_dim]`. Storing heads contiguously improves spatial locality and reduces cache misses during causal attention scans.
- **Fused lm_head Argmax (R14-A2)**: The decode epilogue's INT8 `lm_head` argmax runs inside the fused decode region instead of as a separate pass (−2.8% segmented / −3.3% streaming decode time).

## 3. SIMD & Platform Acceleration

- **Explicit SIMD Intrinsics**: 
  - Vectorized `rms_norm`, `gelu`, and `swiglu` using fast polynomial exponential approximations.
  - RoPE uses NEON vector code for pairwise sub-vector rotations.
  - Bulk BF16 conversions use `vshll_n_u16` (NEON) and `_mm256_cvtepu16_epi32` (AVX2).
- **Apple Accelerate & vDSP**: Dense linear algebra (causal attention scores, mel spectrogram generation) is offloaded to Accelerate (BLAS). Uses `vvexpf` for batched softmax exponentiation and `vDSP_dotpr` for AMX coprocessor utilization.

## 4. Threading & Concurrency

- **Lock-Free Thread Pool Fast Path**: Work scheduling uses atomics and spin-waiting before falling back to mutex/condvar sleep, reducing OS context-switch latency for micro-jobs.
- **Default Thread Heuristic**: With no `-t N` override, the thread count defaults to `P + min(E, P + (E − P) / 2)` (integer division) on Apple Silicon — all performance cores plus a bounded slice of efficiency cores — clamped to 16; non-macOS/Intel falls back to the total CPU count. When `E <= P` this reduces to the older `P + min(E, P)` (= `P + E`), leaving untouched any machine we have no sweep data for. Once the multi-token GEMM phase became pool-parallel the efficiency cores had real work to share (M5 Pro 5P/10E → 10 threads beat the P-cores-only default); Round 11's dynamic work-stealing chunks then made *more* E-cores profitable, since they steal proportionally instead of straggling on a fixed even slice. A fresh t5..t15 sweep put the M5 Pro optimum at **12 threads** (−2.87% 3-mode average vs the t10 default, WER unchanged); t14+ oversubscribes the P+E cores against the process's auxiliary/OS threads and regresses, so the formula stays just below all-cores. The 12-thread optimum is validated on M5 Pro (5P/10E) only.
- **Pool-Parallel GEMM Slices**: A lone Accelerate `cblas_sgemm` call runs almost entirely on the calling thread, so multi-token GEMMs (`linear`, `linear_accumulate`, conv2d) split their output columns/channels across the persistent thread pool with one BLAS call per slice. Each output element remains a single full-K dot product, keeping results numerically stable; gated by size thresholds so single-token decode is untouched.
- **Dynamic Work-Stealing Chunks**: On heterogeneous P/E cores a static even split makes every parallel op wait on the slowest (E-core) slice while P-cores spin idle. `parallel_for_dynamic(n_items, f)` instead hands out **fixed-size** work items via a shared atomic counter, so faster cores drain more items and the op finishes near the balanced optimum. Item boundaries depend only on the problem size (not the thread count or schedule), so deterministic per-item work stays reproducible run-to-run. Used by the pooled GEMM slices (128-column items), conv2d im2col/GEMM, `bf16→f32` weight widening, the BF16 matvec/SwiGLU matvecs, threaded `gelu`/`swiglu`, and per-head bidirectional attention. ~5–7% faster across offline/segmented/streaming on M5 Pro, WER unchanged.
- **Threaded Non-Matmul Operations**: Parallelized operations beyond GEMMs:
  - `im2col` packing for encoder convolutions.
  - `gelu` and `swiglu` activations over large FFN buffers.
  - Bidirectional attention across attention heads.
  - Reshape/transpose of large activations (R14-A4c, threshold-gated) with vectorized bias add (~+2.8%).
- **Parallel Timestamped Transcription (R14-A5)**: The `--timestamps` paths run on the same parallel segment scheduler as plain segmented mode instead of a legacy serial loop (long-2min 1.35×, long-10min 1.60×; output differs only by near-tie token flips, same mechanism as the shipped L3 path).

## 5. Algorithmic Improvements

- **Silence Compaction**: Energy-based VAD preprocesses audio to strip non-speech segments. Edge padding is reduced to 2 windows and extra non-voice hangover is eliminated, minimizing data sent to the encoder.
- **Lazy Encoder Re-encoding**: In streaming mode, the partial encoder tail is only re-encoded every other chunk. This provides near-perfect Longest Common Prefix (LCP) reuse and reduces decoder prefill cost by ~50% on skipped chunks.
- **Online Softmax**: Single-token causal attention uses an online softmax scan, combining score tracking, normalization, and value accumulation into a single loop. This avoids temporary score buffer allocations and separate exponentiation passes for `seq_len = 1` queries.
- **Speculative Decoding with Greedy Verification**: Multi-token forward passes verify drafted tokens against the sequential greedy argmax in one weight stream, so accepted drafts are nearly-free tokens and output stays byte-identical. Two draft sources exist: streaming overlap drafts (R13, default ON — the previous chunk's tail predicts the re-decoded overlap) and offline prompt-lookup n-gram self-match (R14-B1, opt-in via `QWEN_ASR_VERIFY=1`, default OFF — measured acceptance on ASR text is low: ~8% of decode steps speculate on long audio, ≈3% fewer decode forwards).
