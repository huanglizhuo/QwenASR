# QwenASR 性能优化清单

本文档汇总了 QwenASR 纯 Rust CPU 推理引擎中使用的核心优化技术。当前版本在 Apple M5 上对 28.2 秒基准样本的离线转写可达 63+× RTF。

## 1. 内存带宽与分配优化

- **Decoder INT8 量化**: Decoder 权重（`QKV`、`O` 投影、`FFN` gate/up/down、`lm_head`）在加载时量化为带逐行缩放因子的 INT8。相比 FP32，内存读写减少约 4 倍，并去除了 BF16 到 F32 的转换开销。由专用 NEON INT8 算子执行。
- **可复用工作区**: 消除了热点路径的堆内存分配。
  - **Encoder**: `EncoderBuffers` 持久化 `chunk_mel`、卷积变量和 `im2col` 矩阵。主激活张量 (`x`) 和 `window_starts` 在多次调用间复用。
  - **Decoder**: `DecoderBuffers` 预分配 BF16 到 F32 的转换空间，每次 Prefill 消除约 140 次分配。
  - **Transcription**: 流式模式直接复用 Embedding 组装缓冲区。
- **静态权重预转换**: 多 Token Decoder Prefill 权重在加载时从 BF16 转换为 F32 并缓存，避免流式或分段推理时重复转换。

## 2. 算子融合与缓存局部性

- **融合残差累加**: 移除独立的 `y = y + x` 循环。通过 `linear_accumulate` 和 `linear_nobias_bf16_addto` 算子，矩阵乘结果直接累加到残差目标缓冲区，每层减少一次读写。
- **融合 Matvec + SwiGLU**: 使用融合算子在计算 `gate_up` 投影的同时立即执行 `SwiGLU` 激活，使中间结果保持在 L1 缓存中。
- **Head 连续的 KV Cache**: KV Cache 内存布局改为 `[layer][head][pos][head_dim]`。在 Causal Attention 按 Head 遍历历史位置时，连续内存大幅降低 Cache Miss。

## 3. SIMD 与平台加速

- **显式 SIMD 内联汇编**:
  - `rms_norm`、`gelu` 和 `swiglu` 激活函数向量化，并使用多项式逼近指数运算。
  - RoPE 旋转位置编码使用 NEON 向量指令成对计算。
  - 使用 `vshll_n_u16` (NEON) 和 `_mm256_cvtepu16_epi32` (AVX2) 批量转换 BF16。
- **Apple Accelerate & vDSP**: 密集矩阵运算（注意力分数、Mel 频谱生成）卸载至 Accelerate (BLAS)。使用 `vvexpf` 批量计算 Softmax 指数，`vDSP_dotpr` 触发 AMX 协处理器。

## 4. 线程与并发

- **无锁线程池快速路径**: 任务调度优先使用原子操作和自旋等待，失败后再休眠，降低细粒度并行任务（如多头注意力）的系统调度延迟。
- **非矩阵运算并行化**:
  - Encoder 卷积的 `im2col` 数据打包。
  - 大型 FFN 缓冲区上的 `gelu` 和 `swiglu` 计算。
  - 跨 Head 的双向注意力计算。

## 5. 算法级优化

- **静音压缩**: 基于能量的 VAD 预处理剥离非语音片段。边缘填充降至 2 个窗口，并消除非语音拖尾，直接减少 Encoder 和 Decoder 输入。
- **流式惰性编码**: 流式模式下，局部的 Encoder 尾部仅每隔一个 Chunk 重新编码。提供最长公共前缀 (LCP) 复用，跳过重新编码的 Chunk 的 Decoder Prefill 开销降低约 50%。
- **在线 Softmax**: 单 Token Causal Attention 使用在线 Softmax 扫描，在单次循环内完成分数追踪、归一化和值累加，避免为 `seq_len = 1` 场景分配临时数组和二次遍历。

## 6. Round 15：工程优化与转写一致性检查

以下改动位于 `r15-speedups` 分支；长上下文 decode 退化调查和空闲机线程数复核仍是合并门禁。
各项测量及取舍见[实验记录](../research/experiments.md#speed-improvement-experiments--round-15-exact-only-audit-follow-up)。

| 项目 | 当前保留实现 |
|---|---|
| R15-1 | 对独立输出样本并行重采样，保留每个样本的运算顺序。 |
| R15-2 | 每个模型只解析一次 tokenizer，与权重加载重叠，并由各会话共享。 |
| R15-3 | macOS/aarch64 单 token decode 参与线程数为 `min(threads, max(8, threads - 2))`，可用 `QWEN_ASR_DECODE_THREADS` 覆盖；批量 decode 和 verify 保持原线程数。空闲机复核尚未完成。 |
| R15-5 | 符合条件的 CLI 多文件组走 lockstep 调度，否则回退到原逐文件 worker。 |
| R15-8 | ASR prefill 跳过末层不再使用的隐藏状态计算，保持 KV cache 一致；aligner 和完整 prefill 的语义不变。 |
| R15-10 | Causal prefill attention 按 128 个 query 分块，跳过被掩码的未来 key，减少临时内存与计算。 |
| R15-11 | 纯文本单文件 CLI 刷新输出后直接退出，省去模型析构。 |
| R15-13 | macOS/aarch64 的线程池 join 和 region barrier 在 yield 后重新启用原有的有限自旋预算，避免长时间等待时每次轮询都 yield。 |

[Round 15 测量工具](../../bench/r15/README.md)保存交替 A/B 和七用例逐字节检查。
端到端 wall time 与推理阶段耗时应同时报告，背景负载与空闲机结果分开记录；
改变浮点归约顺序的优化仍需通过全集准确率门禁。

批量 decode 后续补上了 `--profile` 分阶段计时，并改为互不重叠的 GQA 输出切片。
动态领取 attention 任务的实验保持逐位一致，但未测出可重复的端到端收益，因此仍使用静态调度；
详见[实验结果](../../bench/r15/history/batch-attention/README.md)。

长音频调查保留了上述等待退避修复：有后台负载的三组 10 分钟对照均加速，输出一致。
它修复的是早于 R15 的调度热点，尚未把历史 +21% 退化归因到某个 R15 提交；
详见[测量与限制](../../bench/r15/history/long-audio/README.md)，空闲机复核仍待完成。

11 月 6–7 日复测中，17 个 benchmark 用例与 main 输出一致。全集逐条比较中，双方 WER 均为 2.7591%，2703 条归一化文本全部一致，原始转写有一处逗号差异。[全集比较](../benchmarks/results.md#accuracy)。

## 7. Round 16：逐位一致的内核优化

三项改动与此前的 main 逐位一致：65 个用例的 encoder 输出与 prefill KV 哈希相同，七用例套件和 63 个扩展用例的转写逐字节相同。
测量数据与放弃的方案见[实验记录](../research/experiments.md#speed-improvement-experiments--round-16-exact-kernel-follow-up)。

| 项目 | 当前保留实现 |
|---|---|
| R16-1 | 在支持 SME 的 CPU（M4/M5）上，decoder 与 aligner 的 prefill 投影改用混合 GEMM：少数线程驱动手写的 FMOPA 内核，其余线程运行 NEON 内核；两者都按 k 递增、每个输出一条融合乘加链累加，与 Accelerate 多行 sgemm 的求和顺序相同。bf16 权重在打包时加宽。仅在打包后的激活能放进 E 簇 L2 时启用；`QWEN_ASR_SME=0` 可关闭。 |
| R16-2 | lockstep 批量 INT8 内核让两行权重对所有会话同时推进，每个权重向量只加载一次，最多 16 条 SDOT 链并行。 |
| R16-3 | encoder 的 `conv_out` 投影改为每组最多 160 行做一次 GEMM，不再每个 13 行的块各做一次；只有 1 个 token 的尾块仍单独计算，因为 Accelerate 对 1 行 GEMM 的求和顺序不同。 |

相对此前的 main：28 秒输入墙钟 −4.6…−5.2%（prefill −14%），推流式 −8.4%，10 分钟 `-S 30` −11.4%，12/60 文件批处理 −9.0% / −8.4%。
LibriSpeech dev-clean 全集 2703 条转写与此前的 main 逐字节一致（WER 2.7591%）。
