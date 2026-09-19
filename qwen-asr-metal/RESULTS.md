# 最新结果:ASR 48.51%、对齐 85.98% 耗时降低;整体对齐验收仍待处理基线缺陷

在本机 **Apple M5 Pro / macOS 26.7**,固定 20 条英语 LibriSpeech 音频、1 轮完整预热
+ 7 轮同轮交替测量(640 次完整推理),2026-09-19 两次独立完整运行(中间无干扰)一致给出:

| 任务 | 当前 Rust CPU | GPU 实现 | 耗时降低 | 准确性 |
|---|---:|---:|---:|---|
| Qwen3-ASR-0.6B | 5461.20 ms | **2811.81 ms** | **48.51%** | WER/CER 与 CPU 完全一致 |
| Qwen3-ForcedAligner-0.6B | 4562.26 ms | **639.66 ms** | **85.98%** | 全部词语、时间戳与 Rust 一致(最大差异 0 ms) |

最终运行:`bench/runs/gpu-fused-final-20260919/`(可复现;同日另一次独立运行
`gpu-perf-optimized-20260919` 给出 48.51%/86.15%)。以上时间为整套 20 条的中位总时间,
包含 PCM 特征计算、Encoder、Decoder、GPU 完成等待、结果读回与后处理。
ASR 外层请求计时 5465.32 → 2815.55 ms;对齐 4566.68 → 643.76 ms。
两侧输出全部确定性重复,P95/中位数与稳定性检查均通过。

ASR 两边均为 **4/372 个词错误、6/1542 个字符错误**,WER 1.0753%、CER 0.3891%,
与 CPU 基线逐句完全相同;配对 bootstrap 耗时比值 95% 区间约 [0.498, 0.517]。
7 轮 × 20 条是开发回归证据,不外推为所有输入/芯片的保证。

## 本轮性能工作(均保持逐位一致)

1. **单 token 解码路径融合**:每层 15 个 kernel 降为 9 个——`norm_quant_i8`
   (RMSNorm+激活量化)、`rope_cache_f32`(Q/K RoPE 与 K/V 直写 KV cache)、
   `causal_attention_quant_i8`(单行 GQA attention+输出量化)、
   `gemv_w8a8_residual`(GEMV+残差)。整数累加任意顺序精确、max 归约与顺序无关;
   `gemv_w8a8_residual` 用 volatile 临时量阻止 FMA 收缩,保持与未融合路径相同的舍入。
   `native/perf_probe.mm difftest` 对比新旧路径中间张量为 0 字节差异,20 条输出逐位一致。
2. **全谱 Mel max 分层归约**:原单 32-lane 组串行扫描 12 万–30 万元素(最长 0.6 ms)改为
   两级归约;max 与顺序无关,结果精确一致(修掉了 simd 归约写在非均匀分支内的初版 bug)。
3. **f16 直出消除 cast_half**:norm/bias-GELU/attention/im2col/flatten/swiglu 等
   只被 GEMM 消费的 producer 直接输出 half(与 cast_half 相同的舍入位),Encoder/Prefill
   每次前向省去 326 次整精度往返。
4. **命令缓冲分块提交**:每 64 个 dispatch 自动 commit,主机编码与 GPU 执行重叠;
   串行队列 + tracked 资源保持与单一串行 encoder 完全相同的 RAW/WAR/WAW 顺序。

开发期用过的测量工具保留:`native/perf_probe.mm`(phases/counters/kstorm/stepstorm/
launch/roofline/difftest)与 `bench/quick.py`(GPU-only 快速迭代基准,对比已录制输出)。

## 实测机器上限与剩余空间

- dispatch 下限约 1.5 µs/kernel;实测流带宽约 258 GB/s(copy/add);W8A8 GEMV 实际达到
  ~275 GB/s(lm_head 155.6 MB/step ≈ 0.56 ms),已贴机器实际带宽。
- 单 token step 权重流量 567 MB(28 层 W8A8 + lm_head),字节下限 ≈ 2.1 ms/step;
  冷权重实测 step ≈ 3.8–4.2 ms,其中 kernel 执行 ~70%,其余为逐 kernel 调度周转。
- GEMM tile 变体(64×32 等)微基准噪声大、端到端无收益,已回退。
- **INT4 MLP 实验(`--precision hybrid4`)已评估并否决**:MLP 解码权重 W4 使 step 权重流
  量减少 263 MB,但 20 条 WER 1.6129%(基线 1.0753%,+50% 相对退化)只换来约 8% 端到端
  提速;代码保留为已否决实验(`models/asr-hybrid4` 为其本地包),不用于验收。

## 为什么总报告仍是 failed

原 Rust 对齐存在两条越界(与上一版相同,非本轮引入):

| 样例 | 最后一个词 | 音频长度 | CPU/GPU 结束时间 | 超出 |
|---|---|---:|---:|---:|
| 5895-34622-0004 | WHEELS | 4290 ms | 4640 ms | 350 ms |
| 251-118436-0013 | MAT | 8225 ms | 8320 ms | 95 ms |

协议要求与 CPU 完全一致且不越界一个 80 ms 桶;GPU 逐位复现 CPU 结果(时间戳差异 0 ms),
因此无法同时满足两条规则。**没有删除样本、裁剪时间戳或放宽门槛。**
ASR 比较已 `passed=true`;对齐仅有上述基线边界问题。处理决定仍待回复。

## 可复现证据

- 最终完整 summary:`bench/runs/gpu-fused-final-20260919/summary.json`;
  独立复跑:`bench/runs/gpu-perf-optimized-20260919/`。
- 模型、源模型、词表、全部运行时编译 shader、程序、PCM 的 SHA256 均由 runner 记录。
- 19 项 benchmark/资产追踪测试、8 项模型转换/加载拒绝测试、GPU GELU/argmax/QKV/SwiGLU
  融合回归、ops-check 全部通过。
- 当前 JSONL 入口面向英语、16 kHz 单声道、最长 30 s 的开发 benchmark;多语言、噪声、
  长音频、其他 M 系列芯片尚未认证。构建和运行见 `README.md`。
