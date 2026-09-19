# GPU 执行范围与审计

本文件区分源码审查、硬件观测和速度验收。Trace/计数器运行不是 benchmark 数字来源。

## CPU / GPU 边界

`src/gpu_candidate.rs` 仅从原 crate 调用 WAV 读取与 tokenizer，不创建 `QwenCtx`，
不调用原 `audio::mel_spectrogram`、Encoder、Decoder 或 CPU 矩阵 kernel。

`native/bridge.mm` 的 PCM → 结果路径：

1. PCM 上传后，`MetalRuntime::mel_spectrogram` 调度 DFT、Mel/filterbank/log 与归一化。
   Hann、DFT、Mel 常量表也由 GPU 在加载阶段生成。
2. `encode_mel` 在 GPU 上完成三层卷积与全部 Transformer 层、音频投影。
3. embedding gather、prompt tensor 拼接在 GPU 上完成；CPU 只提供 token ID。
4. `decoder_forward` 在 GPU 上完成 norm、投影、RoPE、KV 写入、Attention、SwiGLU 与残差。
5. logits 和分层 argmax 在 GPU 上完成。识别只读回下一个 token ID；对齐只读回
   timestamp 位置的 class ID。完成 command buffer 后才访问结果。
6. CPU 将 token 转文本，或将 class 乘以 80 ms 后进行原有 LIS/插值后处理。

CPU 还负责文件与包的 SHA256、形状/范围校验、内存分配与调度。这些不属于网络 tensor
计算。NumPy 只出现在离线转换和独立数值验证工具中，生产 FFI 没有 Python 依赖。
`qwen-asr-cpu-reference`、Encoder/Decoder reference 是单独的开发程序。

## 同步和临时缓冲区

使用显式 `MTLDispatchTypeSerial` 与 `MTLResourceHazardTrackingModeTracked`;不使用
untracked heap 或 concurrent encoder。长命令缓冲每 64 个 dispatch 自动 `submit()`
分块提交,让 GPU 尽早开始执行已提交块、主机并行编码后续块;块间顺序仍由串行命令队列
与 tracked 资源保证,与单一串行 encoder 的 RAW/WAR/WAW 语义完全相同(实测 untracked +
显式 barrier 与 tracked 串行无差异,故未引入)。每个 command buffer 在 `finish()` 中
等待全部块完成后才访问结果。

buffer pool 通过共享 lease 管理 Tensor/view 生命周期。最后一个 view 消失后,缓冲区
才可用于后续 GPU 命令复用;复用只发生在已编码的读之后,顺序由上述机制保证。
CPU-written 上传缓冲区不进池,因为 CPU 写入不受 GPU 命令顺序保护。替换成
concurrent/untracked 执行时必须重新设计依赖,不能沿用这个策略。

融合路径的数值范围:单 token 解码融合 kernel(norm+量化、RoPE+KV 直写、attention+量化、
GEMV+残差)与未融合序列在真实 prefill 状态下逐级字节相同(`perf_probe difftest`,
0 差异);依据是整数 GEMV 累加任意顺序精确、max 归约与顺序无关、相同舍入函数。
`gemv_w8a8_residual` 以 volatile 临时量阻断乘加的 FMA 收缩——否则每层输出漂移 1 ulp,
20 条中 1 条近 tie 标点会翻转。producer f16 直出与 cast_half 产生相同的 half 位。
分层 mel max 的初版曾把 simd_max 放在非均匀分支内(只归约 lane 0 的部分和),已修复;
当前两级 max 与顺序无关、精确一致。

## 已取得的硬件证据

本机 Apple M5 Pro,macOS 26.7,Xcode 27.1:

- 独立 `profile-check`/`perf-probe` 使用 GPU stage timestamp counters(每个 command
  buffer 独立 counter sample buffer——跨缓冲复用会让 resolveCounterRange 返回陈旧样本)。
  该模式把每个 kernel 放入独立 pass,改变调度开销,只用于找瓶颈。
- 逐 kernel counters 定位:融合前单 token step 482 kernel、quantize/norm/rope/copy 等
  琐碎 kernel 占约一半时间;融合后 257 kernel。`kstorm`(同 kernel 连发)与
  `stepstorm`(按真实顺序流过 28 层冷权重)区分 L2 命中与 DRAM:冷权重 step ≈3.8 ms,
  其中 GEMV 权重流 ~70%,已接近实测机器带宽(copy ≈258 GB/s,W8A8 GEMV ≈275 GB/s,
  lm_head 155.6 MB ≈ 0.56 ms/step)。
- 两次独立完整 7 轮 benchmark:ASR 48.51%、align 85.98%/86.15% 耗时降低;每条输出与
  CPU 逐位一致(对齐时间戳最大差异 0 ms)。Trace 完全在测速结束后运行,不将采样时间
  混入性能报告。
- 只导出目标进程聚合信息到 `bench/baselines/`;原始 trace/XML 留在被忽略的 `build/audit/`。

## 最新版双路径核验与剩余范围

- 最终运行 `gpu-fused-final-20260919`:ASR 5461.20 → 2811.81 ms(48.51% 降低,
  WER 1.0753%/CER 0.3891% 与 CPU 相同);align 4562.26 → 639.66 ms(85.98% 降低,
  时间戳 0 ms 差异)。候选输出全部确定性重复,稳定性门槛通过。
- 两条 CPU 原有对齐越界需确定处理方式,当前没有静默裁剪或跳过;它们是整体验收
  report 保持 failed 的唯一原因。
- 其他 M 系列芯片、语言、噪声和长音频尚未覆盖;不能将本机结果外推为全平台保证。

`enable_profiling` 仅由独立 `profile-check`/`perf-probe` 调用。生产 bridge 没有调用
入口;它不参与 benchmark。适配器自报 `backend=metal` 仅是协议字段,不单独构成 GPU
审计证据。
