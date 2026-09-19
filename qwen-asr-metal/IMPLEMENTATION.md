# GPU 实现与验收计划

## 固定范围

- macOS 26.4+、Apple Silicon，Metal 4 唯一 GPU 后端，无 CPU tensor fallback。
- 第一版同时覆盖 Qwen3-ASR-0.6B、Qwen3-ForcedAligner-0.6B。
- CPU 负责文件/音频 I/O、tokenizer、命令提交、文本及时间戳后处理。
- 最终 PCM → Mel → Encoder → Prefill → Decode/Classification 均在 GPU 计算。
- 权重优先 INT8，完成准确率闭环后评估 INT4；激活允许 FP16/BF16，归约保留 FP32。
- 当前 CPU 实现保持不变。它是回归参照；GPU 主机层仅复用 WAV/tokenizer，不调用其模型推理。

## 两个 0.6B 模型不是同一套 Encoder

由本仓库两个模型的 config 与现有 `QwenConfig::detect`/`forced_align` 核对：

| 参数 | 识别 0.6B | 强制对齐 0.6B |
|---|---:|---:|
| Encoder hidden | 896 | 1024 |
| Encoder layers | 18 | 24 |
| Encoder heads / head dimension | 14 / 64 | 16 / 64 |
| Encoder FFN | 3584 | 4096 |
| Decoder hidden / layers | 1024 / 28 | 1024 / 28 |
| Q heads / KV heads / head dimension | 16 / 8 / 128 | 16 / 8 / 128 |
| Decoder FFN | 3072 | 3072 |
| 输出 | 151936 个词表项 | 5000 个时间类别 |
| 输出计算方式 | Prefill 后自回归 greedy | 单次 Prefill，取 timestamp 位置分类 |
| 时间类别间隔 | 不适用 | 80 ms |
| embedding 与输出头共享 | 是 | 否 |

必须保留三层 stride-2 Conv2D、每个音频 chunk 的位置编码、Encoder attention window、
LayerNorm/GELU、音频投影、Decoder Q/K RMSNorm、RoPE 与 causal mask 等原模型语义。
不能把 Decoder 的 hidden=1024 当作 Q 输出宽度；Q 实际为 2048，K/V 各 1024，
拼接 QKV 为 4096。模型转换时验证每个 tensor 的名称、shape、dtype、bias 和共享关系。

## 模块边界

1. 模型包：版本化目录与 tensor 索引；区分量化语义和物理 packing；校验 source hash、
   group size、scale、zero point、padding、对齐要求。启动后不在热路径重排权重。
2. Runtime：Metal device/capabilities、pipeline cache、资源驻留、scratch arena、同步；
   先实现清楚的资源依赖，避免逐层 CPU 等待。新 Metal 4 command API 的生命周期和
   residency 必须单独验证；当前微基准只验证 shader/TensorOps，不代表该 Runtime 已完成。
3. Frontend/Encoder：GPU Mel、卷积、位置编码、窗口 attention、FFN、音频投影。
4. Decoder：Prefill 和 Decode 分离；量化 GEMM/GEMV 按实际形状选择；GPU KV 常驻。
5. 输出：识别 LM Head 分块 top1；对齐只对 timestamp 对应 hidden rows 做 5000 类
   projection/argmax，无需给所有位置生成全部 logits，也不执行 ASR greedy loop。
6. 接口：两个任务共享 JSONL 评测协议；生产 Rust API/FFI 在模型闭环验证后再定型。

对齐的 word/token interleave、timestamp 位置、80 ms 类别解码与 LIS 修复必须逐项对齐
现有实现。时间戳后处理不能通过裁剪制造“准确率通过”；边界异常需保留原始类别和
修复前后的序列，优先判断模型本身、CPU 实现或 GPU 数值差异的来源。

## 实施里程碑

| 阶段 | 工作 | 完成条件 | 当前状态 |
|---|---|---|---|
| B0 | 固定 20 条数据、CPU 双模型基线、评分与失败判定 | 可重跑，保留每次原始结果 | 已完成，发现 2 条对齐边界异常 |
| B1 | TensorOps 与量化 GEMV 微基准 | 实机编译、计算完成、数值校验、host/GPU 双计时 | 已有 INT8 TensorOps 与 INT4/8 GEMV 起点 |
| G1 | 版本化权重加载、GPU scratch、完整 Encoder | 与 CPU 中间激活比较，两个 Encoder 都通过 | 已实现；lease 缓冲池复用与原输出 bit-identical，20 条端到端已覆盖 |
| G2 | Prefill、KV、Q/K Norm、RoPE、Attention | 中间输出对照与跨 command KV continuation | 已实现，双模型 FP16/INT8 差分与拆分计算检查通过 |
| G3 | greedy ASR、timestamp classification | 20 条识别与对齐完整跑通 | 已跑通；混合 ASR 无逐条退化，FP16 对齐与基线一致 |
| G4 | GPU Mel 与完整数据常驻 | 从 PCM 到结果无 CPU tensor 计算 | 已接通；最新版两个任务已记录 Trace 并核对输出 |
| O1 | 量化、融合、布局与提交优化 | 两类延迟分别 ≤ CPU 的 70%，准确率门槛不变 | 速度与逐条回归达标：ASR -33.43%、align -77.32%；原有边界问题仍阻止整体验收 |
| V1 | 时间边界核验、trace、跨芯片验证 | 无未决准确率问题，公布各设备证据 | 未完成 |

先完成可验证的完整路径，再按真实分段耗时优化。必要时允许仅开发使用的浮点 GPU
参考路径来定位量化误差，不能把它误标为 INT4/INT8 生产模式。
kernel 数量不作为硬指标；融合以整个推理请求的时间和内存访问收益为依据。

## 准确性与性能门槛

固定规则以 `bench/PROTOCOL.md` 和 runner 实际判定为准。特别注意：

- 识别采用逐条 WER/CER 不退化，避免整体分数改善掩盖某句话变坏。
- 对齐默认逐词边界与 CPU 一致；有异常的 CPU 边界要先核验，不能视作人工真值。
- 识别、对齐分别验收，禁止把二者混合平均来掩盖某一任务未加速。
- 端到端 host 计时包括 GPU 完成等待、读回和后处理；不能只报 command enqueue 时间。
- 同轮交替比较，稳定性失败时保留数据但不能认证性能；不与历史最佳/最慢数字拼接。
- INT4 准确率失败时维持 INT8，不放宽准确率指标。没有证据前不承诺任何芯片均快 30%。
- 20 条用作开发门槛；后续扩大语言、噪声、长音频和人工对齐验证属于发布验收扩展。
