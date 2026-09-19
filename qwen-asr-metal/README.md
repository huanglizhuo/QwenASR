# ⚠️ 实验性项目(Experimental)

**本目录是一个实验性项目**:Qwen3-ASR/ForcedAligner 的 Apple Metal GPU 实现与自托管
ASR 服务,面向研究、开发验证与本机实验。**不提供任何稳定性、准确性或跨平台保证**,
API/模型包格式/内核接口均可能随时变动;尚未覆盖多语言、噪声、长时域与其他 M 系列芯片。
当前状态、量化证据与已否决实验见 `PROGRESS.md` / `RESULTS.md` / `GPU-AUDIT.md`。

# Qwen3-ASR Metal

面向 macOS 26.4+ / Apple Silicon 的独立 GPU 实现目录。目标是在相同输入、
解码策略和准确率约束下，**0.6B 识别与 0.6B 强制对齐分别降低至少 30% 延迟**：
`GPU 时间 / 当前 Rust CPU 时间 ≤ 0.70`，即至少约 1.43 倍吞吐速度。

已实现完整的 **PCM → GPU Mel → Encoder → Decoder → 转写/对齐** 路径，并开始真实
端到端评测。当前可复现的候选是 ASR 混合精度（FP16 Encoder/Prefill，单 token
Decoder 使用与原 ARM Rust 一致的 W8A8）和 FP16 强制对齐。它们都实际在 Metal
计算，CPU 只做 I/O、tokenizer、调度与结果后处理。

最新独立 7 轮配对实测：**ASR 耗时降低 48.51%，对齐降低 85.98%**；逐条 WER/CER
不退化，对齐时间戳与 Rust 完全一致(0 ms 差异)。本数字来自 2026-09-19 性能专项后的
两次独立完整运行。速度目标已达到，但原 Rust 的两条对齐越界仍
阻止整体验收，暂未修改门槛。最新版双路径已完成 Metal System Trace 与源码范围审核。
完整结果与待决事项见 `RESULTS.md`，证据仅来自本机 M5 Pro 与固定 20 条语料。

## 目录

| 路径 | 内容 |
|---|---|
| `src/cpu_reference.rs` | 直接调用现有 Rust 引擎的常驻进程基线，支持识别和对齐 |
| `bench/manifest.json` | 固定 20 条真实音频及人工参考文本、20 位说话人、SHA256 |
| `bench/benchmark.py` | 准备数据、运行基线、交替比较 CPU/GPU、生成 JSON/Markdown |
| `bench/PROTOCOL.md` | GPU 接入协议、计时边界与严格验收规则 |
| `bench/test_benchmark.py` | 评测计分与失败判定测试 |
| `native/probe.mm` | Metal 4 / TensorOps 运行时编译和数值校验 |
| `native/quantized.metal` | W8A16 TensorOps、W8A16/W4A16 专用 GEMV |
| `native/microbench.mm` | 识别/对齐实际矩阵形状的合成微基准，含边界 tile 校验 |
| `tools/convert_model.py` | 离线 INT8/FP16 模型包，去重、SHA256、对齐、量化元数据 |
| `native/runtime.*` | mmap 权重加载、校验、GPU tensor、归一化/矩阵/卷积/Attention 调度 |
| `native/encoder.mm` | 两个模型完整的 GPU Encoder（输入为 Mel） |
| `native/frontend.metal` | GPU DFT、Slaney Mel、全局归一化 |
| `native/decoder.*` | Prefill、GQA/RoPE、KV、W8A8 decode、融合 QKV/SwiGLU |
| `native/bridge.*` / `src/gpu_candidate.rs` | 原生 GPU FFI 与真实常驻 JSONL 候选进程 |
| `native/perf_probe.mm` | 开发专用性能探针:phase 计时、逐 kernel counters、kernel 风暴、冷权重 step、roofline、新旧路径逐字节 diff(`make perf-probe`) |
| `bench/quick.py` | GPU-only 快速迭代基准:对比已录制基线输出与耗时(非验收工具) |
| `tools/check_decoder.py` | CPU/GPU Decoder、跨 command 的 KV continuation 对照 |
| `tools/check_encoder.py` | 单音频 CPU/GPU Encoder 差分验证；不代替端到端验收 |
| `RESULTS.md` | 最新 7 轮结果与两个待决基线缺陷 |
| `GPU-AUDIT.md` | GPU 调用范围、同步策略与硬件 Trace 证据 |
| `PROGRESS.md` | 实现证据、数值差异与优化历史 |
| `IMPLEMENTATION.md` | 模型差异、实现顺序、验收里程碑 |
| `bench/BASELINE.md` | 本机首次完整实测及发现的问题 |

Cargo 使用独立 workspace，不修改现有两个 crate、根 workspace 或已发布接口。
`qwen-asr` 的完整推理只用于 CPU 参考程序；GPU 程序仅复用其 WAV 读取与 tokenizer，
不创建 `QwenCtx`，不调用原 Encoder/Decoder/Mel。

## 准备与运行

从仓库根目录：

```sh
make -C qwen-asr-metal reference
python3 qwen-asr-metal/bench/benchmark.py prepare
make -C qwen-asr-metal test
make -C qwen-asr-metal probe
make -C qwen-asr-metal microbench
python3 qwen-asr-metal/bench/benchmark.py run \
  --output qwen-asr-metal/bench/runs/cpu-first
```

需要本地 `qwen3-asr-0.6b/`、`qwen3-aligner-0.6b/`，以及
`librispeech-wer-bench/dev-clean-2/`。如数据在别处，`prepare --dataset /path/to/dev-clean`
即可。需要 Python 3.10+；音频准备需要 ffmpeg，benchmark runner 只用标准库，离线转换和数值对照工具依赖 NumPy。
模型和音频不重复提交进 Git。
`prepare` 校验源 FLAC 和解码后的 PCM，不能静默更换版本；原清单已存在时 `select`
拒绝覆盖。

默认 1 轮完整预热 + 7 轮实测，CPU 基线共 320 次推理，配对比较共 640 次推理。CPU 线程数采用现有引擎
默认检测；本机为 15。测量时保持供电和系统负载稳定，避免同时运行编译、其他推理或
微基准。结果目录必须是新的，工具不会覆盖历史结果。

结果包含逐次输出、逐句指标、整套 20 条样本的 P50/P95、常规 RTF（耗时/音频时长）、
模型/程序/PCM 哈希、机器与编译器版本、加载耗时、原始 stderr 及评测脚本快照。
不采集序列号、硬件 UUID 或个人环境变量。

## GPU 接入

生成候选模型（目标目录已存在时不可重复运行转换），编译并进行配对测试：

```sh
python3 qwen-asr-metal/tools/convert_model.py qwen3-asr-0.6b \
  qwen-asr-metal/models/asr-hybrid --precision hybrid
python3 qwen-asr-metal/tools/convert_model.py qwen3-aligner-0.6b \
  qwen-asr-metal/models/align-fp16 --precision fp16
make -C qwen-asr-metal gpu
mkdir -p qwen-asr-metal/models/hybrid-profile
ln -s ../asr-hybrid qwen-asr-metal/models/hybrid-profile/asr
ln -s ../align-fp16 qwen-asr-metal/models/hybrid-profile/align
python3 qwen-asr-metal/bench/benchmark.py run \
  --candidate '["qwen-asr-metal/target/release/qwen-asr-gpu","{task}","qwen-asr-metal/models/hybrid-profile/{task}","qwen-asr-metal/native"]' \
  --output qwen-asr-metal/bench/runs/new-comparison
```

`hybrid-profile` 只是两个实际模型包的路径映射，不改变精度声明。runner 校验转换源、
词表、实际 weights 和全部运行时 shader 的哈希，避免仅记录可执行文件却漏掉 kernel
变化。混合包约 2.16 GB，同时保留 Prefill FP16 与 Decode INT8 权重，尚未优化容量。
当前 JSONL 适配器限定英语、16 kHz 单声道、10 ms–30 s 音频和 greedy 解码，面向
benchmark；通用多语言 SDK 接口尚未定型。

CPU/GPU 按样本交替运行，不并发执行；ASR 请求不向候选程序提供参考答案。
两个任务必须分别达到 30% 耗时下降，逐句 WER/CER 不增加，对齐时间戳默认零差异。
任何异常、漏样本、计时不可信、输出不稳定或质量退化都不能通过。

20 条仅作为开发回归集：它们是英语干净语音，不能证明所有语言和噪声条件下无损。
没有人工时间边界标注，因此对齐结果衡量的是 CPU 回归一致性；同时单独检查时间顺序
与音频范围。当前 CPU 对齐存在两条越界记录，保留并显式阻塞对齐验收，等待进一步
核验，不能靠删样本或放宽阈值绕过。

## Metal 实验的解释

`probe` 和 `microbench` 用系统运行时编译 shader，未依赖缺失的独立 `metal`
工具链。系统 Metal 4 能力通过查询确认。本机 M5 Pro 已实际完成 FP16 TensorOps
以及 INT8 TensorOps / INT4、INT8 GEMV 数值检查。

微基准测量 10 类矩阵形状：单 token QKV/FFN/LM Head、小 batch QKV、Prefill、
两个 Encoder 的 FFN、5000 类对齐输出头与卷积后投影。INT4 暂只有专用 GEMV，
还没有 INT4 TensorOps 路径。两个量化格式使用相同、可精确表示的合成逻辑权重；
输出包含 GPU 时间和提交/等待在内的 host 时间，TensorOps 的 scale epilogue
也计入。热驻留权重和合成数据不能代表完整模型内存流量或量化准确率。

GPU 通过评测后，仍需用 Metal System Trace 和代码审查确认网络与音频特征计算
在 GPU 上完成；程序自报 `backend=metal` 不是充分证据。

## 模型转换和 Encoder 验证

离线工具依赖 `tools/requirements.txt` 中的 NumPy；生产 native Runtime 不依赖 Python。

```sh
python3 qwen-asr-metal/tools/convert_model.py qwen3-asr-0.6b \
  qwen-asr-metal/models/asr-int8 --precision int8
python3 qwen-asr-metal/tools/convert_model.py qwen3-aligner-0.6b \
  qwen-asr-metal/models/align-int8 --precision int8
make -C qwen-asr-metal reference runtime-check encoder-check ops-check
python3 -m unittest discover -s qwen-asr-metal/tools -p 'test_*.py' -v
python3 qwen-asr-metal/tools/check_encoder.py \
  --model qwen3-asr-0.6b --package qwen-asr-metal/models/asr-int8 \
  --audio qwen-asr-metal/bench/data/5338-284437-0008.wav \
  --output qwen-asr-metal/build/encoder-validation/asr-int8
```

转换器拒绝覆盖已有包。使用 `--precision fp16` 和不同目录生成开发参考包。
`check_encoder.py` 的 Mel 由 CPU 产生，以隔离 Encoder 数值误差；这个验证程序不能
冒充“纯 GPU 端到端推理”。GELU 的大激活 NaN 已有实机回归测试保护。
