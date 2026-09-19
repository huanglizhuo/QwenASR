# GPU 实现进度(2026-09-19)

两个 0.6B 模型均已有完整 GPU 推理。最新无干扰 7 轮结果:**ASR 降低 48.51%,
强制对齐降低 86.0%**;每条识别错误数不增加,全部对齐时间戳与 Rust 一致(0 ms 差异)。
2026-09-19 性能专项:逐 kernel profile、单 token 解码融合、Mel max 分层、f16 直出、
命令缓冲分块;全部保持逐位一致。详见 `RESULTS.md` 与下文"性能专项"。
整体状态仍未完成验收:两条原有对齐越界的处理决定尚未收到;清单、原 Rust 与门槛均未修改。

## 性能专项(2026-09-19 晚)

测量手段(`make perf-probe`;`bench/quick.py` 为 GPU-only 迭代基准):

- GPU stage counters 定位逐 kernel 占比;修复跨 command buffer 复用 counter sample buffer
  导致 resolve 返回陈旧数据的问题(每个 command buffer 现使用独立 sample buffer)。
- `kstorm`/`stepstorm`:同 kernel 连发与按真实顺序流过全部 28 层权重,区分 L2 命中与
  冷 DRAM;launch 风暴测得 dispatch 下限 ≈1.5 µs,roofline 测得实际流带宽 ≈258 GB/s。
- `difftest`:对同一 prefill 状态逐级字节对比新旧 kernel 路径,是本轮保持逐位一致的
  直接证据。

改动与验证:

- 单 token 每层 kernel 数 15 → 9(norm+量化、RoPE+KV 直写、attention+量化、GEMV+残差融合);
  decode step 中位 5.75 → ≈4.2 ms。逐层输出 `difftest` 0 字节差异。
  两个数值陷阱已修复并记录:gemv+残差融合的 FMA 收缩(用 volatile 临时量阻断,否则每层
  1 ulp 漂移)、simd_max 写在非均匀分支内(初版分层 mel_max 只归约 lane 0 的部分和)。
- `quantize_activation_i8` 32→256 线程:max 与顺序无关,逐位一致;3072 宽行 28 µs → 1.6 µs。
- mel 全局 max 两级归约:最长 0.6 ms → 微秒级,结果精确一致。
- producer f16 直出:消除 Encoder/Prefill 326 次 cast_half;half(v) 与先存 f32 再
  cast_half 位相同。
- 命令缓冲每 64 dispatch 自动分块提交,主机编码与 GPU 执行重叠;对单步耗时中性,
  保留为结构性改进。untracked+barrier 与 tracked 串行 dispatch 实测无差异。
- 逐 token GEMV 已达 ~275 GB/s(≥ copy 实测带宽),lm_head 0.56 ms/step 为带宽下限。
- 实验否决记录:GEMM tile 64×32 等端到端无收益(微基准为机器状态噪声);
  INT4 MLP(`hybrid4`)WER 1.6129% vs 1.0753%,仅 +8% 速度,否决为默认路径。

## 突破性优化专项:8 项逐一实验(2026-09-19/20)

按"收益×可行性÷风险"优先级逐项实现/测量,统一以 10 条冻结子集交错 A/B 判定
(保留门槛:收益超噪声 ±2% 且输出可接受;否则放弃留档)。基准 10 条:asr 2791.0ms、
align 627.7ms(求和中位,同机器状态)。

| # | 方案 | 结果 | 依据 | 处置 |
|---|---|---|---|---|
| 8 | align 分词/GPU 重叠 | **放弃** | 实测 align 主机开销中位仅 0.82ms/条(elapsed 35.6 vs gpu 34.8ms),早期 14ms 已被融合优化消除;上限 ~2% < 噪声 | 未实现 FFI 拆分 |
| 9 | prefill/encoder 融合 QKV/gate+up GEMM | **放弃** | 转换器 `--fused-prefill` 生成融合权重(包大小不变)+split kernel;交错 2 轮:asr -0.4%/align +0.2%,皆噪声。逐位一致性通过(修掉 split_qkv 只写 q 前半的 bug;发现 m>1 时视图行跨度≠宽度的陷阱,逐层 GEMM 列归约不变故逐位一致) | 代码保留为存在性开关;新包 models/*-fused 留档 |
| 10 | encoder attention 每 4 查询共享 K/V 装载 | **放弃** | 交错 3 轮 encode 相位:h4 13.03/13.03/13.04 vs 旧 13.01/13.04/12.96ms——K/V 本就 L2 命中,瓶颈是在线 softmax 递推延迟;已回退 | kernel 删除 |
| 7 | gate_up+量化原子合一 | **放弃** | kstorm 实测 gate_up 17.58µs + quantize 1.55µs,融合上限 43µs/step ≈ 1% ASR < 噪声;自旋屏障有死锁风险,不值得 | 未实现 |
| 6 | Indirect Command Buffer | **放弃** | stepstorm 新增 cpu_encode 计时:0.21ms/step(5.5% of 3.86ms)——CPU 编码非瓶颈(与分块提交实验互证);ICB 还需 slot/start 参数间接化改造 | 未实现 |
| 3 | 分组 scale INT4 MLP(g=64) | **放弃(默认)** | 逐行→分组精度改善:WER 1.6129%→**1.3441%**(W8 基线 1.0753%),但速度收益从 8% 缩到 **5.6%**(组 scale 浮点累加耗 ALU),交换率更差;100 条审计标准下不可接受 | 内核/转换器保留,models/asr-hybrid4 留档 |
| 1 | 贪心投机解码 | **实测否决(无训练 draft 时)** | 已完整实现自链投机(bridge `QWEN_SPEC`、`decoder_forward_decode_batch` 走 tensor_w8a16,默认关闭):m=8 每批 9.5ms、875 kernel;**接受率结构性锁死 1**(批量行预测以本批 stale draft 为条件,断链后仅位置 0 可回收;spec=2/8 均恒接受 1)→ 9.5ms/token = 2.3× 变慢。早前"零接受保底 1.5×"的估算被两处实测推翻:批路径 dispatch 开销(875×~5µs)与未融合投影。结论:除非先训练 draft 模型(接受率≥2 才回本),否则不可行 | 代码留档于 env 开关后,默认关 |
| 5 | 持久 megakernel(单步/单块) | **未实施** | 预算不足以安全完成全局屏障原型(死锁验证+逐位 diff+多轮基准);投机实验顺带再证 dispatch 周转 ~5µs/个:批路径 875 kernel 的实测开销与单步 257 kernel 一致外推,预算 1.2-1.5ms/step 仍成立 | 需要专门立项 |

结论:本轮 8 项中 6 项以测量数据否决、1 项确证可行性待立项、1 项未实施。默认路径
(W8A8 hybrid + bit-exact 融合)保持不变,最终验证 0 mismatch、19+8 测试通过。
**剩余两个真实机会:投机解码(≥1.5× decode)与 megakernel(≤25% step),均已量化。**

## 100 条语音准确率审计(2026-09-19)

`bench/accuracy100.py`(开发工具)从 dev-clean 另选 100 条(共 691.9 s,种子独立于
冻结清单),CPU 参考与 GPU 候选对同一 PCM 逐一推理:

- **词错误 42/42、字错误 66/66 完全相同,WER 2.1944%、CER 0.8020% 两侧一致。**
- 94/100 转写逐字节相同;6 条文本不同:4 条纯标点近 tie(词错 0=0),2 条专名近 tie
  (GPU 一条少 1 词、一条多 1 词,净零)。
- 该 CPU↔GPU 文本差是混合精度设计的固有算术差(fp16 Encoder/Prefill vs CPU fp32),
  早于本轮性能工作:冻结 20 条上也存在 1 条(3752-4944-0057,词错相同)。本轮优化
  与原 GPU 实现逐位一致,已在差异样本上用 `perf-probe difftest` 复核(0 字节差异)。
- 副带计时(单次、非正式):CPU 中位 340.8 ms/条,GPU 150.3 ms/条。
- 明细:`build/accuracy100/accuracy100.json` 与 `accuracy100-rows.json`(含每条
  PCM SHA256,WAV 不提交)。

## 完整实现

- 版本化模型包：FP16、逐行 INT8，以及 ASR hybrid（FP16 Encoder/Prefill/embedding，
  W8A8 单 token Decoder/输出头）。混合包 2,163,408,896 字节；模型不提交 Git。
- CPU 参考程序保留原 Rust 推理；GPU 程序仅复用 WAV 读取、tokenizer 和等价的标量
  timestamp LIS 后处理。GPU FFI 没有 CPU Encoder、Decoder、Mel 或矩阵回退。
- GPU 前端：400 点 DFT、periodic Hann、center reflect padding、Slaney 128-bin Mel、
  log/global-max/clamp。9.25 s 音频的 Mel 相对 L2 误差 0.00012126、最大绝对差 0.002692。
- 完整双 Encoder、28 层 Decoder、causal GQA、Q/K RMSNorm + NeoX RoPE、常驻 KV、
  greedy top1、timestamp hidden row gather + 5000 分类。只读回最终 token/class ID。
- ASR 与 Rust 一样 Prefill 到倒数第二个 prompt 位置，最后一个位置进入单 token 路径。
- lease 缓冲池复用 GPU 临时内存，上限 256 MiB；CPU-written upload 不进入池，避免
  CPU 写入覆盖尚未执行的 GPU 工作。Encoder 与 split Decoder 对照复用前输出完全相同。
- 单 token hybrid 的 QKV 共用一次激活量化；gate/up 与 SwiGLU 融合。三个实际模型层、
  三档输入幅度下与未融合路径比较通过。
- runner 除 executable/model/PCM 哈希，还校验包的源模型、词表、weights 以及所有
  运行时编译 shader 哈希。正式评测不与其他 GPU 任务并发。

## 端到端试验记录

以下都是完整 20 条样例、1 轮预热，前三组 3 轮、融合后 7 轮同轮交替测量，不拼接历史 CPU 时间。

| 方案 | ASR 耗时下降 | ASR WER（CPU 1.0753%） | 对齐耗时下降 | 结论 |
|---|---:|---:|---:|---|
| 全 INT8，未复用 buffer | 13.74% | 1.6129% | 72.12% | 两条识别退化、对齐时间戳差异，失败 |
| FP16，buffer pool | 5.20% | 1.3441% | 78.01% | 一条识别退化；对齐与 CPU 一致 |
| hybrid ASR + FP16 align，融合前 | 29.28% | 1.0753% | 77.59% | ASR/CER 逐条无退化，仍未到 30% |
| hybrid 融合后（7 轮） | 22.11% | 1.0753% | 77.61% | 数值/输出稳定；CPU 基线更快，ASR 仍未达标 |
| 分层 argmax，最终独立 7 轮 | **33.43%** | **1.0753%** | **77.32%** | 速度与逐条回归通过；对齐仍有原基线边界问题 |

全量原始结果在 `bench/runs/gpu-*-20260919/`，摘要快照为
`bench/baselines/gpu-development-20260919.json`。不同方案不能直接用各自 CPU 基线数值
横向解释绝对快慢；较早持续运行时 CPU 有波动；最终独立 7 轮的两边稳定性门槛均通过。
最新完整 summary 为 `bench/baselines/gpu-m5pro-clean-20260919.json`。

关键发现：原 Rust ARM NEON 权重量化对完整 8 元素块使用向零截断，激活使用 nearest-away。
全 INT8 GPU 包原本采用 ties-to-even 权重量化与 FP16 激活；两者不是同一算术。
hybrid 将兼容语义显式写入 `decode_quantization`，仅用于单 token 解码，不针对样例
替换文字或硬编码答案。原 Rust、清单与参考文本未修改。

## Decoder 数值验证

测试输入是 CPU 的 embedding，以隔离 Decoder；这不是生产 fallback。
`tools/check_decoder.py` 对 full prefill 和跨 command 的最后一 token 做差分：

| 模型/精度 | hidden 相对 L2 | logits 相对 L2 | argmax 与 CPU |
|---|---:|---:|---|
| ASR FP16 | 0.001277 | 0.001911 | 一致（单输出位置） |
| ASR INT8 | 0.12519 | 0.07341 | 一致（不足以认证准确率） |
| Aligner FP16 | 0.00015594 | 0.00007936 | 56 个位置一致 |
| Aligner INT8 | 0.0060673 | 0.0026762 | 56 个位置一致 |

四组 KV continuation 相对 L2 均小于 0.005 的回归门槛。INT8 的中间误差确实在完整
20 条任务中暴露了准确率问题，因此没有将中间 cosine 或单个 argmax 当作准确率通过。

## 数值证据

对固定样本 `5338-284437-0008`（9.25 s，925 Mel frames，121 audio tokens）运行：

| 模型/精度 | 相对 L2 差异 | cosine |
|---|---:|---:|
| ASR FP16 | 0.00058204 | 0.99999983 |
| ASR INT8 | 0.02769380 | 0.99961647 |
| Aligner FP16 | 0.00025500 | 0.99999997 |
| Aligner INT8 | 0.00765189 | 0.99997090 |

输出 shape 与 CPU 一致，所有值有限。这仅支持 Encoder 集成进展，不能证明量化后
识别/对齐准确率无损；最终判断仍需完整解码和 20 条真实任务比较。
INT8 ASR Encoder 的数值差异比 aligner 大，应在 Decoder 接通后重点观察 WER/CER。

同样本的阶段探针中，热 GPU Encoder 的 host 时间约 ASR 34 ms、aligner 40 ms，
CPU 参考分别约 64 ms、91 ms。它们不是交替 7 轮的正式性能比较，也不包括 Mel、
Decoder、tokenizer 或对齐后处理，**不能据此宣布 30% 总目标达成**。

证据文件在 `build/encoder-validation/*/summary.json`；汇总快照在
`bench/baselines/encoder-development-20260919.json`。

## 已定位并修复的数值问题

最初 Encoder 在第 14 层 FFN 首次产生 NaN。进一步缩小到真实 GELU 路径，
`build/ops-check native` 可稳定复现输入 `12` → `NaN`。原因是当前 Metal 快速
`tanh` 计算在 GELU 的 cubic argument 上溢出，而非模型读取或 attention mask。
对 |x|≥10 使用已经在 FP32 精度下饱和的等价 GELU 尾部，修复后 fused 与 standalone
GELU 回归测试都通过，完整双模型 Encoder 输出不再出现 NaN。
临时逐层同步/打印已移除，不存在调试路径改变当前计时。

## 验证与下一步

- benchmark 判定/资产追踪 19 项测试通过；量化舍入/零行/卷积/CPU 向零截断 5 项通过。
- 双模型 FP16/INT8 Encoder/Decoder 实机差分、GELU 极值、混合 W8A8 投影、融合回归已运行。
- GPU stage counters 定位到单 SIMD 词表 argmax 占约 16% 的单 token ticks；改为并行分层
  归约后，最终独立 7 轮达到目标。最新版双任务 Trace 已记录并与 benchmark 输出一致。
- 唯一当前验收待决事项：两条原有对齐越界的处理决定；未裁剪时间戳、未删除样本、未放宽门槛。
- 生产接口、多语言/噪声/长音频覆盖和其他 M 系列芯片仍需扩展；本轮只认证本机与固定语料。

可复现命令见 README。所有 GPU 实机证据目前仅来自 M5 Pro。

## qwen-asr-server M1/M2/M3-简版(2026-09-20)

Groq/OpenAI 兼容服务(`server/`,独立 cargo 包):axum + 常驻引擎子进程
(qwen-asr-gpu asr/align 双进程,JSONL 复用)+ 启动预热 + Bearer 强制。
管线:multipart file → ffmpeg 16k mono **pcm_s16le**(引擎只吃 s16,已修 f32 wav 陷阱)
→ >28s 硬切(0.4s 重叠)→ 每段 ASR → 文本逐词喂 aligner 自对齐 → 偏移拼接 +
重叠去重 → 标点/gap/15s 软上限分段 → verbose_json 全字段(text/words/segments/
vtt/word_count/transcription_info;NaN 消毒、end≤duration clamp、空文本短路)。
响应格式 verbose_json/json/text/vtt;错误 OpenAI 语义(401/400/413/415)。

实测:4.65s 单段全字段正确;48.9s 双块 139 词 7 段端到端 2.6s;损坏文件 415;
无 token 401;text 格式正确;静音 200(ASR 对纯静音有轻幻觉词——待 Silero)。

待办:silero-vad v6.2.2 接入(替硬切+静音跳过)、url 模式+SSRF 防护、队列 429、
Mac mini 实机部署。启动:`QWEN_ASR_TOKEN=x QWEN_METAL_ROOT=. server/target/release/qwen-asr-server`。

- 能量分割层(2026-09-20 第二轮):`speech_regions()`(20ms RMS+自适应噪声底+滞回)
  驱动切分——切点落静音、>1.5s 间隙跳过、±200ms padding;纯静音现返回空转写
  (不再幻觉)。已知小项:首词偶现 start==end(80ms 桶+裁剪偏移,契约合法,下轮
  与 silero v6.2.2 一并处理,接口已按替换设计)。

- 第三轮(2026-09-20):url 模式 + SSRF 防护(getaddrinfo 校验全部 IP 拒环回/私网/
  链路本地,curl --resolve 固定 IP 防 TOCTOU,https only,手动重定向≤3 跳逐跳校验,
  180s/200MB 上限);admission 信号量(1 跑 + 1 排队,排队>120s→429)+ 30min 请求超时;
  首词 start==end 修复(前置 padding 0.35s);**silero-vad v6.2.2(ort 2.0-rc10 静态
  onnxruntime)接入**:512 样本块 + 64 上下文、state[2,1,128] 有状态流式、双阈值滞回,
  替换 speech_regions,能量法保留为回退(QWEN_SILENO=0 可关;模型 server/assets/
  silero-vad.onnx 2.3MB,来自 v6.2.2 tag 源码树——该 tag 无 release 附件)。实测:
  短/长/静音全过,w0 时间戳正常,SSRF 拒 http 与环回,415/401/429 路径就绪。

## PodParrot 真实播客基准(2026-09-20,BBC Happy Pod 全集 26.5min)

应用真实路径对比:worker /asr(Cloudflare Whisper)vs qwen-asr-server(file 上传)。
音频经 cloudflared quick tunnel 供 worker 抓取;基准与音频归档 `bench/podbench/`。

| 指标 | worker /asr | qwen-asr-server |
|---|---|---|
| 词数 | 4173 | 4379 |
| 段数 | 967(Whisper 原生粒度) | 233(标点/间隙聚合) |
| duration | 1662.537s | 1662.582s(一致 ✓) |
| 耗时 | ~98s | **85s(0.05× 实时)** |
| 两引擎词级分歧 WER | — | 9.05%(0.6B vs Whisper-v3 正常差) |

契约差异:worker 归一化无顶层 `words`(词嵌在 segments 内)+ 有 `usage`;我方多顶层
`words`(兼容,app 读 segments.words)。修复:multipart 大文件 bug(显式 Err 传播 +
DefaultBodyLimit;15MB 曾被静默丢弃)。遗留可选:段粒度对齐 Whisper(~2s 细分)、
首词 80ms 桶 start==end。

## 长音频优化(2026-09-20,#1/#2 按优先级)

- #1 内存 PCM 直传(audio_b64 进 JSONL,去掉逐 chunk WAV 落盘/读盘):
  **85s → 57.5s(-32%)**,输出逐字节一致。
- #2 ASR/align 双进程流水(align(N-1) 与 ASR(N) 重叠):57.5 → 56.2-56.7s(仅 ~2%,
  align 本身 ~100ms/chunk 且与 ASR 争用同一 GPU,重叠空间天然小;保留,结构更优)。
- 26.5min 全集最终 **~56s = 0.034× 实时**;剩余耗时构成:ffmpeg 解码 ~10s + ASR GPU
  ~40s(单 token 解码带宽下限,见前文短音频专项)。#3(silero 跳过/28s 上限)已无余量。
  26.5min 集从 85s→56s 累计 1.5×;长音频优化到此收敛,后续增益只在 ASR kernel 本身。
