# Round 16 benchmark refresh — 2026-11-12

This run measures commit `b6ed4ea4` (branch `r16-exact-kernels`, clean tree;
its code is identical to `7e2bdc2b`). The binary was built with
`cargo build --release -p qwen-asr-cli`: default features, `target-cpu=native`
from `.cargo/config.toml`, rustc 1.97.1. Its SHA256 is
`c86d2194333bbb578bbf8ea107c194faef14ff2c4c816ddef921991047e057d2`, the same
binary used for the full-set accuracy check below. Main `6a88ff62` was built
with the same toolchain (`bb24df72f681c515bc6e7d5beb919f79517ba2879212986f9d909fe678ea70a9`).

Hardware: Apple M5 Pro, 15 cores, 48 GiB, macOS 26.7. Desktop processes kept
running, including BTLEServer at about one core. The 1-minute load average was
2.0 when the comparison started and 15.3 at the end of the long-audio runs.
No iOS simulator was booted, unlike on November 6. Builds, tests and
benchmarks ran one at a time.

## Cross-implementation comparison

```sh
printf 'mlx-audio==0.4.5\n' > tmp/r16/bench/pip-constraints.txt
PIP_CONSTRAINT="$PWD/tmp/r16/bench/pip-constraints.txt" \
  bash bench/benchmark-all.sh --runs 10 --modes offline \
  --current-binary "$PWD/tmp/r16/bench/qwen-asr" \
  --tmp-dir "$PWD/tmp/r15/cross-2026-11-06" \
  --report-dir "$PWD/bench/compare-results/r16-2026-11-12" \
  --mlx-audio-model /Users/lizhuo/.cache/huggingface/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit/snapshots/89e96d92ba34aca20b3e29fb10cc284097d1219f
```

The script reused the November 6 external checkouts and virtual environments.
It rebuilt the first port, the C upstream (still `9246942`) and second-state
`v0.2.0`. All five implementations ran sequentially in one invocation; every
run succeeded. Results: [summary](cross/summary.json) ·
[all 50 timings](cross/runs.json) · [system](cross/system_info.json) ·
[comparison page](../comparison.md).

## Long audio versus main

```sh
python3 bench/r15/ab.py tmp/r16/bin/base-6a88ff62 tmp/r16/bench/qwen-asr \
  --cases 2min_offline,2min_seg30,10min_offline,10min_seg30 \
  --runs 3 --threads 12 --out tmp/r16/bench/long-ab
```

Each side ran one discarded warmup, then three alternating AB/BA pairs. All four
cases kept identical stdout on both binaries.

| Case | Current median inference | Current median wall | Current wall range | Main wall range | Paired median wall change |
|---|---:|---:|---:|---:|---:|
| 120.705 s offline | 4.042 s | 4.141 s | 3.742–4.251 s | 4.222–4.432 s | −4.09% |
| 120.705 s segmented (-S30) | 2.449 s | 2.562 s | 2.416–2.585 s | 2.470–2.652 s | −2.51% |
| 602.465 s offline | 48.158 s | 48.324 s | 46.839–54.454 s | 51.874–57.398 s | −14.95% |
| 602.465 s segmented (-S30) | 9.744 s | 9.910 s | 9.779–9.910 s | 10.991–11.108 s | −10.78% |

Only the 602 s segmented ranges do not overlap. The 602 s single pass is
load-sensitive: on November 6 the same case ranged 49.650–57.994 s. Its −15%
is therefore not an established gain. [Summary with every value](long/summary.json) ·
[metadata](long/metadata.json).

## Accuracy

The full LibriSpeech dev-clean offline pass (2,703 utterances, 12 threads)
used the same binary. WER is 2.7591% (1,501 / 54,402 words) and CER 0.7542%,
with no failures ([summary](wer-full.json)). Every raw hypothesis has the
SHA256 recorded per utterance in the November 6 full run
([2026-11-06/wer-full-utterances.jsonl](../2026-11-06/wer-full-utterances.jsonl)).
Tensor-level identity checks against main are in the Round 16 section of the
[experiment log](../../research/experiments.md).
