#!/usr/bin/env bash
# Long-audio benchmark runner for q-asr (see bench/long/build_long_samples.py
# for how the samples are constructed). Style-consistent with bench/run.sh:
# per-mode JSON results with median-of-N inference/wall timing, peak child RSS
# via getrusage, and WER/CER scored against the sidecar reference using the
# shared LibriSpeech normalizer (bench/long/score_long.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$BENCH_DIR/.." && pwd)"

# Defaults
BINARY="$PROJECT_DIR/target/release/qwen-asr"
MODEL_DIR="$PROJECT_DIR/qwen3-asr-0.6b"
SAMPLES_DIR="$SCRIPT_DIR/samples"
LABEL=""
OUTPUT_DIR="$SCRIPT_DIR/results"
MODES="offline,segmented"
SAMPLES=""
THREADS=""
RUNS=3
SEGMENT_SEC=30

usage() {
    cat >&2 <<EOF
Usage: bench/long/run_long.sh [options]

  --binary PATH       Path to ASR binary (default: ./target/release/qwen-asr)
  --model-dir DIR     Model directory (default: qwen3-asr-0.6b)
  --samples-dir DIR   Long sample directory (default: bench/long/samples)
  --samples LIST      Comma-separated sample basenames (default: all *.wav)
  --label NAME        Label for this run (default: git short rev or timestamp)
  --output-dir DIR    Where to save results (default: bench/long/results)
  --modes LIST        Comma-separated: offline,segmented,streaming
                      (default: offline,segmented)
  --segment-sec N     Segment length for segmented mode (default: 30)
  --threads N         Thread count (default: performance cores)
  --runs N            Repeat each test N times, use median inference (default: 3)
  -h, --help          Show this help

Samples are built by bench/long/build_long_samples.py; each <name>.wav must
have a sidecar <name>.ref.txt reference transcript next to it.
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)      BINARY="$2"; shift 2;;
        --model-dir)   MODEL_DIR="$2"; shift 2;;
        --samples-dir) SAMPLES_DIR="$2"; shift 2;;
        --samples)     SAMPLES="$2"; shift 2;;
        --label)       LABEL="$2"; shift 2;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2;;
        --modes)       MODES="$2"; shift 2;;
        --segment-sec) SEGMENT_SEC="$2"; shift 2;;
        --threads)     THREADS="$2"; shift 2;;
        --runs)        RUNS="$2"; shift 2;;
        -h|--help)     usage;;
        *)             echo "Unknown option: $1" >&2; usage;;
    esac
done

# Resolve label / git rev
GIT_REV=""
if git -C "$PROJECT_DIR" rev-parse --short HEAD &>/dev/null; then
    GIT_REV="$(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
fi
if [[ -z "$LABEL" ]]; then
    LABEL="${GIT_REV:-$(date +%Y%m%d-%H%M%S)}"
fi

# Validate
if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found or not executable: $BINARY" >&2
    exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: model directory not found: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! -d "$SAMPLES_DIR" ]]; then
    echo "Error: samples directory not found: $SAMPLES_DIR" >&2
    echo "Build the long samples first: python3 bench/long/build_long_samples.py" >&2
    exit 1
fi

RESULT_DIR="$OUTPUT_DIR/$LABEL"
mkdir -p "$RESULT_DIR"

THREAD_FLAG=""
if [[ -n "$THREADS" ]]; then
    THREAD_FLAG="-t $THREADS"
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Collect wav files (optionally filtered by --samples)
WAV_FILES=()
while IFS= read -r f; do
    base="$(basename "$f" .wav)"
    if [[ -n "$SAMPLES" ]]; then
        keep=0
        IFS=',' read -ra WANTED <<< "$SAMPLES"
        for w in "${WANTED[@]}"; do
            if [[ "$base" == "$w" ]]; then keep=1; fi
        done
        [[ "$keep" -eq 1 ]] || continue
    fi
    WAV_FILES+=("$f")
done < <(find "$SAMPLES_DIR" -name '*.wav' -type f | sort)

if [[ ${#WAV_FILES[@]} -eq 0 ]]; then
    echo "Error: no matching .wav files in $SAMPLES_DIR" >&2
    echo "Build the long samples first: python3 bench/long/build_long_samples.py" >&2
    exit 1
fi

IFS=',' read -ra MODE_LIST <<< "$MODES"

echo "Long benchmark: label=$LABEL, binary=$BINARY, modes=$MODES, runs=$RUNS"
echo "Samples: ${#WAV_FILES[@]} files in $SAMPLES_DIR"
echo "Results: $RESULT_DIR"
echo ""

for wav in "${WAV_FILES[@]}"; do
    base="$(basename "$wav" .wav)"
    ref_file="${wav%.wav}.ref.txt"
    if [[ ! -f "$ref_file" ]]; then
        echo "Warning: no sidecar reference for $base ($ref_file), skipping" >&2
        continue
    fi

    for mode in "${MODE_LIST[@]}"; do
        echo "== $base / $mode =="

        CMD=("$BINARY" -d "$MODEL_DIR" -i "$wav")
        if [[ -n "$THREAD_FLAG" ]]; then
            CMD+=($THREAD_FLAG)
        fi
        MODE_SEGMENT_SEC=0
        case "$mode" in
            offline)    ;;
            segmented)  CMD+=(-S "$SEGMENT_SEC"); MODE_SEGMENT_SEC="$SEGMENT_SEC";;
            streaming)  CMD+=(--stream);;
            *)          echo "  Unknown mode: $mode, skipping" >&2; continue;;
        esac

        RUNS_TSV="$(mktemp)"
        for run_i in $(seq 1 "$RUNS"); do
            STDOUT_FILE="$(mktemp)"
            STDERR_FILE="$(mktemp)"

            timing_line="$(python3 - "$STDOUT_FILE" "$STDERR_FILE" "${CMD[@]}" <<'PY'
import platform, resource, subprocess, sys, time
stdout_file, stderr_file = sys.argv[1:3]
cmd = sys.argv[3:]
with open(stdout_file, "wb") as so, open(stderr_file, "wb") as se:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=so, stderr=se)
    t1 = time.perf_counter()
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if platform.system() == "Darwin":
    rss_kb = int(round(rss / 1024))
else:
    rss_kb = int(rss)
print(f"rc={proc.returncode} wall_ms={(t1 - t0) * 1000:.1f} peak_rss_kb={rss_kb}")
PY
)"
            rc="$(echo "$timing_line" | sed -n 's/.*rc=\([0-9]*\).*/\1/p')"
            this_wall="$(echo "$timing_line" | sed -n 's/.*wall_ms=\([0-9.]*\).*/\1/p')"
            this_rss="$(echo "$timing_line" | sed -n 's/.*peak_rss_kb=\([0-9]*\).*/\1/p')"

            if [[ "$rc" != "0" ]]; then
                echo "  FAILED (run $run_i)" >&2
                rm -f "$STDOUT_FILE" "$STDERR_FILE"
                continue
            fi

            this_total=$(bash "$BENCH_DIR/parse_stderr.sh" < "$STDERR_FILE" | grep '^total_ms=' | head -1 | cut -d= -f2 || true)
            if [[ -z "$this_total" ]]; then
                rm -f "$STDOUT_FILE" "$STDERR_FILE"
            else
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run_i" "$this_total" "$this_wall" "${this_rss:-0}" "$STDOUT_FILE" "$STDERR_FILE" >>"$RUNS_TSV"
                echo "  run $run_i: inference ${this_total}ms, wall ${this_wall}ms, rss ${this_rss}kb"
            fi
        done

        if [[ ! -s "$RUNS_TSV" ]]; then
            echo "  All runs failed, skipping" >&2
            rm -f "$RUNS_TSV"
            continue
        fi

        OUT_FILE="$RESULT_DIR/${base}_${mode}.json"
        python3 - "$RUNS_TSV" "$OUT_FILE" "$ref_file" "$LABEL" "$GIT_REV" "$TIMESTAMP" \
            "$base.wav" "$mode" "$MODE_SEGMENT_SEC" "$MODEL_DIR" "$BINARY" "$SCRIPT_DIR" "$BENCH_DIR" <<'PY'
import json, statistics, subprocess, sys
(runs_tsv, out_file, ref_file, label, git_rev, timestamp,
 wav_name, mode, segment_sec, model_dir, binary, script_dir, bench_dir) = sys.argv[1:14]

rows = []
with open(runs_tsv, "r", encoding="utf-8") as fh:
    for line in fh:
        run_i, total_ms, wall_ms, peak_rss_kb, stdout_file, stderr_file = line.rstrip("\n").split("\t")
        rows.append({
            "run": int(run_i),
            "total_ms": float(total_ms),
            "wall_ms": float(wall_ms),
            "peak_rss_kb": int(peak_rss_kb),
            "stdout": stdout_file,
            "stderr": stderr_file,
        })

median_total = statistics.median(row["total_ms"] for row in rows)
median_wall = statistics.median(row["wall_ms"] for row in rows)
median_row = min(rows, key=lambda row: (abs(row["total_ms"] - median_total), row["total_ms"]))

parsed = {}
proc = subprocess.run(["bash", f"{bench_dir}/parse_stderr.sh"],
                      stdin=open(median_row["stderr"], "rb"),
                      capture_output=True, text=True)
for line in proc.stdout.splitlines():
    if "=" in line:
        key, val = line.split("=", 1)
        parsed[key] = val

with open(median_row["stdout"], "r", encoding="utf-8") as fh:
    transcript = fh.read().strip()
with open(ref_file, "r", encoding="utf-8") as fh:
    reference = fh.read().strip()

acc_proc = subprocess.run(["python3", f"{script_dir}/score_long.py", ref_file],
                          input=transcript, capture_output=True, text=True)
accuracy = json.loads(acc_proc.stdout) if acc_proc.returncode == 0 else None

audio_s = float(parsed.get("audio_duration_s", 0) or 0)
data = {
    "version": "qwen-asr-long-bench-v1",
    "label": label,
    "binary": binary,
    "git_rev": git_rev,
    "timestamp": timestamp,
    "file": wav_name,
    "mode": mode,
    "config": {
        "segment_sec": int(segment_sec),
        "model_dir": model_dir,
        "run_isolation": "new_process_per_run",
        "cache_state": "os_page_cache_uncontrolled",
        "scoring": "librispeech_wer.py normalize/score (shared with short WER gate)",
    },
    "audio_duration_s": audio_s,
    "timing": {
        "statistic": "median",
        "total_ms": round(median_total, 3),
        "wall_ms": round(median_wall, 3),
        "encode_ms": float(parsed.get("encode_ms", 0) or 0),
        "decode_ms": float(parsed.get("decode_ms", 0) or 0),
        "tokens": int(float(parsed.get("tokens", 0) or 0)),
        "tokens_per_sec": float(parsed.get("tokens_per_sec", 0) or 0),
        "realtime_factor": round(audio_s / (median_total / 1000.0), 3) if median_total > 0 else 0,
        "inference_best_ms": min(row["total_ms"] for row in rows),
        "wall_best_ms": min(row["wall_ms"] for row in rows),
        "peak_rss_median_kb": int(statistics.median(row["peak_rss_kb"] for row in rows)),
        "peak_rss_max_kb": max(row["peak_rss_kb"] for row in rows),
        "runs": [
            {"run": row["run"], "total_ms": row["total_ms"],
             "wall_ms": row["wall_ms"], "peak_rss_kb": row["peak_rss_kb"]}
            for row in sorted(rows, key=lambda r: r["run"])
        ],
    },
    "accuracy": accuracy,
    "transcript": transcript,
    "reference": reference,
}
with open(out_file, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)
    fh.write("\n")

rtf = data["timing"]["realtime_factor"]
wer = accuracy["wer"] if accuracy else "n/a"
cer = accuracy["cer"] if accuracy else "n/a"
print(f"  -> {out_file}")
print(f"     median inference {median_total:.0f}ms, wall {median_wall:.0f}ms, "
      f"{rtf}x realtime, rss {data['timing']['peak_rss_median_kb']}kb, wer={wer}, cer={cer}")
PY

        # Clean up temp stdout/stderr files
        while IFS=$'\t' read -r _ _ _ _ stdout_path stderr_path; do
            rm -f "$stdout_path" "$stderr_path"
        done < "$RUNS_TSV"
        rm -f "$RUNS_TSV"
    done
done

echo ""
echo "Done. Results in $RESULT_DIR/"
