#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
BINARY="$PROJECT_DIR/target/release/qwen-asr"
MODEL_DIR="$PROJECT_DIR/qwen3-asr-0.6b"
SAMPLES_DIR="$SCRIPT_DIR/samples"
LABEL=""
OUTPUT_DIR="$SCRIPT_DIR/results"
MODES="offline,segmented,streaming"
THREADS=""
RUNS=10
PROFILE=0

usage() {
    cat >&2 <<EOF
Usage: bench/run.sh [options]

  --binary PATH       Path to ASR binary (default: ./target/release/qwen-asr)
  --model-dir DIR     Model directory (default: qwen3-asr-0.6b)
  --samples-dir DIR   Audio samples directory (default: bench/samples)
  --label NAME        Label for this run (default: git short rev or timestamp)
  --output-dir DIR    Where to save results (default: bench/results)
  --modes LIST        Comma-separated: offline,segmented,streaming (default: all)
  --threads N         Thread count (default: binary policy)
  --runs N            Repeat each test N times, use median latency (default: 10)
  --profile           Enable kernel profile counters during measured runs
  -h, --help          Show this help
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)     BINARY="$2"; shift 2;;
        --model-dir)  MODEL_DIR="$2"; shift 2;;
        --samples-dir) SAMPLES_DIR="$2"; shift 2;;
        --label)      LABEL="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --modes)      MODES="$2"; shift 2;;
        --threads)    THREADS="$2"; shift 2;;
        --runs)       RUNS="$2"; shift 2;;
        --profile)    PROFILE=1; shift;;
        -h|--help)    usage;;
        *)            echo "Unknown option: $1" >&2; usage;;
    esac
done

# Resolve label
if [[ -z "$LABEL" ]]; then
    if git -C "$PROJECT_DIR" rev-parse --short HEAD &>/dev/null; then
        LABEL="$(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
    else
        LABEL="$(date +%Y%m%d-%H%M%S)"
    fi
fi

# Git rev (best effort)
GIT_REV=""
if git -C "$PROJECT_DIR" rev-parse --short HEAD &>/dev/null; then
    GIT_REV="$(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
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
if [[ ! -d "$SAMPLES_DIR" ]] && [[ ! -f "$SAMPLES_DIR" ]]; then
    echo "Error: samples directory not found: $SAMPLES_DIR" >&2
    exit 1
fi

RESULT_DIR="$OUTPUT_DIR/$LABEL"
mkdir -p "$RESULT_DIR"

THREAD_FLAG=""
if [[ -n "$THREADS" ]]; then
    THREAD_FLAG="-t $THREADS"
fi

# Get thread count for JSON
if [[ -n "$THREADS" ]]; then
    THREAD_COUNT="$THREADS"
else
    THREAD_COUNT=""
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SYSTEM_JSON="$(python3 - <<'PY'
import json, os, platform, subprocess

def sysctl(name):
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return ""

machine = platform.machine()
system = platform.system()
features = []
if machine == "arm64" or machine == "aarch64":
    features.append("NEON")
    if sysctl("hw.optional.arm.FEAT_DotProd") == "1":
        features.append("DotProd")
    if sysctl("hw.optional.arm.FEAT_I8MM") == "1":
        features.append("I8MM")
elif machine == "x86_64":
    cpu_features = set()
    if system == "Darwin":
        cpu_features.update(sysctl("machdep.cpu.features").split())
        cpu_features.update(sysctl("machdep.cpu.leaf7_features").split())
    else:
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("flags"):
                        cpu_features.update(line.split(":", 1)[1].split())
                        break
        except OSError:
            pass
    for name in ["avx", "avx2", "avx512f", "vnni", "amx_int8", "fma", "sse4_1"]:
        if name.upper() in cpu_features or name in cpu_features:
            features.append(name.upper())

payload = {
    "os": system,
    "os_release": platform.release(),
    "machine": machine,
    "cpu_brand": sysctl("machdep.cpu.brand_string") or platform.processor(),
    "logical_cpus": os.cpu_count() or 0,
    "performance_cores": int(sysctl("hw.perflevel0.physicalcpu") or 0),
    "efficiency_cores": int(sysctl("hw.perflevel1.physicalcpu") or 0),
    "features": features,
}
print(json.dumps(payload))
PY
)"

# Collect wav files
WAV_FILES=()
if [[ -f "$SAMPLES_DIR" ]]; then
    WAV_FILES=("$SAMPLES_DIR")
    SAMPLES_DIR="$(dirname "$SAMPLES_DIR")"
else
    while IFS= read -r f; do
        WAV_FILES+=("$f")
    done < <(find "$SAMPLES_DIR" -name '*.wav' -type f | sort)
fi

if [[ ${#WAV_FILES[@]} -eq 0 ]]; then
    echo "Error: no .wav files found in $SAMPLES_DIR" >&2
    exit 1
fi

echo "Benchmark: label=$LABEL, binary=$BINARY, modes=$MODES"
echo "Samples: ${#WAV_FILES[@]} files in $SAMPLES_DIR"
echo "Results: $RESULT_DIR"
echo ""


# Build mode list
IFS=',' read -ra MODE_LIST <<< "$MODES"

TOTAL=0
DONE=0
for _wav in "${WAV_FILES[@]}"; do
    for _mode in "${MODE_LIST[@]}"; do
        TOTAL=$((TOTAL + 1))
    done
done

for wav in "${WAV_FILES[@]}"; do
    base="$(basename "$wav" .wav)"
    ref_file="${wav%.wav}.txt"

    for mode in "${MODE_LIST[@]}"; do
        DONE=$((DONE + 1))
        echo "[$DONE/$TOTAL] $base / $mode"

        # Build command
        CMD=("$BINARY" -d "$MODEL_DIR" -i "$wav")
        if [[ "$PROFILE" -eq 1 ]]; then
            CMD+=(--profile)
        fi
        if [[ -n "$THREAD_FLAG" ]]; then
            CMD+=($THREAD_FLAG)
        fi

        SEGMENT_SEC=0
        case "$mode" in
            offline)    ;;
            segmented)  CMD+=(-S 30); SEGMENT_SEC=30;;
            streaming)  CMD+=(--stream);;
            *)          echo "  Unknown mode: $mode, skipping" >&2; continue;;
        esac

        # Run (possibly multiple times). Keep median inference run as the representative result.
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

            # Parse timing
            this_total=$(bash "$SCRIPT_DIR/parse_stderr.sh" < "$STDERR_FILE" | grep '^total_ms=' | head -1 | cut -d= -f2 || true)

            if [[ -z "$this_total" ]]; then
                rm -f "$STDOUT_FILE" "$STDERR_FILE"
            else
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run_i" "$this_total" "$this_wall" "${this_rss:-0}" "$STDOUT_FILE" "$STDERR_FILE" >>"$RUNS_TSV"
            fi
        done

        if [[ ! -s "$RUNS_TSV" ]]; then
            echo "  All runs failed, skipping" >&2
            rm -f "$RUNS_TSV"
            continue
        fi

        STATS_JSON="$(python3 - "$RUNS_TSV" <<'PY'
import json, statistics, sys

rows = []
with open(sys.argv[1], "r", encoding="utf-8") as fh:
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

rows.sort(key=lambda row: row["total_ms"])
median_total = statistics.median(row["total_ms"] for row in rows)
median_wall = statistics.median(row["wall_ms"] for row in rows)
median_row = min(rows, key=lambda row: (abs(row["total_ms"] - median_total), row["total_ms"]))
payload = {
    "median_run": median_row,
    "inference": {
        "median_ms": median_total,
        "mean_ms": statistics.fmean(row["total_ms"] for row in rows),
        "best_ms": rows[0]["total_ms"],
    },
    "wall": {
        "median_ms": median_wall,
        "mean_ms": statistics.fmean(row["wall_ms"] for row in rows),
        "best_ms": min(row["wall_ms"] for row in rows),
    },
    "memory": {
        "peak_rss_median_kb": int(statistics.median(row["peak_rss_kb"] for row in rows)),
        "peak_rss_max_kb": max(row["peak_rss_kb"] for row in rows),
    },
    "runs": [
        {
            "run": row["run"],
            "total_ms": row["total_ms"],
            "wall_ms": row["wall_ms"],
            "peak_rss_kb": row["peak_rss_kb"],
        }
        for row in rows
    ],
}
print(json.dumps(payload))
PY
)"
        BEST_STDOUT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["median_run"]["stdout"])' "$STATS_JSON")"
        BEST_STDERR="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["median_run"]["stderr"])' "$STATS_JSON")"
        BEST_WALL_MS="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall"]["median_ms"])' "$STATS_JSON")"

        if [[ -z "$THREADS" ]]; then
            THREAD_COUNT="$(sed -n 's/^Thread pool: \([0-9]*\) threads.*/\1/p' "$BEST_STDERR" | head -1)"
        fi

        TRANSCRIPT="$(cat "$BEST_STDOUT")"

        # Parse stderr
        PARSED="$(bash "$SCRIPT_DIR/parse_stderr.sh" < "$BEST_STDERR")"
        total_ms=$(echo "$PARSED" | grep '^total_ms=' | cut -d= -f2)
        encode_ms=$(echo "$PARSED" | grep '^encode_ms=' | cut -d= -f2)
        decode_ms=$(echo "$PARSED" | grep '^decode_ms=' | cut -d= -f2)
        tokens=$(echo "$PARSED" | grep '^tokens=' | cut -d= -f2)
        tokens_per_sec=$(echo "$PARSED" | grep '^tokens_per_sec=' | cut -d= -f2)
        audio_duration_s=$(echo "$PARSED" | grep '^audio_duration_s=' | cut -d= -f2)
        realtime_factor=$(echo "$PARSED" | grep '^realtime_factor=' | cut -d= -f2)

        # Defaults for missing values
        total_ms="${total_ms:-0}"
        encode_ms="${encode_ms:-0}"
        decode_ms="${decode_ms:-0}"
        tokens="${tokens:-0}"
        tokens_per_sec="${tokens_per_sec:-0}"
        audio_duration_s="${audio_duration_s:-0}"
        realtime_factor="${realtime_factor:-0}"
        wall_ms="${BEST_WALL_MS:-0}"

        # Profile ops → JSON object
        PROFILE_JSON="{"
        first=true
        while IFS='=' read -r key val; do
            if [[ "$key" == profile_* ]]; then
                op_name="${key#profile_}"
                op_name="${op_name%_ms}"
                if $first; then first=false; else PROFILE_JSON+=", "; fi
                PROFILE_JSON+="\"${op_name}_ms\": $val"
            fi
        done <<< "$PARSED"
        PROFILE_JSON+="}"

        # Accuracy
        REFERENCE=""
        WER="null"
        CER="null"
        LEV_WORDS="null"
        LEV_CHARS="null"
        EXACT="null"
        if [[ -f "$ref_file" ]]; then
            REFERENCE="$(cat "$ref_file")"
            ACC="$(echo "$TRANSCRIPT" | python3 "$SCRIPT_DIR/wer.py" "$REFERENCE" 2>/dev/null || echo "")"
            if [[ -n "$ACC" ]]; then
                WER=$(echo "$ACC" | sed -n 's/.*wer=\([^ ]*\).*/\1/p')
                CER=$(echo "$ACC" | sed -n 's/.*cer=\([^ ]*\).*/\1/p')
                LEV_WORDS=$(echo "$ACC" | sed -n 's/.*lev_words=\([^ ]*\).*/\1/p')
                LEV_CHARS=$(echo "$ACC" | sed -n 's/.*lev_chars=\([^ ]*\).*/\1/p')
                EXACT=$(echo "$ACC" | sed -n 's/.*exact=\([^ ]*\).*/\1/p')
            fi
        fi

        # Write JSON result via Python for safe serialization
        OUT_FILE="$RESULT_DIR/${base}_${mode}.json"
        python3 -c "
import json, sys
data = {
    'version': 'qwen-asr-bench-v1',
    'label': sys.argv[1],
    'binary': sys.argv[2],
    'git_rev': sys.argv[3],
    'timestamp': sys.argv[4],
    'file': sys.argv[5],
    'mode': sys.argv[6],
    'threads': int(sys.argv[7]) if sys.argv[7] else None,
    'system': json.loads(sys.argv[27]),
    'config': {
        'segment_sec': int(sys.argv[8]),
        'model_dir': sys.argv[9],
        'run_isolation': 'new_process_per_run',
        'cache_state': 'os_page_cache_uncontrolled',
    },
    'audio_duration_s': float(sys.argv[10]),
    'transcript': sys.argv[11],
    'reference': sys.argv[12],
        'timing': {
        'wall_ms': float(sys.argv[26]),
        'total_ms': float(sys.argv[13]),
        'encode_ms': float(sys.argv[14]),
        'decode_ms': float(sys.argv[15]),
        'tokens': int(float(sys.argv[16])),
        'tokens_per_sec': float(sys.argv[17]),
        'realtime_factor': float(sys.argv[18]),
    },
    'profile': json.loads(sys.argv[19]),
    'accuracy': {
        'wer': None if sys.argv[20] == 'null' else float(sys.argv[20]),
        'cer': None if sys.argv[21] == 'null' else float(sys.argv[21]),
        'levenshtein_words': None if sys.argv[22] == 'null' else int(sys.argv[22]),
        'levenshtein_chars': None if sys.argv[23] == 'null' else int(sys.argv[23]),
        'exact_match': None if sys.argv[24] == 'null' else (sys.argv[24] == 'true'),
    },
}
with open(sys.argv[25], 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write('\n')
" \
            "$LABEL" "$BINARY" "$GIT_REV" "$TIMESTAMP" \
            "$base.wav" "$mode" "$THREAD_COUNT" "$SEGMENT_SEC" "$MODEL_DIR" \
            "$audio_duration_s" "$TRANSCRIPT" "$REFERENCE" \
            "$total_ms" "$encode_ms" "$decode_ms" "$tokens" "$tokens_per_sec" "$realtime_factor" \
            "$PROFILE_JSON" \
            "$WER" "$CER" "$LEV_WORDS" "$LEV_CHARS" "$EXACT" \
            "$OUT_FILE" "$wall_ms" "$SYSTEM_JSON"

        python3 - "$OUT_FILE" "$STATS_JSON" "$wav" <<'PY'
import json, sys, wave

path, stats_json, wav_path = sys.argv[1:]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh)
try:
    with wave.open(wav_path, "rb") as audio:
        data["audio_duration_s"] = audio.getnframes() / audio.getframerate()
    data["config"]["duration_source"] = "wav_header"
except wave.Error:
    # Older Python versions cannot read all WAV encodings accepted by the CLI.
    data["config"]["duration_source"] = "cli"
stats = json.loads(stats_json)
data["timing"]["statistic"] = "median"
data["timing"]["total_ms"] = round(stats["inference"]["median_ms"], 3)
data["timing"]["wall_ms"] = round(stats["wall"]["median_ms"], 3)
data["timing"]["realtime_factor"] = round(data["audio_duration_s"] / (stats["inference"]["median_ms"] / 1000.0), 3)
data["timing"]["inference_mean_ms"] = round(stats["inference"]["mean_ms"], 3)
data["timing"]["inference_best_ms"] = round(stats["inference"]["best_ms"], 3)
data["timing"]["wall_mean_ms"] = round(stats["wall"]["mean_ms"], 3)
data["timing"]["wall_best_ms"] = round(stats["wall"]["best_ms"], 3)
data["timing"]["peak_rss_median_kb"] = stats["memory"]["peak_rss_median_kb"]
data["timing"]["peak_rss_max_kb"] = stats["memory"]["peak_rss_max_kb"]
data["timing"]["runs"] = stats["runs"]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
timing = data["timing"]
print(f"  -> {path} ({timing['wall_ms']}ms median wall, {timing['total_ms']}ms median inference, {timing['realtime_factor']}x inference)")
PY

        while IFS=$'\t' read -r _ _ _ stdout_path stderr_path; do
            if [[ "$stdout_path" != "$BEST_STDOUT" ]]; then rm -f "$stdout_path"; fi
            if [[ "$stderr_path" != "$BEST_STDERR" ]]; then rm -f "$stderr_path"; fi
        done < "$RUNS_TSV"
        rm -f "$BEST_STDOUT" "$BEST_STDERR" "$RUNS_TSV"
    done
done

# Generate summary.json
echo ""
echo "Generating summary..."

SUMMARY_FILE="$RESULT_DIR/summary.json"

# Aggregate stats using awk across all result JSON files
python3 -c "
import json, glob, os, sys

result_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(result_dir, '*.json')))
files = [f for f in files if not f.endswith('summary.json')]

results = []
for f in files:
    with open(f) as fh:
        results.append(json.load(fh))

if not results:
    print('No results found', file=sys.stderr)
    sys.exit(1)

total_ms_vals = [r['timing']['total_ms'] for r in results if r['timing']['total_ms'] > 0]
encode_ms_vals = [r['timing']['encode_ms'] for r in results if r['timing']['encode_ms'] > 0]
decode_ms_vals = [r['timing']['decode_ms'] for r in results if r['timing']['decode_ms'] > 0]
rt_vals = [r['timing']['realtime_factor'] for r in results if r['timing']['realtime_factor'] > 0]
wer_vals = [r['accuracy']['wer'] for r in results if r['accuracy']['wer'] is not None]

by_mode = {}
for r in results:
    m = r['mode']
    if m not in by_mode:
        by_mode[m] = {'total_ms': [], 'realtime': [], 'wer': []}
    by_mode[m]['total_ms'].append(r['timing']['total_ms'])
    by_mode[m]['realtime'].append(r['timing']['realtime_factor'])
    if r['accuracy']['wer'] is not None:
        by_mode[m]['wer'].append(r['accuracy']['wer'])

def avg(lst):
    return sum(lst)/len(lst) if lst else 0

mode_summary = {}
for m, v in by_mode.items():
    mode_summary[m] = {
        'count': len(v['total_ms']),
        'avg_total_ms': round(avg(v['total_ms']), 1),
        'avg_realtime_factor': round(avg(v['realtime']), 2),
        'avg_wer': round(avg(v['wer']), 4) if v['wer'] else None
    }

summary = {
    'version': 'qwen-asr-bench-v1',
    'label': results[0]['label'],
    'git_rev': results[0]['git_rev'],
    'timestamp': results[0]['timestamp'],
    'binary': results[0]['binary'],
    'threads': results[0]['threads'],
    'total_files': len(results),
    'overall': {
        'avg_total_ms': round(avg(total_ms_vals), 1),
        'avg_encode_ms': round(avg(encode_ms_vals), 1),
        'avg_decode_ms': round(avg(decode_ms_vals), 1),
        'avg_realtime_factor': round(avg(rt_vals), 2),
        'avg_wer': round(avg(wer_vals), 4) if wer_vals else None,
    },
    'by_mode': mode_summary,
    'results': [os.path.basename(f) for f in files]
}

with open(os.path.join(result_dir, 'summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
    fh.write('\n')

print(f'Summary: {len(results)} results')
for m, v in sorted(mode_summary.items()):
    wer_str = f', WER={v[\"avg_wer\"]:.4f}' if v['avg_wer'] is not None else ''
    print(f'  {m}: {v[\"count\"]} files, avg {v[\"avg_total_ms\"]:.0f}ms, {v[\"avg_realtime_factor\"]:.2f}x realtime{wer_str}')
" "$RESULT_DIR"

echo ""
echo "Done. Results in $RESULT_DIR/"
