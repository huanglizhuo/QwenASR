#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
BINARY="$PROJECT_DIR/target/release/q-asr"
MODEL_DIR="$PROJECT_DIR/qwen3-asr-0.6b"
SAMPLES_DIR="$SCRIPT_DIR/samples"
LABEL=""
OUTPUT_DIR="$SCRIPT_DIR/results"
MODES="offline,segmented,streaming"
THREADS=""
RUNS=1

usage() {
    cat >&2 <<EOF
Usage: bench/run.sh [options]

  --binary PATH       Path to ASR binary (default: ./target/release/q-asr)
  --model-dir DIR     Model directory (default: qwen3-asr-0.6b)
  --samples-dir DIR   Audio samples directory (default: bench/samples)
  --label NAME        Label for this run (default: git short rev or timestamp)
  --output-dir DIR    Where to save results (default: bench/results)
  --modes LIST        Comma-separated: offline,segmented,streaming (default: all)
  --threads N         Thread count (default: system CPUs)
  --runs N            Repeat each test N times, take median (default: 1)
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
    THREAD_COUNT="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 0)"
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

# Helper: compute median of values (one per line)
median() {
    sort -n | awk '{a[NR]=$1} END {
        if (NR%2==1) print a[(NR+1)/2];
        else print (a[NR/2]+a[NR/2+1])/2;
    }'
}

# Helper: emit JSON string (escape quotes/backslashes/newlines)
json_str() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n' ' '
}

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
        CMD=("$BINARY" -d "$MODEL_DIR" -i "$wav" --profile)
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

        # Run (possibly multiple times)
        BEST_TOTAL_MS=""
        BEST_STDOUT=""
        BEST_STDERR=""

        for run_i in $(seq 1 "$RUNS"); do
            STDOUT_FILE="$(mktemp)"
            STDERR_FILE="$(mktemp)"

            if ! "${CMD[@]}" >"$STDOUT_FILE" 2>"$STDERR_FILE"; then
                echo "  FAILED (run $run_i)" >&2
                rm -f "$STDOUT_FILE" "$STDERR_FILE"
                continue
            fi

            # Parse timing
            this_total=$(bash "$SCRIPT_DIR/parse_stderr.sh" < "$STDERR_FILE" | grep '^total_ms=' | head -1 | cut -d= -f2 || true)

            # Keep best (lowest total_ms) run
            keep=false
            if [[ -z "$BEST_TOTAL_MS" ]]; then
                keep=true
            elif [[ -n "$this_total" ]] && awk "BEGIN{exit !($this_total < $BEST_TOTAL_MS)}" 2>/dev/null; then
                keep=true
            fi

            if $keep; then
                BEST_TOTAL_MS="$this_total"
                if [[ -n "$BEST_STDOUT" ]]; then rm -f "$BEST_STDOUT"; fi
                if [[ -n "$BEST_STDERR" ]]; then rm -f "$BEST_STDERR"; fi
                BEST_STDOUT="$STDOUT_FILE"
                BEST_STDERR="$STDERR_FILE"
            else
                rm -f "$STDOUT_FILE" "$STDERR_FILE"
            fi
        done

        if [[ -z "$BEST_STDOUT" ]]; then
            echo "  All runs failed, skipping" >&2
            continue
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

        # Write JSON result
        OUT_FILE="$RESULT_DIR/${base}_${mode}.json"
        cat > "$OUT_FILE" <<ENDJSON
{
  "version": "q-asr-bench-v1",
  "label": "$(json_str "$LABEL")",
  "binary": "$(json_str "$BINARY")",
  "git_rev": "$(json_str "$GIT_REV")",
  "timestamp": "$TIMESTAMP",
  "file": "$base.wav",
  "mode": "$mode",
  "threads": $THREAD_COUNT,
  "config": {
    "segment_sec": $SEGMENT_SEC,
    "model_dir": "$(json_str "$MODEL_DIR")"
  },
  "audio_duration_s": $audio_duration_s,
  "transcript": "$(json_str "$TRANSCRIPT")",
  "reference": "$(json_str "$REFERENCE")",
  "timing": {
    "total_ms": $total_ms,
    "encode_ms": $encode_ms,
    "decode_ms": $decode_ms,
    "tokens": $tokens,
    "tokens_per_sec": $tokens_per_sec,
    "realtime_factor": $realtime_factor
  },
  "profile": $PROFILE_JSON,
  "accuracy": {
    "wer": $WER,
    "cer": $CER,
    "levenshtein_words": $LEV_WORDS,
    "levenshtein_chars": $LEV_CHARS,
    "exact_match": $EXACT
  }
}
ENDJSON

        echo "  -> $OUT_FILE (${total_ms}ms, ${realtime_factor}x)"
        rm -f "$BEST_STDOUT" "$BEST_STDERR"
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
    'version': 'q-asr-bench-v1',
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
