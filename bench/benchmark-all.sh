#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="$PROJECT_DIR/qwen3-asr-0.6b"
INPUT_FILE="$SCRIPT_DIR/samples/audio.wav"
RUNS=3
MODES="offline"
TMP_DIR="$PROJECT_DIR/tmp/benchmark-all"
REPORT_DIR=""
CHARTS_DIR="$SCRIPT_DIR/charts"
VENV_DIR="$SCRIPT_DIR/.venv-bench"
BASELINE_REF="bf52daf"
CURRENT_REF="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
C_REPO_URL="https://github.com/antirez/qwen-asr.git"
UPSTREAM_REF="main"
DO_CLEAN=0

usage() {
    cat >&2 <<EOF
Usage: $0 [options]

  --model-dir DIR             Model directory (default: ../qwen3-asr-0.6b)
  --input FILE                Input WAV file (default: ./samples/audio.wav)
  --runs N                    Number of runs per target/mode (default: 3)
  --modes LIST                Comma-separated modes (default: offline)
  --tmp-dir DIR               Temp/worktree directory (default: ../tmp/benchmark-all)
  --report-dir DIR            Output report directory (default: ./compare-results/<timestamp>)
  --charts-dir DIR            Stable chart output directory (default: ./charts)
  --venv-dir DIR              Python venv for benchmark tooling (default: ./.venv-bench)
  --baseline-ref REF          Baseline Rust ref (default: bf52daf)
  --current-ref REF           Current Rust ref (default: HEAD)
  --upstream-ref REF          Upstream C ref to clone/reset (default: main)
  --clean                     Remove tmp/worktree directory and exit
  -h, --help                  Show this help
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --input) INPUT_FILE="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --tmp-dir) TMP_DIR="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --charts-dir) CHARTS_DIR="$2"; shift 2 ;;
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        --baseline-ref) BASELINE_REF="$2"; shift 2 ;;
        --current-ref) CURRENT_REF="$2"; shift 2 ;;
        --upstream-ref) UPSTREAM_REF="$2"; shift 2 ;;
        --clean) DO_CLEAN=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

abs_path() {
    python3 - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

wav_duration_seconds() {
    python3 - "$1" <<'PY'
import sys, wave
with wave.open(sys.argv[1], "rb") as w:
    frames = w.getnframes()
    rate = w.getframerate()
print(frames / rate if rate else 0.0)
PY
}

log() {
    printf '[benchmark-all] %s\n' "$*"
}

remove_registered_worktree() {
    local path="$1"
    if git -C "$PROJECT_DIR" worktree list --porcelain | awk '/^worktree /{print $2}' | grep -Fxq "$path"; then
        git -C "$PROJECT_DIR" worktree remove --force "$path"
    elif [[ -d "$path" ]]; then
        rm -rf "$path"
    fi
}

cleanup_tmp() {
    remove_registered_worktree "$TMP_DIR/baseline-rust"
    remove_registered_worktree "$TMP_DIR/current-rust"
    rm -rf "$TMP_DIR/antirez-qwen-asr"
    mkdir -p "$(dirname "$TMP_DIR")"
}

ensure_python_env() {
    local requirements="$SCRIPT_DIR/requirements-bench.txt"
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        log "Creating benchmark venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    if ! "$VENV_DIR/bin/python" -c "import matplotlib" >/dev/null 2>&1; then
        log "Installing benchmark Python dependencies"
        "$VENV_DIR/bin/pip" install -r "$requirements" >/dev/null
    fi
}

MODEL_DIR="$(abs_path "$MODEL_DIR")"
INPUT_FILE="$(abs_path "$INPUT_FILE")"
TMP_DIR="$(abs_path "$TMP_DIR")"
CHARTS_DIR="$(abs_path "$CHARTS_DIR")"
VENV_DIR="$(abs_path "$VENV_DIR")"

if [[ $DO_CLEAN -eq 1 ]]; then
    cleanup_tmp
    log "Removed $TMP_DIR"
    exit 0
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Model directory not found: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Input file not found: $INPUT_FILE" >&2
    exit 1
fi
if [[ "$INPUT_FILE" != *.wav ]]; then
    echo "Input file must be a WAV file: $INPUT_FILE" >&2
    exit 1
fi
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$REPORT_DIR" ]]; then
    REPORT_DIR="$SCRIPT_DIR/compare-results/$TIMESTAMP"
fi
REPORT_DIR="$(abs_path "$REPORT_DIR")"
ROOT_REPORT="$PROJECT_DIR/report.md"
SUMMARY_DIR="$REPORT_DIR/normalized"
RAW_DIR="$REPORT_DIR/raw"
ROOT_SUMMARY="$REPORT_DIR/summary.json"
mkdir -p "$SUMMARY_DIR" "$RAW_DIR" "$TMP_DIR" "$CHARTS_DIR"

IFS=',' read -r -a MODE_LIST <<< "$MODES"
AUDIO_DURATION_S="$(wav_duration_seconds "$INPUT_FILE")"
BASELINE_SHORT="$(git -C "$PROJECT_DIR" rev-parse --short "$BASELINE_REF")"
CURRENT_SHORT="$(git -C "$PROJECT_DIR" rev-parse --short "$CURRENT_REF")"

ensure_worktree() {
    local path="$1"
    local ref="$2"
    remove_registered_worktree "$path"
    mkdir -p "$(dirname "$path")"
    git -C "$PROJECT_DIR" worktree add --force --detach "$path" "$ref" >/dev/null
}

ensure_c_clone() {
    local path="$1"
    if [[ -d "$path/.git" ]]; then
        log "Updating C repo clone"
        git -C "$path" fetch --depth 1 origin "$UPSTREAM_REF" >/dev/null
        git -C "$path" reset --hard "origin/$UPSTREAM_REF" >/dev/null
        git -C "$path" clean -fd >/dev/null
    else
        rm -rf "$path"
        log "Cloning C repo into $path"
        git clone --depth 1 --branch "$UPSTREAM_REF" "$C_REPO_URL" "$path" >/dev/null
    fi
}

write_result_json() {
    local outfile="$1"
    local impl="$2"
    local accel="$3"
    local mode="$4"
    local build_ok="$5"
    local run_ok="$6"
    local supports_mode="$7"
    local total_ms="$8"
    local realtime_factor="$9"
    local transcript="${10}"
    local note="${11}"
    local source_artifact="${12}"
    local commit_ref="${13}"
    local benchmark_date="${14}"
    local historical="${15}"

    python3 - "$outfile" "$impl" "$accel" "$mode" "$build_ok" "$run_ok" "$supports_mode" "$total_ms" "$realtime_factor" "$transcript" "$note" "$source_artifact" "$commit_ref" "$benchmark_date" "$historical" <<'PY'
import json, sys
outfile, impl, accel, mode, build_ok, run_ok, supports_mode, total_ms, rtf, transcript, note, source, commit_ref, benchmark_date, historical = sys.argv[1:]
def parse_num(v):
    if v in ("", "null", "None"):
        return None
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return None
payload = {
    "impl": impl,
    "accelerate": accel == "true",
    "mode": mode,
    "build_ok": build_ok == "true",
    "run_ok": run_ok == "true",
    "supports_mode": supports_mode == "true",
    "total_ms": parse_num(total_ms),
    "realtime_factor": parse_num(rtf),
    "transcript": transcript if transcript else None,
    "note": note if note else None,
    "source_artifact": source if source else None,
    "commit": commit_ref if commit_ref else None,
    "benchmark_date": benchmark_date if benchmark_date else None,
    "historical": historical == "true",
}
with open(outfile, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

normalize_rust_results() {
    local label="$1"
    local impl="$2"
    local accel="$3"
    local result_dir="$4"
    local commit_ref="$5"
    python3 - "$label" "$impl" "$accel" "$result_dir" "$SUMMARY_DIR" "$commit_ref" "$TIMESTAMP" <<'PY'
import glob, json, os, sys
label, impl, accel, result_dir, out_dir, commit_ref, benchmark_date = sys.argv[1:]
paths = [
    p for p in sorted(glob.glob(os.path.join(result_dir, "*.json")))
    if os.path.basename(p) != "summary.json"
]
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    mode = raw.get("mode", "offline")
    timing = raw.get("timing", {})
    payload = {
        "impl": impl,
        "accelerate": accel == "true",
        "mode": mode,
        "build_ok": True,
        "run_ok": True,
        "supports_mode": True,
        "total_ms": timing.get("total_ms"),
        "realtime_factor": timing.get("realtime_factor"),
        "transcript": raw.get("transcript"),
        "note": None,
        "source_artifact": path,
        "commit": commit_ref,
        "benchmark_date": benchmark_date,
        "historical": False,
    }
    out = os.path.join(out_dir, f"{label}-{mode}.json")
    with open(out, "w", encoding="utf-8") as g:
        json.dump(payload, g, indent=2, ensure_ascii=False)
PY
}

run_c_once() {
    local binary="$1"
    local mode="$2"
    local stdout_file="$3"
    local stderr_file="$4"
    python3 - "$binary" "$MODEL_DIR" "$INPUT_FILE" "$mode" "$stdout_file" "$stderr_file" <<'PY'
import subprocess, sys
binary, model_dir, input_file, mode, stdout_file, stderr_file = sys.argv[1:]
cmd = [binary, "-d", model_dir, "-i", input_file]
if mode == "segmented":
    cmd += ["-S", "30"]
elif mode == "streaming":
    cmd += ["--stream"]
with open(stdout_file, "wb") as so, open(stderr_file, "wb") as se:
    proc = subprocess.run(cmd, stdout=so, stderr=se)
print(f"{proc.returncode}")
PY
}

detect_c_mode_support() {
    local binary="$1"
    local mode="$2"
    local help_text
    help_text="$("$binary" -h 2>&1 || true)"
    case "$mode" in
        offline) return 0 ;;
        streaming)
            [[ "$help_text" == *"--stream"* ]]
            return
            ;;
        segmented)
            [[ "$help_text" == *"-S"* ]]
            return
            ;;
    esac
    return 1
}

benchmark_rust_target() {
    local label="$1"
    local impl="$2"
    local worktree="$3"
    local commit_ref="$4"
    local build_log="$RAW_DIR/$label/build.log"
    local output_root="$RAW_DIR/$label"
    local result_dir="$output_root/$label"
    mkdir -p "$output_root"

    log "Building $label"
    if ! /bin/zsh -lc "cd '$worktree' && cargo clean >/dev/null && RUSTFLAGS='-C target-cpu=native' cargo build --release" >"$build_log" 2>&1; then
        for mode in "${MODE_LIST[@]}"; do
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "$impl" true "$mode" false false true null null "" "Rust build failed; see $build_log" "$build_log" "$commit_ref" "$TIMESTAMP" false
        done
        return
    fi

    log "Running $label"
    if ! "$SCRIPT_DIR/run.sh" \
        --binary "$worktree/target/release/qwen-asr" \
        --model-dir "$MODEL_DIR" \
        --samples-dir "$INPUT_FILE" \
        --output-dir "$output_root" \
        --label "$label" \
        --modes "$MODES" \
        --runs "$RUNS" >"$RAW_DIR/$label/run.log" 2>&1; then
        for mode in "${MODE_LIST[@]}"; do
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "$impl" true "$mode" true false true null null "" "Rust benchmark failed; see $RAW_DIR/$label/run.log" "$RAW_DIR/$label/run.log" "$commit_ref" "$TIMESTAMP" false
        done
        return
    fi

    normalize_rust_results "$label" "$impl" true "$result_dir" "$commit_ref"
}

benchmark_c_target() {
    local label="$1"
    local clone_dir="$2"
    local build_log="$RAW_DIR/$label/build.log"
    local run_dir="$RAW_DIR/$label"
    local binary="$clone_dir/qwen_asr"
    local commit_ref
    commit_ref="$(git -C "$clone_dir" rev-parse --short HEAD)"
    mkdir -p "$run_dir"

    log "Building $label"
    if ! /bin/zsh -lc "cd '$clone_dir' && make blas" >"$build_log" 2>&1; then
        for mode in "${MODE_LIST[@]}"; do
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" true "$mode" false false true null null "" "C build failed; see $build_log" "$build_log" "$commit_ref" "$TIMESTAMP" false
        done
        return
    fi

    for mode in "${MODE_LIST[@]}"; do
        if ! detect_c_mode_support "$binary" "$mode"; then
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" true "$mode" true false false null null "" "Mode unsupported by upstream CLI" "$build_log" "$commit_ref" "$TIMESTAMP" false
            continue
        fi

        local best_ms=""
        local best_stdout=""
        local best_stderr=""
        local best_run_log="$run_dir/$mode-runs.log"
        : >"$best_run_log"

        for run_i in $(seq 1 "$RUNS"); do
            local stdout_file stderr_file rc
            stdout_file="$(mktemp "$run_dir/${mode}.stdout.run${run_i}.XXXX")"
            stderr_file="$(mktemp "$run_dir/${mode}.stderr.run${run_i}.XXXX")"
            rc="$(run_c_once "$binary" "$mode" "$stdout_file" "$stderr_file")"

            # Parse timing from C stderr (same format as Rust)
            local parsed_total_ms=""
            if [[ "$rc" == "0" ]]; then
                parsed_total_ms=$(bash "$SCRIPT_DIR/parse_stderr.sh" < "$stderr_file" | grep '^total_ms=' | head -1 | cut -d= -f2 || true)
            fi
            printf 'run=%s rc=%s total_ms=%s stdout=%s stderr=%s\n' "$run_i" "$rc" "${parsed_total_ms:-failed}" "$stdout_file" "$stderr_file" >>"$best_run_log"

            if [[ "$rc" != "0" ]] || [[ -z "$parsed_total_ms" ]]; then
                continue
            fi
            if [[ -z "$best_ms" ]] || awk "BEGIN{exit !($parsed_total_ms < $best_ms)}"; then
                best_ms="$parsed_total_ms"
                best_stdout="$stdout_file"
                best_stderr="$stderr_file"
            fi
        done

        if [[ -z "$best_ms" ]]; then
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" true "$mode" true false true null null "" "All C runs failed; see $best_run_log" "$best_run_log" "$commit_ref" "$TIMESTAMP" false
            continue
        fi

        # Parse realtime factor from C stderr
        local rtf
        rtf=$(bash "$SCRIPT_DIR/parse_stderr.sh" < "$best_stderr" | grep '^realtime_factor=' | head -1 | cut -d= -f2 || true)
        rtf="${rtf:-0}"
        local transcript
        transcript="$(cat "$best_stdout")"
        write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" true "$mode" true true true "$best_ms" "$rtf" "$transcript" "" "$best_run_log" "$commit_ref" "$TIMESTAMP" false
    done
}

generate_summary() {
    python3 - "$SUMMARY_DIR" "$ROOT_SUMMARY" <<'PY'
import glob, json, os, sys
src_dir, out_file = sys.argv[1:]
items = []
for path in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        items.append(json.load(f))
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
PY
}

generate_report_and_charts() {
    "$VENV_DIR/bin/python" "$SCRIPT_DIR/render_benchmark_report.py" \
        --summary "$ROOT_SUMMARY" \
        --report "$REPORT_DIR/report.md" \
        --root-report "$ROOT_REPORT" \
        --charts-dir "$CHARTS_DIR" \
        --baseline-ref "$BASELINE_SHORT" \
        --current-ref "$CURRENT_SHORT" \
        --model-dir "$MODEL_DIR" \
        --input-file "$INPUT_FILE" \
        --runs "$RUNS" \
        --modes "$MODES"
}

ensure_python_env

log "Preparing worktrees"
ensure_worktree "$TMP_DIR/baseline-rust" "$BASELINE_REF"
ensure_worktree "$TMP_DIR/current-rust" "$CURRENT_REF"

log "Preparing upstream C repo"
ensure_c_clone "$TMP_DIR/antirez-qwen-asr"

benchmark_rust_target "rust-before-accelerate" "rust-before" "$TMP_DIR/baseline-rust" "$BASELINE_SHORT"
benchmark_rust_target "rust-current-accelerate" "rust-current" "$TMP_DIR/current-rust" "$CURRENT_SHORT"
benchmark_c_target "c-antirez-accelerate" "$TMP_DIR/antirez-qwen-asr"

generate_summary
generate_report_and_charts

log "Summary: $ROOT_SUMMARY"
log "Report:  $REPORT_DIR/report.md"
log "Latest report copied to: $ROOT_REPORT"
