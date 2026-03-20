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
C_REPO_URL="https://github.com/antirez/qwen-asr.git"
DO_CLEAN=0

usage() {
    cat >&2 <<EOF
Usage: $0 [options]

  --model-dir DIR    Model directory (default: ../qwen3-asr-0.6b)
  --input FILE       Input WAV file (default: ./samples/audio.wav)
  --runs N           Number of runs per target/mode (default: 3)
  --modes LIST       Comma-separated modes: offline,segmented,streaming (default: offline)
  --tmp-dir DIR      Temp/worktree directory (default: ../tmp/benchmark-all)
  --report-dir DIR   Output report directory (default: ./compare-results/<timestamp>)
  --clean            Remove tmp/worktree directory and exit
  -h, --help         Show this help
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

json_escape() {
    python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
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
    remove_registered_worktree "$TMP_DIR/main-rust"
    remove_registered_worktree "$TMP_DIR/current-rust"
    rm -rf "$TMP_DIR/antirez-qwen-asr"
    mkdir -p "$(dirname "$TMP_DIR")"
}

MODEL_DIR="$(abs_path "$MODEL_DIR")"
INPUT_FILE="$(abs_path "$INPUT_FILE")"
TMP_DIR="$(abs_path "$TMP_DIR")"

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
mkdir -p "$SUMMARY_DIR" "$RAW_DIR" "$TMP_DIR"

IFS=',' read -r -a MODE_LIST <<< "$MODES"
CURRENT_BRANCH="$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD)"
CURRENT_REF="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
MAIN_REF="main"
AUDIO_DURATION_S="$(wav_duration_seconds "$INPUT_FILE")"

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
        git -C "$path" fetch --depth 1 origin main >/dev/null
        git -C "$path" reset --hard origin/main >/dev/null
        git -C "$path" clean -fd >/dev/null
    else
        rm -rf "$path"
        log "Cloning C repo into $path"
        git clone --depth 1 "$C_REPO_URL" "$path" >/dev/null
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

    python3 - "$outfile" "$impl" "$accel" "$mode" "$build_ok" "$run_ok" "$supports_mode" "$total_ms" "$realtime_factor" "$transcript" "$note" "$source_artifact" <<'PY'
import json, sys
outfile, impl, accel, mode, build_ok, run_ok, supports_mode, total_ms, rtf, transcript, note, source = sys.argv[1:]
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
    python3 - "$label" "$impl" "$accel" "$result_dir" "$SUMMARY_DIR" <<'PY'
import glob, json, os, sys
label, impl, accel, result_dir, out_dir = sys.argv[1:]
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
import subprocess, sys, time
binary, model_dir, input_file, mode, stdout_file, stderr_file = sys.argv[1:]
cmd = [binary, "-d", model_dir, "-i", input_file]
if mode == "segmented":
    cmd += ["-S", "30"]
elif mode == "streaming":
    cmd += ["--stream"]
start = time.perf_counter()
with open(stdout_file, "wb") as so, open(stderr_file, "wb") as se:
    proc = subprocess.run(cmd, stdout=so, stderr=se)
elapsed_ms = (time.perf_counter() - start) * 1000.0
print(f"{proc.returncode} {elapsed_ms:.3f}")
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
    local accel="$3"
    local worktree="$4"
    local build_log="$RAW_DIR/$label/build.log"
    local output_root="$RAW_DIR/$label"
    local result_dir="$output_root/$label"
    mkdir -p "$output_root"

    log "Building $label"
    if [[ "$accel" == "true" ]]; then
        if ! /bin/zsh -lc "cd '$worktree' && cargo clean >/dev/null && RUSTFLAGS='-C target-cpu=native' cargo build --release" >"$build_log" 2>&1; then
            for mode in "${MODE_LIST[@]}"; do
                write_result_json "$SUMMARY_DIR/$label-$mode.json" "$impl" "$accel" "$mode" false false true null null "" "Rust build failed; see $build_log" "$build_log"
            done
            return
        fi
    else
        if ! /bin/zsh -lc "cd '$worktree' && cargo clean >/dev/null && RUSTFLAGS='-C target-cpu=native' cargo build --release --no-default-features" >"$build_log" 2>&1; then
            for mode in "${MODE_LIST[@]}"; do
                write_result_json "$SUMMARY_DIR/$label-$mode.json" "$impl" "$accel" "$mode" false false true null null "" "Rust build failed; see $build_log" "$build_log"
            done
            return
        fi
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
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "$impl" "$accel" "$mode" true false true null null "" "Rust benchmark failed; see $RAW_DIR/$label/run.log" "$RAW_DIR/$label/run.log"
        done
        return
    fi

    normalize_rust_results "$label" "$impl" "$accel" "$result_dir"
}

benchmark_c_target() {
    local label="$1"
    local accel="$2"
    local clone_dir="$3"
    local build_log="$RAW_DIR/$label/build.log"
    local run_dir="$RAW_DIR/$label"
    local binary="$clone_dir/qwen_asr"
    mkdir -p "$run_dir"

    log "Building $label"
    if [[ "$accel" == "true" ]]; then
        if ! /bin/zsh -lc "cd '$clone_dir' && make blas" >"$build_log" 2>&1; then
            for mode in "${MODE_LIST[@]}"; do
                write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" "$accel" "$mode" false false true null null "" "C build failed; see $build_log" "$build_log"
            done
            return
        fi
    else
        if ! /bin/zsh -lc "cd '$clone_dir' && make clean && make qwen_asr CFLAGS='-Wall -Wextra -O3 -march=native -ffast-math' LDFLAGS='-lm -lpthread'" >"$build_log" 2>&1; then
            for mode in "${MODE_LIST[@]}"; do
                write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" "$accel" "$mode" false false true null null "" "C build failed; see $build_log" "$build_log"
            done
            return
        fi
    fi

    for mode in "${MODE_LIST[@]}"; do
        if ! detect_c_mode_support "$binary" "$mode"; then
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" "$accel" "$mode" true false false null null "" "Mode unsupported by upstream CLI" "$build_log"
            continue
        fi

        local best_ms=""
        local best_stdout=""
        local best_stderr=""
        local best_run_log="$run_dir/$mode-runs.log"
        : >"$best_run_log"

        for run_i in $(seq 1 "$RUNS"); do
            local stdout_file stderr_file result rc elapsed_ms
            stdout_file="$(mktemp "$run_dir/${mode}.stdout.run${run_i}.XXXX")"
            stderr_file="$(mktemp "$run_dir/${mode}.stderr.run${run_i}.XXXX")"
            result="$(run_c_once "$binary" "$mode" "$stdout_file" "$stderr_file")"
            rc="${result%% *}"
            elapsed_ms="${result##* }"
            printf 'run=%s rc=%s total_ms=%s stdout=%s stderr=%s\n' "$run_i" "$rc" "$elapsed_ms" "$stdout_file" "$stderr_file" >>"$best_run_log"

            if [[ "$rc" != "0" ]]; then
                continue
            fi
            if [[ -z "$best_ms" ]] || awk "BEGIN{exit !($elapsed_ms < $best_ms)}"; then
                best_ms="$elapsed_ms"
                best_stdout="$stdout_file"
                best_stderr="$stderr_file"
            fi
        done

        if [[ -z "$best_ms" ]]; then
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" "$accel" "$mode" true false true null null "" "All C runs failed; see $best_run_log" "$best_run_log"
            continue
        fi

        local rtf
        rtf="$(python3 - "$AUDIO_DURATION_S" "$best_ms" <<'PY'
import sys
audio_s = float(sys.argv[1])
total_ms = float(sys.argv[2])
print(round(audio_s / (total_ms / 1000.0), 2) if total_ms > 0 else 0.0)
PY
)"
        local transcript
        transcript="$(cat "$best_stdout")"
        write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" "$accel" "$mode" true true true "$best_ms" "$rtf" "$transcript" "" "$best_run_log"
    done
}

generate_summary() {
    python3 - "$SUMMARY_DIR" "$REPORT_DIR/summary.json" <<'PY'
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

generate_report() {
    python3 - "$REPORT_DIR/summary.json" "$REPORT_DIR/report.md" "$ROOT_REPORT" "$CURRENT_BRANCH" "$CURRENT_REF" "$MODEL_DIR" "$INPUT_FILE" "$RUNS" "$MODES" <<'PY'
import json, os, platform, subprocess, sys
summary_path, report_path, root_report, current_branch, current_ref, model_dir, input_file, runs, modes = sys.argv[1:]

with open(summary_path, "r", encoding="utf-8") as f:
    items = json.load(f)

order = [
    ("rust-main", True, "main+accel"),
    ("rust-main", False, "main+noaccel"),
    ("rust-current", True, "current+accel"),
    ("rust-current", False, "current+noaccel"),
    ("c-antirez", True, "c+accel"),
    ("c-antirez", False, "c+noaccel"),
]

def get(impl, accel, mode):
    for item in items:
        if item["impl"] == impl and item["accelerate"] == accel and item["mode"] == mode:
            return item
    return None

def fmt(v):
    return "n/a" if v is None else str(v)

def chart(mode, key, title):
    labels = []
    values = []
    for impl, accel, label in order:
        item = get(impl, accel, mode)
        if item and item.get("supports_mode") and item.get("run_ok") and item.get(key) is not None:
            labels.append(label)
            values.append(item[key])
    if not labels:
        return f"_No data for {mode} {key}_\n"
    ymax = max(values)
    ymax = int(ymax * 1.15) + 1
    labels_str = ", ".join(f'"{x}"' for x in labels)
    values_str = ", ".join(str(round(v, 2)) for v in values)
    return (
        "```mermaid\n"
        "xychart-beta\n"
        f'    title "{title}"\n'
        f"    x-axis [{labels_str}]\n"
        f'    y-axis "{key}" 0 --> {ymax}\n'
        f"    bar [{values_str}]\n"
        "```\n"
    )

def env(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"

lines = []
lines.append("# Benchmark Report")
lines.append("")
lines.append("## Methodology")
lines.append("")
lines.append(f"- Compared three implementations on the same input WAV and model: local Rust `main`, local Rust current branch `{current_branch}`, and `antirez/qwen-asr`.")
lines.append("- Each implementation was built twice on macOS: Accelerate enabled and disabled.")
lines.append(f"- Runs per target: `{runs}`.")
lines.append(f"- Modes requested: `{modes}`.")
lines.append("")
lines.append("## Environment")
lines.append("")
lines.append(f"- Machine arch: `{env(['uname', '-m'])}`")
lines.append(f"- macOS: `{env(['sw_vers', '-productVersion'])}`")
lines.append(f"- Current branch ref: `{current_ref}`")
lines.append(f"- Model dir: `{model_dir}`")
lines.append(f"- Input file: `{input_file}`")
lines.append("")
lines.append("## Results")
lines.append("")
lines.append("| Target | Mode | Build | Run | Total ms | RTF | Note |")
lines.append("|---|---:|---:|---:|---:|---:|---|")
for impl, accel, label in order:
    for mode in [m.strip() for m in modes.split(",") if m.strip()]:
        item = get(impl, accel, mode)
        if not item:
            continue
        note = item.get("note") or ""
        lines.append(
            f"| `{label}` | `{mode}` | `{'ok' if item['build_ok'] else 'fail'}` | "
            f"`{'ok' if item['run_ok'] else ('unsupported' if not item['supports_mode'] else 'fail')}` | "
            f"`{fmt(item.get('total_ms'))}` | `{fmt(item.get('realtime_factor'))}` | {note} |"
        )
lines.append("")
lines.append("## Offline Total Latency")
lines.append("")
lines.append(chart("offline", "total_ms", "Offline Total Latency (ms)"))
lines.append("## Offline Realtime Factor")
lines.append("")
lines.append(chart("offline", "realtime_factor", "Offline Realtime Factor"))

stream_supported = any((item.get("mode") == "streaming" and item.get("run_ok")) for item in items)
if stream_supported:
    lines.append("## Streaming Total Latency")
    lines.append("")
    lines.append(chart("streaming", "total_ms", "Streaming Total Latency (ms)"))
    lines.append("## Streaming Realtime Factor")
    lines.append("")
    lines.append(chart("streaming", "realtime_factor", "Streaming Realtime Factor"))

lines.append("## Findings")
lines.append("")
lines.append("- Compare `main+accel` vs `current+accel` to measure the net effect of the autoresearch branch.")
lines.append("- Compare each `+accel` and `+noaccel` pair to isolate the value of macOS Accelerate for that implementation.")
lines.append("- Compare `current+accel` vs `c+accel` for the strongest practical head-to-head result on macOS.")

content = "\n".join(lines) + "\n"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(content)
with open(root_report, "w", encoding="utf-8") as f:
    f.write(content)
PY
}

log "Preparing worktrees"
ensure_worktree "$TMP_DIR/main-rust" "$MAIN_REF"
ensure_worktree "$TMP_DIR/current-rust" "$CURRENT_REF"

log "Preparing upstream C repo"
ensure_c_clone "$TMP_DIR/antirez-qwen-asr"

benchmark_rust_target "rust-main-accelerate" "rust-main" "true" "$TMP_DIR/main-rust"
benchmark_rust_target "rust-main-no-accelerate" "rust-main" "false" "$TMP_DIR/main-rust"
benchmark_rust_target "rust-current-accelerate" "rust-current" "true" "$TMP_DIR/current-rust"
benchmark_rust_target "rust-current-no-accelerate" "rust-current" "false" "$TMP_DIR/current-rust"
benchmark_c_target "c-antirez-accelerate" "true" "$TMP_DIR/antirez-qwen-asr"
benchmark_c_target "c-antirez-no-accelerate" "false" "$TMP_DIR/antirez-qwen-asr"

generate_summary
generate_report

log "Summary: $REPORT_DIR/summary.json"
log "Report:  $REPORT_DIR/report.md"
log "Latest report copied to: $ROOT_REPORT"
