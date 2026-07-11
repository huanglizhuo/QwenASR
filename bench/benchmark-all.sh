#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="$PROJECT_DIR/qwen3-asr-0.6b"
INPUT_FILE="$SCRIPT_DIR/samples/audio.wav"
RUNS=10
MODES="offline"
THREADS=""
TMP_DIR="$PROJECT_DIR/tmp/benchmark-all"
REPORT_DIR=""
CHARTS_DIR="$SCRIPT_DIR/charts"
VENV_DIR="$SCRIPT_DIR/.venv-bench"
BASELINE_REF="bf52daf"
CURRENT_REF="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
C_REPO_URL="https://github.com/antirez/qwen-asr.git"
UPSTREAM_REF="main"
SECOND_STATE_REPO="https://github.com/second-state/qwen3_asr_rs.git"
SECOND_STATE_REF="v0.2.0"
MLX_AUDIO_VENV="$SCRIPT_DIR/.venv-mlx-audio"
MLX_AUDIO_MODEL="mlx-community/Qwen3-ASR-0.6B-8bit"
DO_CLEAN=0

usage() {
    cat >&2 <<EOF
Usage: $0 [options]

  --model-dir DIR             Model directory (default: ../qwen3-asr-0.6b)
  --input FILE                Input WAV file (default: ./samples/audio.wav)
  --runs N                    Number of standalone runs per target/mode (default: 10; report uses median)
  --modes LIST                Comma-separated modes (default: offline)
  --threads N                 Force thread count for every CPU implementation (default: none — each implementation uses its own default, "out-of-the-box" methodology)
  --tmp-dir DIR               Temp/worktree directory (default: ../tmp/benchmark-all)
  --report-dir DIR            Output report directory (default: ./compare-results/<timestamp>)
  --charts-dir DIR            Stable chart output directory (default: ./charts)
  --venv-dir DIR              Python venv for benchmark tooling (default: ./.venv-bench)
  --baseline-ref REF          First Rust ref (default: bf52daf)
  --current-ref REF           Current Rust ref (default: HEAD)
  --upstream-ref REF          Upstream C ref to clone/reset (default: main)
  --second-state-ref REF      second-state/qwen3_asr_rs ref (default: main)
  --mlx-audio-venv DIR        Python venv for mlx-audio (default: ./.venv-mlx-audio)
  --mlx-audio-model NAME      mlx-audio model (default: mlx-community/Qwen3-ASR-0.6B-8bit)
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
        --threads) THREADS="$2"; shift 2 ;;
        --tmp-dir) TMP_DIR="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --charts-dir) CHARTS_DIR="$2"; shift 2 ;;
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        --baseline-ref) BASELINE_REF="$2"; shift 2 ;;
        --current-ref) CURRENT_REF="$2"; shift 2 ;;
        --upstream-ref) UPSTREAM_REF="$2"; shift 2 ;;
        --second-state-ref) SECOND_STATE_REF="$2"; shift 2 ;;
        --mlx-audio-venv) MLX_AUDIO_VENV="$2"; shift 2 ;;
        --mlx-audio-model) MLX_AUDIO_MODEL="$2"; shift 2 ;;
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
    rm -rf "$TMP_DIR/second-state-qwen3-asr-rs"
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
MLX_AUDIO_VENV="$(abs_path "$MLX_AUDIO_VENV")"

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
# THREADS empty = out-of-the-box methodology: no thread flag is passed and
# each implementation picks its own default (the historical behavior forced
# min(system CPUs, 16) onto every CPU implementation).
BASELINE_SHORT="$(git -C "$PROJECT_DIR" rev-parse --short "$BASELINE_REF")"
CURRENT_SHORT="$(git -C "$PROJECT_DIR" rev-parse --short "$CURRENT_REF")"

ensure_worktree() {
    local path="$1"
    local ref="$2"
    remove_registered_worktree "$path"
    mkdir -p "$(dirname "$path")"
    git -C "$PROJECT_DIR" worktree add --force --detach "$path" "$ref" >/dev/null
}

reset_to_ref() {
    local path="$1" ref="$2"
    # Tags are not available as origin/<tag>; fetch the tag explicitly then reset to the local ref.
    if git -C "$path" rev-parse -q --verify "refs/tags/$ref" >/dev/null 2>&1; then
        git -C "$path" fetch --depth 1 origin "refs/tags/$ref:refs/tags/$ref" >/dev/null
        git -C "$path" reset --hard "$ref" >/dev/null
    else
        git -C "$path" fetch --depth 1 origin "$ref" >/dev/null
        git -C "$path" reset --hard "origin/$ref" >/dev/null
    fi
}

ensure_c_clone() {
    local path="$1"
    if [[ -d "$path/.git" ]]; then
        log "Updating C repo clone"
        reset_to_ref "$path" "$UPSTREAM_REF"
        git -C "$path" clean -fd >/dev/null
    else
        rm -rf "$path"
        log "Cloning C repo into $path"
        git clone --depth 1 --branch "$UPSTREAM_REF" "$C_REPO_URL" "$path" >/dev/null
    fi
}

ensure_second_state_clone() {
    local path="$1"
    if [[ -d "$path/.git" ]]; then
        log "Updating second-state repo clone"
        reset_to_ref "$path" "$SECOND_STATE_REF"
        git -C "$path" clean -fd >/dev/null
    else
        rm -rf "$path"
        log "Cloning second-state/qwen3_asr_rs into $path"
        git clone --depth 1 --branch "$SECOND_STATE_REF" "$SECOND_STATE_REPO" "$path" >/dev/null
    fi
    if [[ ! -f "$path/mlx-c/CMakeLists.txt" ]]; then
        log "Initializing second-state mlx-c submodule"
        git -C "$path" submodule update --init --recursive >/dev/null
    fi
}

ensure_tokenizer_json() {
    if [[ -f "$MODEL_DIR/tokenizer.json" ]]; then
        return 0
    fi
    log "Generating tokenizer.json for second-state"
    python3 - "$MODEL_DIR" <<'PY' 2>/dev/null || true
import os, sys
model_dir = sys.argv[1]
json_path = os.path.join(model_dir, "tokenizer.json")
if os.path.exists(json_path):
    raise SystemExit(0)
try:
    from transformers import Qwen2TokenizerFast
    tok = Qwen2TokenizerFast.from_pretrained(model_dir, trust_remote_code=True)
    tok.backend_tokenizer.save(json_path)
except Exception as e:
    print(f"Warning: could not generate tokenizer.json: {e}", file=sys.stderr)
    raise SystemExit(1)
PY
}

patch_second_state_bench_timer() {
    local main_rs="$1/src/main.rs"
    python3 - "$main_rs" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """    // Run transcription
    tracing::info!(\"Transcribing: {}\", audio_file);
    let result = model
        .transcribe(audio_file, language)
        .context(\"Transcription failed\")?;

    // Output result
    println!(\"Language: {}\", result.language);
    println!(\"Text: {}\", result.text);
"""
new = """    // Run transcription
    tracing::info!(\"Transcribing: {}\", audio_file);
    #[cfg(feature = \"mlx\")]
    qwen3_asr::backend::mlx::stream::synchronize();
    let inference_t0 = std::time::Instant::now();
    let result = model
        .transcribe(audio_file, language)
        .context(\"Transcription failed\")?;
    #[cfg(feature = \"mlx\")]
    qwen3_asr::backend::mlx::stream::synchronize();
    let inference_ms = inference_t0.elapsed().as_secs_f64() * 1000.0;

    // Output result
    println!(\"Language: {}\", result.language);
    println!(\"Text: {}\", result.text);
    println!(\"BENCH_INFERENCE_MS={:.1}\", inference_ms);
"""
if old not in text:
    raise SystemExit(f"Could not patch second-state benchmark timer in {path}")
path.write_text(text.replace(old, new), encoding="utf-8")
PY
}

ensure_mlx_audio() {
    if [[ ! -x "$MLX_AUDIO_VENV/bin/python" ]]; then
        log "Creating mlx-audio venv at $MLX_AUDIO_VENV"
        python3 -m venv "$MLX_AUDIO_VENV"
    fi
    log "Installing/upgrading mlx-audio"
    "$MLX_AUDIO_VENV/bin/pip" install -U mlx-audio >/dev/null
    MLX_AUDIO_VERSION="$($MLX_AUDIO_VENV/bin/pip show mlx-audio 2>/dev/null | awk '/^Version:/ {print $2}' || echo 'unknown')"
    log "mlx-audio version: $MLX_AUDIO_VERSION"
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
    local wall_clock_ms="${16:-}"
    local wall_clock_rtf="${17:-}"
    local inference_mean_ms="${18:-}"
    local inference_best_ms="${19:-}"
    local wall_clock_mean_ms="${20:-}"
    local wall_clock_best_ms="${21:-}"

    python3 - "$outfile" "$impl" "$accel" "$mode" "$build_ok" "$run_ok" "$supports_mode" "$total_ms" "$realtime_factor" "$transcript" "$note" "$source_artifact" "$commit_ref" "$benchmark_date" "$historical" "$wall_clock_ms" "$wall_clock_rtf" "$inference_mean_ms" "$inference_best_ms" "$wall_clock_mean_ms" "$wall_clock_best_ms" <<'PY'
import json, sys
outfile, impl, accel, mode, build_ok, run_ok, supports_mode, total_ms, rtf, transcript, note, source, commit_ref, benchmark_date, historical, wall_clock_ms, wall_clock_rtf, inference_mean_ms, inference_best_ms, wall_clock_mean_ms, wall_clock_best_ms = sys.argv[1:]
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
wall_ms = parse_num(wall_clock_ms)
wall_rtf = parse_num(wall_clock_rtf)
if wall_ms is not None:
    payload["wall_clock_ms"] = wall_ms
if wall_rtf is not None:
    payload["wall_clock_realtime_factor"] = wall_rtf
payload["statistic"] = "median"
for key, value in (
    ("inference_mean_ms", inference_mean_ms),
    ("inference_best_ms", inference_best_ms),
    ("wall_clock_mean_ms", wall_clock_mean_ms),
    ("wall_clock_best_ms", wall_clock_best_ms),
):
    parsed = parse_num(value)
    if parsed is not None:
        payload[key] = parsed
with open(outfile, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

stats_json_from_tsv() {
    local tsv_file="$1"
    python3 - "$tsv_file" <<'PY'
import json, statistics, sys

rows = []
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        run_i, inference_ms, wall_ms, stdout_file = line.rstrip("\n").split("\t")
        rows.append({
            "run": int(run_i),
            "inference_ms": float(inference_ms),
            "wall_ms": float(wall_ms),
            "stdout": stdout_file,
        })
if not rows:
    raise SystemExit(1)
rows.sort(key=lambda row: row["inference_ms"])
inference_median = statistics.median(row["inference_ms"] for row in rows)
wall_median = statistics.median(row["wall_ms"] for row in rows)
median = min(rows, key=lambda row: (abs(row["inference_ms"] - inference_median), row["inference_ms"]))
print(json.dumps({
    "median": median,
    "inference_median_ms": inference_median,
    "inference_mean_ms": statistics.fmean(row["inference_ms"] for row in rows),
    "inference_best_ms": rows[0]["inference_ms"],
    "wall_median_ms": wall_median,
    "wall_mean_ms": statistics.fmean(row["wall_ms"] for row in rows),
    "wall_best_ms": min(row["wall_ms"] for row in rows),
}))
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
    inference_ms = timing.get("total_ms")
    inference_rtf = timing.get("realtime_factor")
    wall_ms = timing.get("wall_ms") or inference_ms
    audio_duration_s = raw.get("audio_duration_s") or 0
    wall_rtf = (float(audio_duration_s) / (float(wall_ms) / 1000.0)) if wall_ms else None
    payload = {
        "impl": impl,
        "accelerate": accel == "true",
        "mode": mode,
        "build_ok": True,
        "run_ok": True,
        "supports_mode": True,
        "total_ms": inference_ms,
        "realtime_factor": inference_rtf,
        "wall_clock_ms": wall_ms,
        "wall_clock_realtime_factor": wall_rtf,
        "statistic": timing.get("statistic", "median"),
        "inference_mean_ms": timing.get("inference_mean_ms"),
        "inference_best_ms": timing.get("inference_best_ms"),
        "wall_clock_mean_ms": timing.get("wall_mean_ms"),
        "wall_clock_best_ms": timing.get("wall_best_ms"),
        "transcript": raw.get("transcript"),
        "note": "Inference time excludes process startup and model load; wall-clock retained as wall_clock_ms",
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
import os, subprocess, sys
binary, model_dir, input_file, mode, stdout_file, stderr_file = sys.argv[1:]
cmd = [binary, "-d", model_dir, "-i", input_file]
if os.environ.get("BENCH_THREADS"):
    cmd += ["-t", os.environ["BENCH_THREADS"]]
if mode == "segmented":
    cmd += ["-S", "30"]
elif mode == "streaming":
    cmd += ["--stream"]
with open(stdout_file, "wb") as so, open(stderr_file, "wb") as se:
    import time
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=so, stderr=se)
    t1 = time.perf_counter()
print(f"{proc.returncode} {(t1 - t0) * 1000:.1f}")
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

    local thread_args=()
    if [[ -n "$THREADS" ]]; then
        thread_args=(--threads "$THREADS")
    fi
    log "Running $label"
    if ! "$SCRIPT_DIR/run.sh" \
        --binary "$worktree/target/release/qwen-asr" \
        --model-dir "$MODEL_DIR" \
        --samples-dir "$INPUT_FILE" \
        --output-dir "$output_root" \
        --label "$label" \
        --modes "$MODES" \
        ${thread_args[@]+"${thread_args[@]}"} \
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

        local runs_tsv
        runs_tsv="$(mktemp "$run_dir/${mode}.runs.XXXX")"
        local best_run_log="$run_dir/$mode-runs.log"
        : >"$best_run_log"

        for run_i in $(seq 1 "$RUNS"); do
            local stdout_file stderr_file line rc wall_ms
            stdout_file="$(mktemp "$run_dir/${mode}.stdout.run${run_i}.XXXX")"
            stderr_file="$(mktemp "$run_dir/${mode}.stderr.run${run_i}.XXXX")"
            line="$(run_c_once "$binary" "$mode" "$stdout_file" "$stderr_file")"
            rc="$(echo "$line" | awk '{print $1}')"
            wall_ms="$(echo "$line" | awk '{print $2}')"

            # Parse timing from C stderr (same format as Rust)
            local parsed_total_ms=""
            if [[ "$rc" == "0" ]]; then
                parsed_total_ms=$(bash "$SCRIPT_DIR/parse_stderr.sh" < "$stderr_file" | grep '^total_ms=' | head -1 | cut -d= -f2 || true)
            fi
            printf 'run=%s rc=%s wall_ms=%s inference_ms=%s stdout=%s stderr=%s\n' "$run_i" "$rc" "${wall_ms:-failed}" "${parsed_total_ms:-failed}" "$stdout_file" "$stderr_file" >>"$best_run_log"

            if [[ "$rc" != "0" ]] || [[ -z "$parsed_total_ms" ]]; then
                continue
            fi
            printf '%s\t%s\t%s\t%s\n' "$run_i" "$parsed_total_ms" "$wall_ms" "$stdout_file" >>"$runs_tsv"
        done

        if [[ ! -s "$runs_tsv" ]]; then
            write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" true "$mode" true false true null null "" "All C runs failed; see $best_run_log" "$best_run_log" "$commit_ref" "$TIMESTAMP" false
            rm -f "$runs_tsv"
            continue
        fi

        local stats_json best_ms best_wall_ms best_stdout inference_mean_ms inference_best_ms wall_mean_ms wall_best_ms rtf wall_rtf
        stats_json="$(stats_json_from_tsv "$runs_tsv")"
        best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_median_ms"])' "$stats_json")"
        best_wall_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_median_ms"])' "$stats_json")"
        best_stdout="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["median"]["stdout"])' "$stats_json")"
        inference_mean_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_mean_ms"])' "$stats_json")"
        inference_best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_best_ms"])' "$stats_json")"
        wall_mean_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_mean_ms"])' "$stats_json")"
        wall_best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_best_ms"])' "$stats_json")"
        rtf="$(calc_rtf_from_ms "$best_ms")"
        wall_rtf="$(calc_rtf_from_ms "$best_wall_ms")"
        local transcript
        transcript="$(cat "$best_stdout")"
        write_result_json "$SUMMARY_DIR/$label-$mode.json" "c-antirez" true "$mode" true true true "$best_ms" "$rtf" "$transcript" "Median of standalone runs; inference time excludes process startup and model load; wall-clock retained as wall_clock_ms" "$best_run_log" "$commit_ref" "$TIMESTAMP" false "$best_wall_ms" "$wall_rtf" "$inference_mean_ms" "$inference_best_ms" "$wall_mean_ms" "$wall_best_ms"
        rm -f "$runs_tsv"
    done
}

calc_rtf_from_ms() {
    local wall_ms="$1"
    python3 - "$AUDIO_DURATION_S" "$wall_ms" <<'PY'
import sys
duration = float(sys.argv[1])
wall_ms = float(sys.argv[2])
print(f"{duration / (wall_ms / 1000.0):.2f}" if wall_ms > 0 else "0")
PY
}

run_external_once() {
    local label="$1"
    local binary="$2"
    local stdout_file="$3"
    local stderr_file="$4"
    shift 4
    python3 - "$label" "$binary" "$stdout_file" "$stderr_file" "$@" <<'PY'
import subprocess, sys, time
label, binary, stdout_file, stderr_file = sys.argv[1:5]
args = sys.argv[5:]
with open(stdout_file, "wb") as so, open(stderr_file, "wb") as se:
    t0 = time.perf_counter()
    proc = subprocess.run([binary] + args, stdout=so, stderr=se)
    t1 = time.perf_counter()
print(f"{proc.returncode} {(t1 - t0) * 1000:.1f}")
PY
}

benchmark_second_state_mlx() {
    local label="second-state-mlx"
    local clone_dir="$1"
    local build_log="$RAW_DIR/$label/build.log"
    local run_dir="$RAW_DIR/$label"
    local binary="$clone_dir/target/release/asr"
    local commit_ref
    commit_ref="$(git -C "$clone_dir" rev-parse --short HEAD)"
    mkdir -p "$run_dir"

    log "Building $label"
    patch_second_state_bench_timer "$clone_dir"
    if ! /bin/zsh -lc "cd '$clone_dir' && if ! grep -q '^\\[workspace\\]$' Cargo.toml 2>/dev/null; then printf '\\n[workspace]\\n' >> Cargo.toml; fi && cargo build --release --no-default-features --features mlx" >"$build_log" 2>&1; then
        write_result_json "$SUMMARY_DIR/$label-offline.json" "second-state-mlx" false "offline" false false true null null "" "second-state MLX build failed; see $build_log" "$build_log" "$commit_ref" "$TIMESTAMP" false
        return
    fi

    log "Running $label"
    local runs_tsv
    runs_tsv="$(mktemp "$run_dir/offline.runs.XXXX")"
    local best_run_log="$run_dir/offline-runs.log"
    : >"$best_run_log"
    for run_i in $(seq 1 "$RUNS"); do
        local stdout_file stderr_file line rc wall_ms inference_ms
        stdout_file="$(mktemp "$run_dir/offline.stdout.run${run_i}.XXXX")"
        stderr_file="$(mktemp "$run_dir/offline.stderr.run${run_i}.XXXX")"
        line="$(run_external_once "$label" "$binary" "$stdout_file" "$stderr_file" "$MODEL_DIR" "$INPUT_FILE")"
        rc="$(echo "$line" | awk '{print $1}')"
        wall_ms="$(echo "$line" | awk '{print $2}')"
        inference_ms="$(grep '^BENCH_INFERENCE_MS=' "$stdout_file" | head -1 | cut -d= -f2 || true)"
        printf 'run=%s rc=%s inference_ms=%s wall_ms=%s stdout=%s stderr=%s\n' "$run_i" "$rc" "${inference_ms:-failed}" "$wall_ms" "$stdout_file" "$stderr_file" >>"$best_run_log"
        if [[ "$rc" != "0" ]]; then
            continue
        fi
        if [[ -z "$inference_ms" ]]; then
            continue
        fi
        printf '%s\t%s\t%s\t%s\n' "$run_i" "$inference_ms" "$wall_ms" "$stdout_file" >>"$runs_tsv"
    done

    if [[ ! -s "$runs_tsv" ]]; then
        write_result_json "$SUMMARY_DIR/$label-offline.json" "second-state-mlx" false "offline" true false true null null "" "All second-state MLX runs failed; see $best_run_log" "$best_run_log" "$commit_ref" "$TIMESTAMP" false
        rm -f "$runs_tsv"
        return
    fi

    local stats_json best_ms best_wall_ms best_stdout inference_mean_ms inference_best_ms wall_mean_ms wall_best_ms transcript rtf wall_rtf
    stats_json="$(stats_json_from_tsv "$runs_tsv")"
    best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_median_ms"])' "$stats_json")"
    best_wall_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_median_ms"])' "$stats_json")"
    best_stdout="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["median"]["stdout"])' "$stats_json")"
    inference_mean_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_mean_ms"])' "$stats_json")"
    inference_best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_best_ms"])' "$stats_json")"
    wall_mean_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_mean_ms"])' "$stats_json")"
    wall_best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_best_ms"])' "$stats_json")"
    transcript="$(grep '^Text: ' "$best_stdout" 2>/dev/null | sed 's/^Text: //' || true)"
    rtf="$(calc_rtf_from_ms "$best_ms")"
    wall_rtf="$(calc_rtf_from_ms "$best_wall_ms")"
    write_result_json "$SUMMARY_DIR/$label-offline.json" "second-state-mlx" false "offline" true true true "$best_ms" "$rtf" "$transcript" "Median of standalone runs; inference time excludes process startup and model load; wall-clock retained as wall_clock_ms" "$best_run_log" "$commit_ref" "$TIMESTAMP" false "$best_wall_ms" "$wall_rtf" "$inference_mean_ms" "$inference_best_ms" "$wall_mean_ms" "$wall_best_ms"
    rm -f "$runs_tsv"
}

benchmark_mlx_audio() {
    local label="mlx-audio"
    local run_dir="$RAW_DIR/$label"
    mkdir -p "$run_dir"

    log "Running $label"
    local runs_tsv
    runs_tsv="$(mktemp "$run_dir/offline.runs.XXXX")"
    local best_run_log="$run_dir/offline-runs.log"
    : >"$best_run_log"
    for run_i in $(seq 1 "$RUNS"); do
        local out_path="$run_dir/run${run_i}/output"
        local stdout_file="$run_dir/offline.stdout.run${run_i}"
        local stderr_file="$run_dir/offline.stderr.run${run_i}"
        if python3 "$SCRIPT_DIR/run_mlx_audio.py" \
            --venv-python "$MLX_AUDIO_VENV/bin/python" \
            --model "$MLX_AUDIO_MODEL" \
            --audio "$INPUT_FILE" \
            --output-path "$out_path" >"$stdout_file" 2>"$stderr_file"; then
            local wall_ms inference_ms transcript
            wall_ms="$(grep '^wall_ms=' "$stdout_file" | head -1 | cut -d= -f2 || true)"
            inference_ms="$(grep '^inference_ms=' "$stdout_file" | head -1 | cut -d= -f2 || true)"
            transcript="$(grep '^transcript=' "$stdout_file" | sed 's/^transcript=//' || true)"
            printf 'run=%s rc=0 inference_ms=%s wall_ms=%s stdout=%s stderr=%s\n' "$run_i" "$inference_ms" "$wall_ms" "$stdout_file" "$stderr_file" >>"$best_run_log"
            if [[ -n "$inference_ms" ]]; then
                printf '%s\t%s\t%s\t%s\n' "$run_i" "$inference_ms" "$wall_ms" "$stdout_file" >>"$runs_tsv"
            fi
        else
            printf 'run=%s rc=1 wall_ms=failed stdout=%s stderr=%s\n' "$run_i" "$stdout_file" "$stderr_file" >>"$best_run_log"
        fi
    done

    if [[ ! -s "$runs_tsv" ]]; then
        write_result_json "$SUMMARY_DIR/$label-offline.json" "mlx-audio" false "offline" true false true null null "" "All mlx-audio runs failed; see $best_run_log" "$best_run_log" "$MLX_AUDIO_VERSION" "$TIMESTAMP" false
        rm -f "$runs_tsv"
        return
    fi

    local stats_json best_ms best_wall_ms best_stdout best_transcript inference_mean_ms inference_best_ms wall_mean_ms wall_best_ms rtf wall_rtf
    stats_json="$(stats_json_from_tsv "$runs_tsv")"
    best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_median_ms"])' "$stats_json")"
    best_wall_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_median_ms"])' "$stats_json")"
    best_stdout="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["median"]["stdout"])' "$stats_json")"
    best_transcript="$(grep '^transcript=' "$best_stdout" | sed 's/^transcript=//' || true)"
    inference_mean_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_mean_ms"])' "$stats_json")"
    inference_best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["inference_best_ms"])' "$stats_json")"
    wall_mean_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_mean_ms"])' "$stats_json")"
    wall_best_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_best_ms"])' "$stats_json")"
    rtf="$(calc_rtf_from_ms "$best_ms")"
    wall_rtf="$(calc_rtf_from_ms "$best_wall_ms")"
    write_result_json "$SUMMARY_DIR/$label-offline.json" "mlx-audio" false "offline" true true true "$best_ms" "$rtf" "$best_transcript" "Median of standalone runs; inference time excludes process startup and model load; wall-clock retained as wall_clock_ms" "$best_run_log" "$MLX_AUDIO_VERSION" "$TIMESTAMP" false "$best_wall_ms" "$wall_rtf" "$inference_mean_ms" "$inference_best_ms" "$wall_mean_ms" "$wall_best_ms"
    rm -f "$runs_tsv"
}

collect_system_info() {
    local outfile="$1"
    python3 - "$outfile" <<'PY'
import json, sys, subprocess

def safe_output(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"

cpu_brand = safe_output(["sysctl", "-n", "machdep.cpu.brand_string"])
physical_cores = safe_output(["sysctl", "-n", "hw.physicalcpu"])
logical_cores = safe_output(["sysctl", "-n", "hw.logicalcpu"])
memsize = safe_output(["sysctl", "-n", "hw.memsize"])
memory_gb = "unknown"
if memsize not in ("unknown", ""):
    try:
        memory_gb = round(int(memsize) / (1024 ** 3), 1)
    except ValueError:
        pass

rustc_ver = safe_output(["rustc", "--version"])
if rustc_ver and rustc_ver != "unknown":
    rustc_ver = rustc_ver.splitlines()[0]

info = {
    "cpu_brand": cpu_brand,
    "physical_cores": physical_cores,
    "logical_cores": logical_cores,
    "memory_gb": memory_gb,
    "arch": safe_output(["uname", "-m"]),
    "macos_version": safe_output(["sw_vers", "-productVersion"]),
    "rustc_version": rustc_ver,
}

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
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
        --modes "$MODES" \
        --system-info "$SYSTEM_INFO_FILE"
}

ensure_python_env
export BENCH_THREADS="$THREADS"
if [[ -n "$THREADS" ]]; then
    log "Forcing $THREADS threads for all CPU implementations"
else
    log "Out-of-the-box thread policy: each implementation uses its own default"
fi

log "Preparing worktrees"
ensure_worktree "$TMP_DIR/baseline-rust" "$BASELINE_REF"
ensure_worktree "$TMP_DIR/current-rust" "$CURRENT_REF"

log "Preparing upstream C repo"
ensure_c_clone "$TMP_DIR/antirez-qwen-asr"

log "Preparing second-state repo"
ensure_second_state_clone "$TMP_DIR/second-state-qwen3-asr-rs"
ensure_tokenizer_json

benchmark_rust_target "rust-before-accelerate" "rust-original" "$TMP_DIR/baseline-rust" "$BASELINE_SHORT"
benchmark_rust_target "rust-current-accelerate" "rust-current" "$TMP_DIR/current-rust" "$CURRENT_SHORT"
benchmark_c_target "c-antirez-accelerate" "$TMP_DIR/antirez-qwen-asr"
benchmark_second_state_mlx "$TMP_DIR/second-state-qwen3-asr-rs"
ensure_mlx_audio
benchmark_mlx_audio

SYSTEM_INFO_FILE="$REPORT_DIR/system_info.json"
collect_system_info "$SYSTEM_INFO_FILE"
log "System info saved to $SYSTEM_INFO_FILE"

generate_summary
generate_report_and_charts

log "Summary: $ROOT_SUMMARY"
log "Report:  $REPORT_DIR/report.md"
log "Latest report copied to: $ROOT_REPORT"
