#!/usr/bin/env bash
# Regenerate the current-project benchmark pages (speed + WER).
# The heavy cross-implementation comparison is run separately via:
#   ./bench/benchmark-all.sh --runs 10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

BINARY="$PROJECT_DIR/target/release/qwen-asr"
MODEL_DIR="$PROJECT_DIR/qwen3-asr-0.6b"
DATASET="$PROJECT_DIR/librispeech-wer-bench/dev-clean-2"

cd "$PROJECT_DIR"

echo "==> Building release binary"
RUSTFLAGS="-C target-cpu=native" cargo build --release

echo "==> Running speed benchmark"
"$PROJECT_DIR/bench/run.sh" --label current --runs 10

echo "==> Running WER benchmark (100-file offline)"
python3 "$PROJECT_DIR/librispeech-wer-bench/librispeech_wer.py" \
  --dataset "$DATASET" \
  --binary "$BINARY" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$PROJECT_DIR/librispeech-wer-bench/results-100" \
  --label current-offline-100 \
  --limit 100 --mode offline

echo "==> Done. Update the following files with the new numbers:"
echo "    docs/benchmarks/results.md"
echo "    README.md"
echo ""
echo "To refresh the cross-implementation comparison, run:"
echo "    ./bench/benchmark-all.sh --runs 10"
