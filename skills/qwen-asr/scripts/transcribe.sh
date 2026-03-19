#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${QWEN_ASR_MODEL_DIR:-${HOME}/.openclaw/tools/qwen-asr/qwen3-asr-0.6b}"

if [ "${1:-}" = "--stdin" ]; then
  shift
  exec qwen-asr -d "$MODEL_DIR" --stdin --silent "$@"
elif [ -n "${1:-}" ]; then
  exec qwen-asr -d "$MODEL_DIR" -i "$1" --silent "${@:2}"
else
  echo "Usage: transcribe.sh <audio-file> [options...]"
  echo "       transcribe.sh --stdin [options...]"
  echo ""
  echo "Options are passed through to qwen-asr (e.g., --language zh, -S 30)"
  exit 1
fi
