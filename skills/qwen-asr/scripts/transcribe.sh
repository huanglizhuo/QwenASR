#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${QWEN_ASR_MODEL_DIR:-${HOME}/.openclaw/tools/qwen-asr/qwen3-asr-0.6b}"

# Convert non-WAV audio to WAV via ffmpeg, return path to use
to_wav() {
  local input="$1"
  case "${input##*.}" in
    wav|WAV) echo "$input" ;;
    *)
      if ! command -v ffmpeg &>/dev/null; then
        echo "Error: ffmpeg is required for non-WAV files. Install it with:" >&2
        echo "  macOS:  brew install ffmpeg" >&2
        echo "  Linux:  sudo apt install ffmpeg" >&2
        exit 1
      fi
      local tmp
      tmp="$(mktemp /tmp/qwen-asr-XXXXXX.wav)"
      ffmpeg -y -i "$input" -ar 16000 -ac 1 -f wav "$tmp" -loglevel error
      echo "$tmp"
      ;;
  esac
}

if [ "${1:-}" = "--stdin" ]; then
  shift
  exec qwen-asr -d "$MODEL_DIR" --stdin --silent "$@"
elif [ -n "${1:-}" ]; then
  INPUT="$(to_wav "$1")"
  trap 'rm -f "$INPUT"' EXIT 2>/dev/null || true
  qwen-asr -d "$MODEL_DIR" -i "$INPUT" --silent "${@:2}"
else
  echo "Usage: transcribe.sh <audio-file> [options...]"
  echo "       transcribe.sh --stdin [options...]"
  echo ""
  echo "Supports: wav, mp3, m4a, ogg, flac, opus, webm, aac, etc. (non-WAV requires ffmpeg)"
  echo "Options are passed through to qwen-asr (e.g., --language zh, -S 30)"
  exit 1
fi
