#!/usr/bin/env bash
# Usage: ident.sh BASE_BINARY CANDIDATE_BINARY NEW_OUTPUT_DIR
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$HERE/ab.py" "$1" "$2" \
  --cases wav_offline,wav_seg30,wav_stream,2min_offline,2min_seg30,10min_seg30,10min_offline \
  --runs 1 --out "$3"
