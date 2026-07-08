#!/usr/bin/env python3
"""Score a long-audio hypothesis against its sidecar reference.

Reuses the exact normalization and WER/CER scoring from
``librispeech-wer-bench/librispeech_wer.py`` (imported, not forked) so long-audio
numbers stay comparable with the short-utterance WER gate.

Usage: python3 bench/long/score_long.py <reference.txt> < hypothesis.txt
Output: single JSON object on stdout with wer/cer/edit counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "librispeech-wer-bench"))
from librispeech_wer import score  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: score_long.py <reference.txt> < hypothesis.txt", file=sys.stderr)
        return 1
    reference = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
    hypothesis = sys.stdin.read().strip()
    metrics = score(reference, hypothesis)
    out = {
        "wer": round(float(metrics["wer"]), 6),
        "cer": round(float(metrics["cer"]), 6),
        "ref_words": int(metrics["ref_words"]),
        "word_edits": int(metrics["word_edits"]),
        "ref_chars": int(metrics["ref_chars"]),
        "char_edits": int(metrics["char_edits"]),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
