#!/usr/bin/env python3
"""Timing wrapper for mlx-audio benchmark."""
import argparse
import json
import subprocess
import sys
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Run mlx-audio with timing")
    parser.add_argument("--venv-python", required=True, help="Path to venv python executable")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--output-path", required=True, help="Temp output path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    child_code = r'''
import argparse
import json
import os
import time

import mlx.core as mx

from mlx_audio.stt.utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--audio", required=True)
parser.add_argument("--output-path", required=True)
args = parser.parse_args()

model = load_model(args.model)
mx.synchronize()

t0 = time.perf_counter()
segments = model.generate(args.audio, verbose=False)
mx.synchronize()
t1 = time.perf_counter()

text = getattr(segments, "text", "") or ""
os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
with open(args.output_path + ".txt", "w", encoding="utf-8") as f:
    f.write(text)

print(json.dumps({
    "inference_ms": (t1 - t0) * 1000.0,
    "transcript": text,
}, ensure_ascii=False))
'''

    cmd = [
        args.venv_python,
        "-c",
        child_code,
        "--model", args.model,
        "--audio", args.audio,
        "--output-path", args.output_path,
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()

    if result.returncode != 0:
        print(f"exit_code={result.returncode} wall_ms=0 transcript=", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    wall_ms = (t1 - t0) * 1000
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    inference_ms = payload["inference_ms"]
    transcript = payload.get("transcript", "").strip()

    print(f"wall_ms={wall_ms:.1f}")
    print(f"inference_ms={inference_ms:.1f}")
    print(f"transcript={transcript}")


if __name__ == "__main__":
    main()
