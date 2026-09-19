#!/usr/bin/env python3
"""Developer-only GPU encoder differential check; not the ASR release gate."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from convert_model import digest


def check(args):
    root = Path(__file__).resolve().parents[1]
    package = json.loads((args.package / "index.json").read_text())
    for name, expected in package["source_sha256"].items():
        if digest(args.model / name) != expected:
            raise ValueError(f"CPU/package source mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output.resolve() / "cpu"
    subprocess.run([str(root / "target/release/qwen-asr-encoder-reference"), str(args.model.resolve()),
                    str(args.audio.resolve()), str(prefix)], check=True)
    reference = json.loads(Path(str(prefix)+".json").read_text())
    gpu_path = args.output.resolve() / "gpu.f32"
    run = subprocess.run([str(root / "build/encoder-check"), str(args.package.resolve()), str(root / "native"),
                          str(prefix)+".mel.f32", str(reference["frames"]), str(gpu_path)],
                         check=True, capture_output=True, text=True)
    gpu_metrics = json.loads(run.stdout)
    cpu = np.fromfile(str(prefix)+".encoded.f32", dtype="<f4").astype(np.float64)
    gpu = np.fromfile(gpu_path, dtype="<f4").astype(np.float64)
    assert cpu.size == gpu.size == reference["tokens"] * reference["hidden"]
    assert gpu_metrics["tokens"] == reference["tokens"] and gpu_metrics["hidden"] == reference["hidden"]
    assert np.isfinite(cpu).all() and np.isfinite(gpu).all()
    norm = np.sum(cpu*cpu)
    assert norm > 0 and np.sum(gpu*gpu) > 0
    result = {"scope": "single_audio_encoder_only", "task": package["task"], "precision": package["precision"],
              "finite_and_shape_checks_passed": True, "end_to_end_goal_verified": False,
              "max_abs_error": float(np.max(np.abs(gpu-cpu))),
              "relative_l2": float(np.sqrt(np.sum((gpu-cpu)**2)/norm)),
              "cosine": float(np.sum(gpu*cpu)/np.sqrt(np.sum(gpu*gpu)*norm)),
              "reference": reference, "gpu": gpu_metrics,
              "note": "CPU features supplied to isolate encoder; quality and 30% end-to-end speedup are not certified."}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.output / "gpu.stderr.log").write_text(run.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    check(p.parse_args())
