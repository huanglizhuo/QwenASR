#!/usr/bin/env python3
"""CPU/GPU decoder differential check with a separate KV continuation regression."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from convert_model import digest


def metrics(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    norm = np.sum(a*a)
    assert norm > 0
    return {"max_abs_error": float(np.max(np.abs(a-b))),
            "relative_l2": float(np.sqrt(np.sum((a-b)**2)/norm))}


def check(args):
    root = Path(__file__).resolve().parents[1]
    package = json.loads((args.package / "index.json").read_text())
    for name, expected in package["source_sha256"].items():
        if digest(args.model / name) != expected:
            raise ValueError(f"CPU/package source mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output.resolve() / "cpu"
    run = subprocess.run([str(root / "target/release/qwen-asr-decoder-reference"), str(args.model.resolve()),
                          str(args.audio.resolve()), str(args.text.resolve()), str(prefix)],
                         check=True, capture_output=True, text=True)
    (args.output / "cpu.stderr.log").write_text(run.stderr)
    cpu = json.loads(Path(str(prefix)+".json").read_text())
    result = {"scope": "single_audio_decoder_only_with_cpu_input_embeddings", "task": package["task"],
              "precision": package["precision"], "end_to_end_goal_verified": False,
              "reference": cpu, "runs": {}}
    arrays = {}
    for mode in ("full", "split"):
        out = args.output.resolve() / mode
        run = subprocess.run([str(root / "build/decoder-check"), str(args.package.resolve()), str(root / "native"),
                              str(prefix), str(out), mode], check=True, capture_output=True, text=True)
        (args.output / f"{mode}.stderr.log").write_text(run.stderr)
        gpu = json.loads(run.stdout)
        entry = {"timing": gpu, "argmax_matches_cpu": gpu["argmax"] == cpu["argmax"]}
        assert gpu["seq"] == cpu["seq"]
        for kind, count in (("hidden", cpu["seq"]*1024), ("logits", len(cpu["rows"])*cpu["output_dim"])):
            a = np.fromfile(str(prefix)+f".{kind}.f32", dtype="<f4").astype(np.float64)
            b = np.fromfile(str(out)+f".{kind}.f32", dtype="<f4").astype(np.float64)
            assert a.size == b.size == count
            entry[kind] = metrics(a, b)
            arrays[mode, kind] = b
        result["runs"][mode] = entry
    result["kv_continuation"] = {k: metrics(arrays["full", k], arrays["split", k]) for k in ("hidden", "logits")}
    # This regression catches stale KV/position/layout errors. It is deliberately
    # independent of the broader model-quality gate, which requires 20 audios.
    assert all(v["relative_l2"] < 0.005 for v in result["kv_continuation"].values()), result
    if package["precision"] == "fp16":
        assert all(result["runs"][m][k]["relative_l2"] < 0.01 for m in ("full", "split") for k in ("hidden", "logits")), result
    (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "reference"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "package", "audio", "text", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    check(p.parse_args())
