#!/usr/bin/env python3
"""Compare real GPU matrix/normalization output with NumPy (test-only CPU math)."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def read_tensor(package, index, name):
    entry = index["tensors"][name]
    dtype = {"f16": "<f2", "f32": "<f4", "i8": "i1"}[entry["dtype"]]
    value = np.memmap(package / "weights.bin", mode="r", offset=entry["offset"], dtype=dtype, shape=tuple(entry["shape"]))
    data = np.asarray(value, dtype=np.float32)
    if entry["dtype"] == "i8":
        scales = read_tensor(package, index, entry["scale"])
        data = data.reshape(len(scales), -1) * scales[:, None]
    return data


def check(package, binary, kernel_dir, output):
    index = json.loads((package / "index.json").read_text())
    cases = [("thinker.model.layers.0.self_attn.q_proj.weight", 1),
             ("thinker.model.layers.0.mlp.down_proj.weight", 13),
             ("thinker.audio_tower.layers.0.fc1.weight", 33),
             ("thinker.lm_head.weight", 3)]
    if index["precision"] == "hybrid":
        cases.append(("thinker.lm_head.weight", 1))
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, (name, rows) in enumerate(cases):
        prefix = output / f"case-{number}"
        execution = subprocess.run([str(binary.resolve()), str(package.resolve()), str(kernel_dir.resolve()), name,
                                    str(rows), str(prefix.resolve())], check=True, capture_output=True, text=True)
        hybrid = index["precision"] == "hybrid" and rows == 1 and (name.startswith("thinker.model.layers.") or name == "thinker.lm_head.weight")
        weight = read_tensor(package, index, name+".decode" if hybrid else name)
        weight = weight.reshape(weight.shape[0], -1)
        width = weight.shape[1]
        x = (((np.arange(rows*width, dtype=np.int64)*17+13)%37)-18).astype(np.float32).reshape(rows, width)/32
        if hybrid:
            maximum = np.max(np.abs(x), axis=1)
            scale = np.where(maximum > 0, maximum/np.float32(127), np.float32(1))
            v = x*(np.float32(127)/np.maximum(maximum, np.float32(1e-10)))[:, None]
            x = np.copysign(np.floor(np.abs(v)+np.float32(.5)), v).clip(-127, 127)*scale[:, None]
        expected = np.einsum("mk,nk->mn", x.astype(np.float64), weight.astype(np.float64), optimize=False)
        actual = np.fromfile(str(prefix)+".f32", dtype="<f4").reshape(expected.shape)
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        # runtime_check uses the same flat input prefix for its independent
        # 1024-wide RMSNorm/GELU/residual sequence.
        xn = (((np.arange(1024, dtype=np.int64)*17+13)%37)-18).astype(np.float32)/32
        norm = read_tensor(package, index, "thinker.model.layers.0.input_layernorm.weight")
        z = xn/np.sqrt(np.mean(xn*xn)+1e-6)*norm
        expected_norm = xn + .5*z*(1+np.tanh(.7978845608*(z+.044715*z*z*z)))
        actual_norm = np.fromfile(str(prefix)+".norm.f32", dtype="<f4")
        np.testing.assert_allclose(actual_norm, expected_norm, atol=2e-5, rtol=2e-5)
        result = {"weight": name, **json.loads(execution.stdout),
                  "max_abs_error": float(np.max(np.abs(actual-expected))),
                  "norm_max_abs_error": float(np.max(np.abs(actual_norm-expected_norm)))}
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / "summary.json").write_text(json.dumps({"package": str(package), "task": index["task"],
                                                   "precision": index["precision"], "passed": True,
                                                   "cases": results}, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("--binary", type=Path, default=Path("qwen-asr-metal/build/runtime-check"))
    p.add_argument("--kernels", type=Path, default=Path("qwen-asr-metal/native"))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    check(a.package, a.binary, a.kernels, a.output)
