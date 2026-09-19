#!/usr/bin/env python3
"""Offline, versioned Qwen3 0.6B packages. NumPy is conversion-only.

FP16 is a development reference; INT8 is the initial production candidate.
Neither conversion mode is certified accurate until end-to-end evaluation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

FORMAT = "qwen-asr-metal-v1"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantize_rows(values):
    """Symmetric per-output-row INT8, ties-to-even, FP32 scales."""
    rows = values.reshape(values.shape[0], -1)
    scales = np.max(np.abs(rows), axis=1).astype(np.float32) / np.float32(127)
    scales[scales == 0] = 1
    quantized = np.clip(np.rint(rows / scales[:, None]), -127, 127).astype(np.int8)
    return quantized.reshape(values.shape), scales


def quantize_w4_rows(values, group=64):
    """Grouped symmetric INT4 (range [-7,7]): one FP32 scale per `group`
    consecutive input elements per output row; ties-to-even rounding; two
    values per byte (even element low nibble, offset binary +8)."""
    rows = values.reshape(-1, values.shape[-1])
    if rows.shape[1] % (2 * group):
        raise ValueError("grouped INT4 requires width divisible by 2*group")
    blocks = rows.reshape(rows.shape[0], -1, group)
    scales = (np.max(np.abs(blocks), axis=2).astype(np.float32) / np.float32(7)).reshape(rows.shape[0], -1)
    scales[scales == 0] = 1
    quantized = np.clip(np.rint(blocks / scales[:, :, None]), -7, 7).astype(np.int8) + 8
    flat = quantized.reshape(rows.shape[0], rows.shape[1])
    packed = flat[:, 0::2].astype(np.uint8) | (flat[:, 1::2].astype(np.uint8) << 4)
    return packed, scales


def quantize_cpu_decode_rows(values):
    """Match the baseline ARM NEON weight conversion, which truncates to zero.

    Every supported decoder matrix has an input width divisible by eight; the
    CPU's scalar round-to-nearest tail is therefore never used for these shapes.
    Activations are independently quantized on GPU with nearest-away rounding.
    """
    rows = values.reshape(values.shape[0], -1)
    if rows.shape[1] % 8:
        raise ValueError("CPU-compatible decode weights require width divisible by eight")
    maximum = np.max(np.abs(rows), axis=1).astype(np.float32)
    scales = maximum / np.float32(127)
    scales[maximum == 0] = 1
    inv = np.float32(127) / np.maximum(maximum, np.float32(1e-10))
    quantized = np.clip(np.trunc(rows*inv[:, None]), -128, 127).astype(np.int8)
    return quantized.reshape(values.shape), scales


def inspect_source(model):
    config = json.loads((model / "config.json").read_text())
    thinker = config["thinker_config"]
    text = thinker["text_config"]
    audio = thinker["audio_config"]
    if (text["hidden_size"], text["num_hidden_layers"], text["head_dim"],
        text["num_attention_heads"], text["num_key_value_heads"], text["intermediate_size"]) != (1024, 28, 128, 16, 8, 3072):
        raise ValueError("only the specified 0.6B decoder is supported")
    aligner = thinker.get("classify_num", 0) == 5000
    expected = (1024, 24, 16, 4096) if aligner else (896, 18, 14, 3584)
    if (audio["d_model"], audio["encoder_layers"], audio["encoder_attention_heads"], audio["encoder_ffn_dim"]) != expected:
        raise ValueError("unexpected encoder architecture")
    header = {}
    files = sorted(model.glob("*.safetensors"))
    if not files:
        raise ValueError("no safetensors weights found")
    for file in files:
        with file.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > file.stat().st_size - 8:
                raise ValueError("truncated safetensors header")
            entries = json.loads(f.read(n))
        for name, entry in entries.items():
            if name == "__metadata__":
                continue
            if name in header or entry["dtype"] != "BF16":
                raise ValueError(f"duplicate tensor or unsupported dtype: {name}")
            shape = entry["shape"]
            start, end = entry["data_offsets"]
            if not shape or any(type(d) is not int or d <= 0 for d in shape):
                raise ValueError(f"invalid shape: {name}")
            if start < 0 or end - start != math.prod(shape) * 2 or 8+n+end > file.stat().st_size:
                raise ValueError(f"tensor outside source file: {name}")
            header[name] = {**entry, "file": file, "offset": 8+n+start}
    expected_shapes = {
        "thinker.model.embed_tokens.weight": [text["vocab_size"], 1024],
        "thinker.lm_head.weight": [5000 if aligner else text["vocab_size"], 1024],
        "thinker.audio_tower.conv_out.weight": [expected[0], 7680],
        "thinker.model.layers.0.self_attn.q_proj.weight": [2048, 1024],
    }
    for name, shape in expected_shapes.items():
        if name not in header or header[name]["shape"] != shape:
            raise ValueError(f"unexpected/missing tensor: {name}")
    return config, header, files, aligner


def convert(model, output, precision, fused_prefill=False):
    model, output = model.resolve(), output.resolve()
    if output.exists():
        raise ValueError("output already exists; refuse to overwrite a model package")
    config, tensors, files, aligner = inspect_source(model)
    if precision not in ("fp16", "int8", "hybrid", "hybrid4") or (aligner and precision.startswith("hybrid")):
        raise ValueError("hybrid* is an ASR-only CPU-compatible decode experiment")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        index = {"format": FORMAT, "task": "align" if aligner else "asr", "precision": precision,
                 "development_reference": precision == "fp16", "config": config,
                 "source_sha256": {p.name: digest(p) for p in files + [model / "config.json"]},
                 "quantization": {"scheme": "symmetric_per_output_row", "range": [-127, 127],
                                  "rounding": "ties_to_even", "scale_dtype": "f32", "zero_point": 0},
                 "tensors": {}}
        if precision == "hybrid":
            index["decode_quantization"] = {"scheme": "baseline_arm_neon_w8a8", "weight_rounding": "toward_zero",
                                             "activation_rounding": "nearest_away", "scope": "single_token_decoder_and_lm_head"}
        if precision == "hybrid4":
            index["decode_quantization"] = {"scheme": "w8a8_attention_and_head_plus_w4_mlp", "weight_rounding": "ties_to_even",
                                             "activation_rounding": "nearest_away", "scope": "single_token_decoder_and_lm_head",
                                             "note": "experiment: MLP decode weights INT4, attention/lm_head INT8"}
        fused_prefill = fused_prefill and precision in ("fp16", "hybrid")
        # Groups whose fp16 members are replaced by one concatenated tensor.
        fused_groups = {}
        for number in range(28):
            base = f"thinker.model.layers.{number}."
            fused_groups[base + "self_attn.fused_qkv_prefill.weight"] = [
                base + "self_attn.q_proj.weight", base + "self_attn.k_proj.weight", base + "self_attn.v_proj.weight"]
            fused_groups[base + "self_attn.fused_qkv_prefill.bias"] = [
                base + "self_attn.q_proj.bias", base + "self_attn.k_proj.bias", base + "self_attn.v_proj.bias"]
            fused_groups[base + "mlp.fused_gate_up_prefill.weight"] = [
                base + "mlp.gate_proj.weight", base + "mlp.up_proj.weight"]
        for layer in range(24):
            base = f"thinker.audio_tower.layers.{layer}."
            fused_groups[base + "self_attn.fused_qkv_prefill.weight"] = [
                base + "self_attn.q_proj.weight", base + "self_attn.k_proj.weight", base + "self_attn.v_proj.weight"]
            fused_groups[base + "self_attn.fused_qkv_prefill.bias"] = [
                base + "self_attn.q_proj.bias", base + "self_attn.k_proj.bias", base + "self_attn.v_proj.bias"]
        fused_members = {m for members in fused_groups.values() for m in members}
        pending_fused = {}
        dedup = {}
        with (work / "weights.bin").open("wb") as out:
            def write_tensor(name, data, dtype, logical_shape=None):
                out.write(b"\0" * ((-out.tell()) % 256))
                entry = {"dtype": dtype, "shape": list(logical_shape if logical_shape is not None else data.shape),
                         "offset": out.tell(), "bytes": data.nbytes}
                data.tofile(out)
                index["tensors"][name] = entry
                return entry
            for number, (name, meta) in enumerate(sorted(tensors.items())):
                raw = np.memmap(meta["file"], mode="r", dtype="<u2", offset=meta["offset"], shape=tuple(meta["shape"]))
                key = (tuple(raw.shape), hashlib.sha256(raw).hexdigest())
                aliased = key in dedup
                if aliased:
                    index["tensors"][name] = {**index["tensors"][dedup[key]], "alias_of": dedup[key]}
                values = (raw.astype(np.uint32) << 16).view(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite source values: {name}")
                if aliased:
                    pass
                elif values.ndim >= 2 and precision == "int8":
                    q, scales = quantize_rows(values)
                    entry = write_tensor(name, q, "i8")
                    scale_name = name + ".scale"
                    write_tensor(scale_name, scales, "f32")
                    entry["scale"] = scale_name
                    del q, scales
                else:
                    dtype = "f16" if values.ndim >= 2 else "f32"
                    data = values.astype("<f2" if dtype == "f16" else "<f4")
                    if not np.isfinite(data).all():
                        raise ValueError(f"FP16 overflow: {name}")
                    if not (fused_prefill and name in fused_members):
                        write_tensor(name, data, dtype)
                    del data
                if fused_prefill and name in fused_members:
                    for fused_name, members in fused_groups.items():
                        if name in members:
                            pending_fused.setdefault(fused_name, {})[name] = values.astype(np.float32)
                            if len(pending_fused[fused_name]) == len(members):
                                merged = np.concatenate([pending_fused[fused_name][m] for m in members], axis=0)
                                if merged.ndim >= 2:
                                    data = merged.astype("<f2")
                                    if not np.isfinite(data).all():
                                        raise ValueError(f"FP16 overflow: {fused_name}")
                                    entry = write_tensor(fused_name, data, "f16")
                                else:
                                    entry = write_tensor(fused_name, merged.astype("<f4"), "f32")
                                entry["fused_members"] = members
                            break
                if precision.startswith("hybrid") and values.ndim == 2 and (name.startswith("thinker.model.layers.") or name == "thinker.lm_head.weight"):
                    if precision == "hybrid4" and any(part in name for part in ("mlp.gate_proj.", "mlp.up_proj.", "mlp.down_proj.")):
                        q, scales = quantize_w4_rows(values)
                        entry = write_tensor(name+".decode", q, "i4", logical_shape=list(values.shape))
                        write_tensor(name+".decode.scale", scales, "f32")
                        entry["scale"] = name+".decode.scale"
                    else:
                        q, scales = quantize_cpu_decode_rows(values)
                        entry = write_tensor(name+".decode", q, "i8")
                        write_tensor(name+".decode.scale", scales, "f32")
                        entry["scale"] = name+".decode.scale"
                    del q, scales
                dedup.setdefault(key, name)
                del raw, values
                if number % 100 == 0:
                    print(f"{number}/{len(tensors)} tensors converted", flush=True)
            # Page-aligned mmap/no-copy Metal buffers, including on 16 KiB hosts.
            out.write(b"\0" * ((-out.tell()) % 65536))
        index["weights_sha256"] = digest(work / "weights.bin")
        index["weights_bytes"] = (work / "weights.bin").stat().st_size
        for name in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            if (model / name).is_file():
                shutil.copyfile(model / name, work / name)
        (work / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        work.rename(output)
    except BaseException:
        shutil.rmtree(work)
        raise
    print(f"{index['task']} {precision}: {len(tensors)} tensors, {index['weights_bytes']} bytes: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--precision", choices=("fp16", "int8", "hybrid", "hybrid4"), default="int8")
    p.add_argument("--fused-prefill", action="store_true",
                   help="write fused qkv/gate_up (and encoder qkv) weights instead of separate copies")
    args = p.parse_args()
    convert(args.model, args.output, args.precision, fused_prefill=args.fused_prefill)
