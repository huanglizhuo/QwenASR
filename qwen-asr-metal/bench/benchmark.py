#!/usr/bin/env python3
"""Fixed-corpus ASR/align benchmark. Python standard library; ffmpeg for prepare.

Adapters are persistent JSONL subprocesses (see PROTOCOL.md). No shell commands
or transcript-derived references are used. Failed/missing cases fail the suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import wave

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "qwen-asr-metal"
CONTRACT = "qwen-asr-metal-bench-v1"
SEED = 20260919


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def pcm_info(path):
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError(f"expected mono PCM16 16000 Hz WAV: {path}")
        n = w.getnframes()
        pcm = w.readframes(n)
        if n == 0 or len(pcm) != n * 2:
            raise ValueError(f"empty or truncated WAV: {path}")
    return {"samples": n, "duration_s": n / 16000, "pcm_sha256": hashlib.sha256(pcm).hexdigest()}


def materialize(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)], check=True)
    return pcm_info(destination)


def select(args):
    """One-time deterministic selection, independent of recognition outputs."""
    manifest = args.manifest.resolve()
    if manifest.exists():
        raise ValueError("manifest already exists; use prepare to reproduce the fixed corpus")
    dataset = args.dataset.resolve()
    by_speaker = {}
    for transcript in sorted(dataset.rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            sample_id, reference = line.split(maxsplit=1)
            audio = transcript.parent / (sample_id + ".flac")
            if audio.is_file():
                by_speaker.setdefault(sample_id.split("-")[0], []).append((sample_id, reference, audio))
    rank = lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()
    selected = []
    # Distinct speakers, capped at 30 s to stay within the initial offline scope.
    for speaker in sorted(by_speaker, key=rank):
        for sample_id, reference, audio in sorted(by_speaker[speaker], key=lambda v: rank(v[0])):
            destination = PROJECT / "bench/data" / (sample_id + ".wav")
            info = materialize(audio, destination)
            if not 2 <= info["duration_s"] <= 30:
                destination.unlink()
                continue
            selected.append({"id": sample_id, "speaker": speaker, "language": "English",
                             "reference": reference, "source": str(audio.relative_to(dataset)),
                             "source_sha256": sha256(audio),
                             "audio": str(destination.relative_to(ROOT)), **info})
            break
        if len(selected) == 20:
            break
    if len(selected) != 20:
        raise ValueError(f"need 20 speakers with eligible audio, found {len(selected)}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {"schema": 1, "corpus": "LibriSpeech dev-clean", "seed": SEED,
                         "selection": "one clip per SHA256-ranked speaker; 2–30 s; independent of predictions",
                         "alignment_reference": "CPU regression oracle; no human boundary labels",
                         "cases": selected})
    print(f"Frozen {len(selected)} cases ({sum(c['duration_s'] for c in selected):.2f} s): {manifest}")


def prepare(args):
    manifest = load_manifest(args.manifest, check_audio=False)
    for case in manifest["cases"]:
        source = args.dataset / case["source"]
        if sha256(source) != case["source_sha256"]:
            raise ValueError(f"source hash mismatch: {case['id']}")
        info = materialize(source, ROOT / case["audio"])
        if any(info[k] != case[k] for k in info):
            raise ValueError(f"decoded PCM differs from frozen manifest: {case['id']}")
    print("20 audio files verified against frozen PCM hashes")


def load_manifest(path, check_audio=True):
    value = json.loads(Path(path).read_text())
    cases = value["cases"]
    if value.get("schema") != 1 or len(cases) != 20 or len({c["id"] for c in cases}) != 20:
        raise ValueError("schema 1 requires exactly 20 unique cases")
    for c in cases:
        if c["language"] != "English" or not tokens(c["reference"]):
            raise ValueError("v1 requires nonempty English references")
        if check_audio:
            info = pcm_info(ROOT / c["audio"])
            if any(info[k] != c[k] for k in info):
                raise ValueError(f"PCM mismatch: {c['id']}; run prepare with the frozen corpus")
    return value


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    text = text.replace("'", "")
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def edit_distance(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        row = [i]
        for j, h in enumerate(hyp, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (r != h)))
        previous = row
    return previous[-1]


def errors(reference, hypothesis):
    ref, hyp = tokens(reference), tokens(hypothesis)
    rc, hc = "".join(ref), "".join(hyp)
    return {"word_errors": edit_distance(ref, hyp), "reference_words": len(ref),
            "char_errors": edit_distance(rc, hc), "reference_chars": len(rc)}


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)]


def positive_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def validate_response(response, request, case, role):
    if response.get("schema") != 1 or response.get("ok") is not True:
        raise ValueError(f"adapter failed: {response}")
    if response.get("id") != request["id"] or response.get("task") != request["task"]:
        raise ValueError("response id/task mismatch")
    if response.get("sample_rate") != 16000 or response.get("audio_samples") != case["samples"]:
        raise ValueError("adapter did not process the complete frozen PCM input")
    positive_finite(response.get("elapsed_ms"), "elapsed_ms")
    if not isinstance(response.get("text"), str) or not isinstance(response.get("words"), list):
        raise ValueError("missing text/words output")
    if role == "candidate":
        if (response.get("backend") != "metal" or response.get("gpu_completed") is not True
                or type(response.get("cpu_tensor_fallbacks")) is not int
                or response["cpu_tensor_fallbacks"] != 0):
            raise ValueError("candidate must report completed Metal work with zero CPU tensor fallbacks")
    elif response.get("backend") != "cpu":
        raise ValueError("baseline must identify as cpu")
    if request["task"] == "align":
        expected = case["reference"].split()
        words = response["words"]
        if [w.get("text") for w in words] != expected:
            raise ValueError("alignment dropped, reordered or changed reference words")
        for w in words:
            start, end = w.get("start_ms"), w.get("end_ms")
            if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in (start, end)):
                raise ValueError("invalid alignment timestamp")


def alignment_issues(words, case):
    """Keep finite but geometrically invalid predictions for audit and timing.

    Existing CPU defects must not silently disappear, change the frozen corpus,
    or prevent measurement of the rest of it. They remain acceptance failures.
    """
    issues = []
    previous_end = 0.0
    for index, w in enumerate(words):
        start, end = w["start_ms"], w["end_ms"]
        if not 0 <= previous_end <= start <= end:
            issues.append({"word": index, "reason": "nonmonotonic", "start_ms": start, "end_ms": end})
        if end > case["duration_s"] * 1000 + 80:
            issues.append({"word": index, "reason": "out_of_audio", "end_ms": end,
                           "audio_ms": case["duration_s"] * 1000})
        previous_end = end
    return issues


class Adapter:
    def __init__(self, command, task, model, stderr_path, timeout):
        self.timeout = timeout
        self.command = [part.replace("{task}", task).replace("{model}", str(model)) for part in command]
        env = os.environ.copy()
        # Explicitly disable only optional speculative decode for matched greedy
        # semantics. Other QWEN_ASR_* tuning overrides are rejected by run().
        env["QWEN_ASR_VERIFY"] = "0"
        self.log = Path(stderr_path).open("w")
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1, env=env)
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.ready = self.receive()
            if self.ready.get("schema") != 1 or self.ready.get("ready") is not True or self.ready.get("contract") != CONTRACT:
                raise ValueError(f"invalid adapter handshake: {self.ready}")
            if self.ready.get("task") != task:
                raise ValueError("adapter loaded wrong task")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def receive(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as e:
            raise TimeoutError("adapter timed out; inspect stderr log") from e
        if line is None:
            raise RuntimeError("adapter exited before responding; inspect stderr log")
        return json.loads(line)

    def infer(self, request):
        start = time.perf_counter()
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self.receive()
        response["request_wall_ms"] = (time.perf_counter() - start) * 1000
        return response

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, BrokenPipeError):
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
        self.reader.join(timeout=1)
        self.log.close()


def summarize(records, cases, task, role):
    result = {"cases": [], "task": task, "role": role}
    round_totals, wall_totals = {}, {}
    totals = {"word_errors": 0, "reference_words": 0, "char_errors": 0, "reference_chars": 0}
    for case in cases:
        rows = [r for r in records if r["role"] == role and r["case_id"] == case["id"] and r["task"] == task]
        if not rows:
            raise ValueError(f"missing result: {task}/{role}/{case['id']}")
        outputs = {(r["text"], json.dumps(r["words"], sort_keys=True)) for r in rows}
        # Every repeated output is scored. A nondeterministic run fails rather
        # than selecting the fastest or most accurate transcript.
        score = errors(case["reference"], rows[0]["text"])
        for k in totals:
            totals[k] += score[k]
        timing = [r["elapsed_ms"] for r in rows]
        for r in rows:
            round_totals[r["round"]] = round_totals.get(r["round"], 0) + r["elapsed_ms"]
            wall_totals[r["round"]] = wall_totals.get(r["round"], 0) + r["request_wall_ms"]
        result["cases"].append({"id": case["id"], "text": rows[0]["text"], "words": rows[0]["words"],
                                "deterministic": len(outputs) == 1, "median_ms": statistics.median(timing),
                                "p95_ms": percentile(timing, .95), "runs": len(rows),
                                "alignment_issues": rows[0].get("alignment_issues", []), **score})
    rt = list(round_totals.values())
    result.update({"median_suite_ms": statistics.median(rt), "p95_suite_ms": percentile(rt, .95),
                   "median_request_suite_ms": statistics.median(wall_totals.values()),
                   "round_totals_ms": round_totals, "deterministic": all(c["deterministic"] for c in result["cases"]),
                   "audio_duration_s": sum(c["duration_s"] for c in cases), **totals})
    result["rtf"] = result["median_suite_ms"] / 1000 / result["audio_duration_s"]
    result["realtime_x"] = 1 / result["rtf"]
    result["wer"] = totals["word_errors"] / totals["reference_words"]
    result["cer"] = totals["char_errors"] / totals["reference_chars"]
    result["alignment_issue_count"] = sum(len(c["alignment_issues"]) for c in result["cases"])
    result["timing_stable"] = result["p95_suite_ms"] <= 1.25 * result["median_suite_ms"]
    return result


def compare(baseline, candidate):
    failures = []
    if not baseline["deterministic"] or not candidate["deterministic"]:
        failures.append("nondeterministic output across repeated runs")
    if not baseline.get("timing_stable", True) or not candidate.get("timing_stable", True):
        failures.append("P95/median spread exceeds 1.25; rerun under stable load")
    ratio = candidate["median_suite_ms"] / baseline["median_suite_ms"]
    if ratio > .70:
        failures.append("median suite latency is above 70% of CPU")
    if candidate["p95_suite_ms"] > baseline["p95_suite_ms"]:
        failures.append("P95 suite latency regressed")
    if candidate["median_request_suite_ms"] > .70 * baseline["median_request_suite_ms"]:
        failures.append("host-observed request latency did not improve by 30%")
    deltas = []
    for b, c in zip(baseline["cases"], candidate["cases"], strict=True):
        if b["id"] != c["id"] or b["runs"] != c["runs"]:
            raise ValueError("unpaired cases or run counts")
        if baseline["task"] == "asr":
            if c["word_errors"] > b["word_errors"] or c["char_errors"] > b["char_errors"]:
                failures.append(f"recognition accuracy regressed: {b['id']}")
        else:
            if b.get("alignment_issues"):
                failures.append(f"CPU alignment requires boundary review before acceptance: {b['id']}")
            if c.get("alignment_issues"):
                failures.append(f"candidate alignment contains invalid boundaries: {b['id']}")
            if len(b["words"]) != len(c["words"]):
                failures.append(f"alignment word count changed: {b['id']}")
                continue
            for bw, cw in zip(b["words"], c["words"]):
                if bw["text"] != cw["text"]:
                    failures.append(f"alignment text changed: {b['id']}")
                deltas.extend(abs(cw[k] - bw[k]) for k in ("start_ms", "end_ms"))
    if deltas and max(deltas) > 0:
        failures.append("alignment timestamps differ from CPU (strict 0 ms regression budget)")
    # Paired bootstrap of full-suite rounds. Report uncertainty; don't pretend
    # seven rounds are a broad hardware or accuracy certification.
    pairs = [(baseline["round_totals_ms"][i], candidate["round_totals_ms"][i])
             for i in baseline["round_totals_ms"]]
    rng = random.Random(SEED)
    ratios = []
    for _ in range(2000):
        sample = [rng.choice(pairs) for _ in pairs]
        ratios.append(statistics.median(v[1] for v in sample) / statistics.median(v[0] for v in sample))
    return {"passed": not failures, "latency_ratio": ratio, "latency_reduction": 1 - ratio,
            "ratio_bootstrap_95_ci": [percentile(ratios, .025), percentile(ratios, .975)],
            "alignment_mean_delta_ms": statistics.mean(deltas) if deltas else None,
            "alignment_max_delta_ms": max(deltas) if deltas else None, "failures": failures}


def command_json(raw):
    value = json.loads(raw)
    if not isinstance(value, list) or not value or any(not isinstance(s, str) for s in value):
        raise ValueError("adapter command must be a nonempty JSON array of strings")
    executable = Path(value[0]).expanduser().resolve()
    if not executable.is_file():
        raise ValueError(f"adapter executable missing: {executable}")
    value[0] = str(executable)
    return value


def model_identity(model):
    model = model.resolve()
    paths = sorted(model.glob("*.safetensors"))
    if not paths or not (model / "config.json").is_file():
        raise ValueError(f"missing safetensors/config: {model}")
    paths += [model / "config.json"]
    paths += [model / name for name in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "generation_config.json") if (model / name).exists()]
    return {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths}


def candidate_identity(ready, task, original):
    """Pin the actually loaded package and runtime-compiled shader sources.

    Native code is covered by the executable hash. Without these extra hashes,
    changing a .metal file would leave that executable hash unchanged.
    """
    package = Path(ready["package_path"]).resolve()
    kernels = Path(ready["kernel_directory"]).resolve()
    index = json.loads((package / "index.json").read_text())
    if index.get("format") != "qwen-asr-metal-v1" or index.get("task") != task:
        raise ValueError("candidate package format/task mismatch")
    expected = {name: info["sha256"] for name, info in original.items()
                if name.endswith(".safetensors") or name == "config.json"}
    if index.get("source_sha256") != expected:
        raise ValueError("candidate package was converted from a different source model")
    files = [package / "index.json", package / "weights.bin"]
    for name, info in original.items():
        if name.endswith(".safetensors") or name in ("config.json", "generation_config.json"):
            continue
        path = package / name
        if not path.is_file() or sha256(path) != info["sha256"]:
            raise ValueError(f"candidate tokenizer asset differs: {name}")
        files.append(path)
    files += [kernels / name for name in ("quantized.metal", "ops.metal", "decoder.metal", "frontend.metal")]
    identities = {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}
    weights = identities[str(package / "weights.bin")]
    if weights["sha256"] != index["weights_sha256"] or weights["sha256"] != ready["weights_sha256"] or weights["bytes"] != index["weights_bytes"]:
        raise ValueError("candidate weight checksum/size differs from loaded package")
    return identities


def environment_info():
    def read(command):
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()
    # Deliberately exclude serial numbers, hardware UUIDs and personal env vars.
    return {"platform": platform.platform(), "machine": platform.machine(),
            "macos": read(["sw_vers", "-productVersion"]),
            "chip": read(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_bytes": read(["sysctl", "-n", "hw.memsize"]),
            "xcode": read(["xcodebuild", "-version"]),
            "commit": read(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "working_tree": read(["git", "-C", str(ROOT), "status", "--short"])}


def report_markdown(report):
    lines = ["# Qwen3-ASR Metal benchmark", "", f"Status: **{report['status']}**", "",
             "20 frozen English LibriSpeech clips; human transcript references. Alignment uses the CPU output as a regression oracle, not human timing ground truth.", "",
             "| Task | Engine | Median suite ms | P95 suite ms | WER | CER |", "|---|---|---:|---:|---:|---:|"]
    for task, entries in report.get("tasks", {}).items():
        for role in ("baseline", "candidate"):
            if role in entries:
                s = entries[role]
                wer = f"{s['wer']:.4%}" if task == "asr" else "N/A"
                cer = f"{s['cer']:.4%}" if task == "asr" else "N/A"
                lines.append(f"| {task} | {role} | {s['median_suite_ms']:.2f} | {s['p95_suite_ms']:.2f} | {wer} | {cer} |")
    lines.append("")
    for task, entries in report.get("tasks", {}).items():
        for role in ("baseline", "candidate"):
            if role in entries and entries[role].get("alignment_issue_count", 0):
                lines.append(f"**{task}/{role}: {entries[role]['alignment_issue_count']} boundary issues need review; alignment acceptance is blocked.**")
            if role in entries and not entries[role].get("timing_stable", True):
                lines.append(f"**{task}/{role}: timing spread exceeds 1.25×; this run cannot certify speedup.**")
        if "comparison" in entries:
            c = entries["comparison"]
            lines += ["", f"{task}: latency reduction {c['latency_reduction']:.2%}; passed={c['passed']}."]
            lines += [f"- {reason}" for reason in c["failures"]]
    if report.get("error"):
        lines += ["", "Error: " + report["error"]]
    lines += ["", "Primary timing includes PCM feature extraction, encoder, decoder and result construction; model loading and WAV decoding are separate. Host request timing includes I/O and protocol overhead.",
              "", "A passing comparison applies only to this corpus and machine. GPU residency and absence of CPU tensor execution also require a Metal System Trace/code audit; adapter declarations alone are not proof."]
    return "\n".join(lines) + "\n"


def run(args):
    if args.runs < 3 or args.warmups < 1:
        raise ValueError("use at least 3 measured rounds and 1 warmup round")
    overrides = sorted(k for k in os.environ if k.startswith("QWEN_ASR_"))
    if overrides:
        raise ValueError("remove QWEN_ASR_* environment overrides for reproducibility: " + ", ".join(overrides))
    manifest = load_manifest(args.manifest)
    baseline_command = command_json(args.baseline)
    candidate_command = command_json(args.candidate) if args.candidate else None
    if candidate_command and candidate_command == baseline_command:
        raise ValueError("candidate cannot be the CPU reference command")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "status": "running", "contract": CONTRACT,
              "manifest_sha256": sha256(args.manifest), "environment": environment_info(),
              "runs": args.runs, "warmups": args.warmups, "seed": SEED,
              "criteria": {"max_latency_ratio": .70, "per_case_wer_cer_regression": 0,
                           "alignment_timestamp_delta_ms": 0, "max_p95_ratio": 1.0,
                           "max_p95_median_spread": 1.25},
              "commands": {"baseline": baseline_command, "candidate": candidate_command},
              "binary_sha256": {"baseline": sha256(baseline_command[0]),
                                "candidate": sha256(candidate_command[0]) if candidate_command else None},
              "tasks": {}}
    # Save the actual corpus and runner, not just the git revision (which may
    # contain uncommitted benchmark work).
    write_json(out / "manifest.json", manifest)
    report["runner_sha256"] = sha256(__file__)
    (out / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    records = []
    try:
        report["models"] = {task: model_identity(model) for task, model in (("asr", args.asr_model), ("align", args.align_model))}
        with (out / "records.jsonl").open("w") as raw:
            for task, model in (("asr", args.asr_model), ("align", args.align_model)):
                adapters = {}
                try:
                    adapters["baseline"] = Adapter(baseline_command, task, model.resolve(), out / f"{task}-baseline.stderr.log", args.timeout)
                    if candidate_command:
                        adapters["candidate"] = Adapter(candidate_command, task, model.resolve(), out / f"{task}-candidate.stderr.log", args.timeout)
                    report["tasks"][task] = {"handshakes": {role: a.ready for role, a in adapters.items()}}
                    if candidate_command:
                        report["tasks"][task]["candidate_artifacts"] = candidate_identity(adapters["candidate"].ready, task, report["models"][task])
                    rng = random.Random(SEED)
                    for iteration in range(-args.warmups, args.runs):
                        cases = list(manifest["cases"])
                        rng.shuffle(cases)
                        for index, case in enumerate(cases):
                            roles = list(adapters)
                            if (iteration + index) % 2:
                                roles.reverse()
                            for role in roles:
                                request = {"op": "infer", "schema": 1, "id": f"{task}/{iteration}/{case['id']}/{role}",
                                           "task": task, "audio": str(ROOT / case["audio"]), "language": case["language"]}
                                # Never expose the reference transcript to an ASR candidate.
                                if task == "align":
                                    request["text"] = case["reference"]
                                response = adapters[role].infer(request)
                                try:
                                    validate_response(response, request, case, role)
                                except Exception as error:
                                    raw.write(json.dumps({"role": role, "case_id": case["id"], "response": response,
                                                          "validation_error": str(error)}, ensure_ascii=False) + "\n")
                                    raw.flush()
                                    raise
                                if response["elapsed_ms"] > response["request_wall_ms"] + 2:
                                    raise ValueError("reported inference time exceeds host request duration")
                                row = {**response, "role": role, "case_id": case["id"], "round": iteration, "warmup": iteration < 0,
                                       "alignment_issues": alignment_issues(response["words"], case) if task == "align" else []}
                                raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                raw.flush()
                                if iteration >= 0:
                                    records.append(row)
                        print(f"{task}: {'warmup' if iteration < 0 else 'round'} {iteration + 1}/{args.runs} complete", flush=True)
                    for role in adapters:
                        report["tasks"][task][role] = summarize(records, manifest["cases"], task, role)
                    if candidate_command:
                        report["tasks"][task]["comparison"] = compare(report["tasks"][task]["baseline"], report["tasks"][task]["candidate"])
                finally:
                    for adapter in adapters.values():
                        adapter.close()
        baseline_stable = all(v["baseline"]["deterministic"] for v in report["tasks"].values())
        if not baseline_stable:
            report["status"] = "failed"
            report["error"] = "CPU baseline is nondeterministic; investigate before comparing"
        elif candidate_command:
            report["status"] = "comparison_passed_requires_gpu_audit" if all(t["comparison"]["passed"] for t in report["tasks"].values()) else "failed"
        else:
            report["status"] = "baseline_only_gpu_not_implemented"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(out / "summary.json", report)
        (out / "report.md").write_text(report_markdown(report))
    print(f"{report['status']}: {out / 'report.md'}")
    return 2 if report["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("select", "prepare", "run"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, default=PROJECT / "bench/manifest.json")
        if name != "run":
            command.add_argument("--dataset", type=Path, default=ROOT / "librispeech-wer-bench/dev-clean-2")
        else:
            command.add_argument("--baseline", default=json.dumps([str(PROJECT / "target/release/qwen-asr-cpu-reference"), "{task}", "{model}", "0"]))
            command.add_argument("--candidate", help="JSON argv, with {task} and {model} placeholders")
            command.add_argument("--asr-model", type=Path, default=ROOT / "qwen3-asr-0.6b")
            command.add_argument("--align-model", type=Path, default=ROOT / "qwen3-aligner-0.6b")
            command.add_argument("--runs", type=int, default=7)
            command.add_argument("--warmups", type=int, default=1)
            command.add_argument("--timeout", type=float, default=120)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return {"select": select, "prepare": prepare, "run": run}[args.action](args) or 0
    except Exception as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
