#!/usr/bin/env python3
"""100-clip ASR accuracy audit (development tool, not an acceptance run).

Deterministically selects 100 fresh LibriSpeech dev-clean clips (seed differs
from the frozen 20-clip manifest), decodes them to frozen PCM, then runs the
CPU reference and the GPU candidate over identical inputs. Reports:
  - per-clip exact text match GPU vs CPU (the bit-exactness claim)
  - WER/CER of both sides against the human transcripts
"""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
SEED = "accuracy100"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
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


def select_clips(dataset, count, workdir):
    candidates = []
    for transcript in sorted(Path(dataset).rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            sample_id, reference = line.split(maxsplit=1)
            audio = transcript.parent / (sample_id + ".flac")
            if audio.is_file():
                candidates.append((sample_id, reference, audio))
    rank = lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()
    selected = []
    for sample_id, reference, audio in sorted(candidates, key=lambda v: rank(v[0])):
        wav = workdir / (sample_id + ".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(audio),
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        with wave.open(str(wav), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
                wav.unlink()
                continue
            duration = w.getnframes() / 16000
        if not 2 <= duration <= 30:
            wav.unlink()
            continue
        selected.append({"id": sample_id, "speaker": sample_id.split("-")[0], "reference": reference,
                         "wav": str(wav), "duration_s": duration, "wav_sha256": sha256(wav)})
        if len(selected) == count:
            return selected
    raise ValueError(f"only found {len(selected)} eligible clips")


class Adapter:
    def __init__(self, command, workdir):
        self.log = (workdir / (Path(command[0]).stem + ".stderr.log")).open("w")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        import queue
        self.lines = queue.Queue()
        threading.Thread(target=lambda: [self.lines.put(l) for l in self.process.stdout] or self.lines.put(None),
                         daemon=True).start()
        self.ready = json.loads(self.lines.get())
        if not self.ready.get("ready"):
            raise ValueError(f"adapter failed to start: {self.ready}")

    def infer(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return json.loads(self.lines.get())

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--dataset", type=Path, default=ROOT / "librispeech-wer-bench/dev-clean-2")
    parser.add_argument("--workdir", type=Path, default=PROJECT / "build/accuracy100")
    parser.add_argument("--cpu-model", type=Path, default=ROOT / "qwen3-asr-0.6b")
    parser.add_argument("--gpu-binary", type=Path, default=PROJECT / "target/release/qwen-asr-gpu")
    parser.add_argument("--gpu-model", type=Path, default=PROJECT / "models/hybrid-profile/asr")
    parser.add_argument("--kernels", type=Path, default=PROJECT / "native")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    clips = select_clips(args.dataset, args.count, args.workdir)
    print(f"selected {len(clips)} clips, {sum(c['duration_s'] for c in clips):.1f}s total audio", flush=True)

    cpu = Adapter([str(PROJECT / "target/release/qwen-asr-cpu-reference"), "asr",
                   str(args.cpu_model.resolve()), "0"], args.workdir)
    gpu = Adapter([str(args.gpu_binary), "asr", str(args.gpu_model.resolve()), str(args.kernels.resolve())],
                  args.workdir)
    rows = []
    for index, clip in enumerate(clips):
        request = {"op": "infer", "schema": 1, "id": f"acc100/{index}", "task": "asr",
                   "audio": clip["wav"], "language": "English"}
        t0 = time.perf_counter()
        cpu_out = cpu.infer(request)
        cpu_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        gpu_out = gpu.infer(request)
        gpu_ms = (time.perf_counter() - t0) * 1000
        if not cpu_out.get("ok") or not gpu_out.get("ok"):
            raise ValueError(f"adapter error on {clip['id']}: cpu={cpu_out} gpu={gpu_out}")
        e_cpu = errors(clip["reference"], cpu_out["text"])
        e_gpu = errors(clip["reference"], gpu_out["text"])
        rows.append({"id": clip["id"], "reference": clip["reference"], "cpu_text": cpu_out["text"],
                     "gpu_text": gpu_out["text"], "identical": cpu_out["text"] == gpu_out["text"],
                     "cpu": e_cpu, "gpu": e_gpu, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms,
                     "wav_sha256": clip["wav_sha256"]})
        if (index + 1) % 10 == 0:
            print(f"{index+1}/{len(clips)} done", flush=True)
    cpu.close()
    gpu.close()

    totals = lambda side: {k: sum(r[side][k] for r in rows) for k in ("word_errors", "reference_words",
                                                                     "char_errors", "reference_chars")}
    tc, tg = totals("cpu"), totals("gpu")
    report = {
        "clips": len(rows),
        "audio_seconds": round(sum(c["duration_s"] for c in clips), 1),
        "identical_texts": sum(r["identical"] for r in rows),
        "cpu_wer": tc["word_errors"] / tc["reference_words"],
        "gpu_wer": tg["word_errors"] / tg["reference_words"],
        "cpu_cer": tc["char_errors"] / tc["reference_chars"],
        "gpu_cer": tg["char_errors"] / tg["reference_chars"],
        "cpu_word_errors": tc["word_errors"], "gpu_word_errors": tg["word_errors"],
        "cpu_char_errors": tc["char_errors"], "gpu_char_errors": tg["char_errors"],
        "cpu_median_ms": statistics.median(r["cpu_ms"] for r in rows),
        "gpu_median_ms": statistics.median(r["gpu_ms"] for r in rows),
        "mismatches": [{"id": r["id"], "cpu": r["cpu_text"], "gpu": r["gpu_text"],
                        "cpu_we": r["cpu"]["word_errors"], "gpu_we": r["gpu"]["word_errors"]}
                       for r in rows if not r["identical"]],
    }
    out = args.workdir / "accuracy100.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    (args.workdir / "accuracy100-rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=1))
    print(f"mismatched transcripts: {len(report['mismatches'])} (details in {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
