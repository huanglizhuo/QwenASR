#!/usr/bin/env python3
"""Fast GPU-only iteration harness (development tool, not an acceptance run).

Runs the frozen 20-clip corpus through the GPU candidate only, several rounds,
and compares every text/word output plus per-case timing against the recorded
gpu-hybrid-clean-20260919 candidate outputs. Any output difference is a
regression signal; speed is reported per case and for the suite.
"""
import argparse
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent / "runs" / "gpu-hybrid-clean-20260919"


def baseline_records(task):
    rows = {}
    for line in (BASE / "records.jsonl").open():
        r = json.loads(line)
        if r["role"] == "candidate" and r["task"] == task and not r["warmup"]:
            rows.setdefault(r["case_id"], []).append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="both", choices=["asr", "align", "both"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--binary", default="qwen-asr-metal/target/release/qwen-asr-gpu")
    parser.add_argument("--models", default="qwen-asr-metal/models/hybrid-profile")
    parser.add_argument("--kernels", default="qwen-asr-metal/native")
    parser.add_argument("--label", default="iteration")
    parser.add_argument("--count", type=int, default=0, help="use only the first N manifest cases (0=all 20)")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parent / "manifest.json").read_text())
    tasks = ["asr", "align"] if args.task == "both" else [args.task]
    report = {"label": args.label, "tasks": {}}
    binary = str((ROOT / args.binary).resolve())
    for task in tasks:
        base = baseline_records(task)
        command = [binary, task, str((ROOT / args.models / task).resolve()), str((ROOT / args.kernels).resolve())]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] or lines.put(None), daemon=True).start()
        ready = json.loads(lines.get())
        assert ready.get("ready"), ready
        timing = {}
        mismatches = []
        assert args.count <= 10 or args.count == 0, "10-clip fast mode only"
        for case in manifest["cases"]:
            req = {"op": "infer", "schema": 1, "id": f"quick/{case['id']}", "task": task,
                   "audio": str(ROOT / case["audio"]), "language": "English"}
            if task == "align":
                req["text"] = case["reference"]
            samples = []
            # warmup + measured rounds for this case
            for r in range(args.runs + 1):
                t0 = time.perf_counter()
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                resp = json.loads(lines.get())
                dt = (time.perf_counter() - t0) * 1000
                assert resp.get("ok"), resp
                if r > 0:
                    samples.append((dt, resp))
            timing[case["id"]] = statistics.median(s[0] for s in samples)
            expected = base[case["id"]][0]
            got = samples[0][1]
            if got["text"] != expected["text"] or json.dumps(got["words"], sort_keys=True) != json.dumps(expected["words"], sort_keys=True):
                mismatches.append({"id": case["id"], "want": expected["text"], "got": got["text"]})
        proc.stdin.close()
        proc.wait(timeout=10)
        base_totals = {cid: statistics.median(r["elapsed_ms"] for r in rows) for cid, rows in base.items()}
        total_now = sum(timing[c["id"]] for c in manifest["cases"])
        total_base = sum(base_totals[c["id"]] for c in manifest["cases"])
        report["tasks"][task] = {
            "suite_median_sum_ms": total_now,
            "baseline_median_sum_ms": total_base,
            "speedup": total_base / total_now,
            "output_mismatches": mismatches,
            "per_case_ms": timing,
            "baseline_per_case_ms": base_totals,
        }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    ok = all(not t["output_mismatches"] for t in report["tasks"].values())
    for task, t in report["tasks"].items():
        print(f"{task}: {t['suite_median_sum_ms']:.1f}ms vs baseline {t['baseline_median_sum_ms']:.1f}ms "
              f"({t['speedup']:.3f}x), mismatches={len(t['output_mismatches'])}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
