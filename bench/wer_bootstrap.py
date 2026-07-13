#!/usr/bin/env python3
"""Paired bootstrap WER gate for two qwen-asr builds (R12-B3).

Consumes two per-utterance results.jsonl files produced by
librispeech-wer-bench/librispeech_wer.py (each row has id/word_edits/ref_words),
aligns them by utterance id, and reports:

  * corpus WER for build A and build B (sum edits / sum ref words),
  * a paired bootstrap (default 10,000 resamples over utterances) for the
    corpus-WER difference dWER = WER_B - WER_A: 95% CI and P(B worse),
  * the same statistics restricted to a continuity subset of utt ids
    (e.g. the historical 100-file dev-clean-2 subset), and
  * per-utterance edit-count summary stats for both builds.

WER is deterministic per binary (greedy decode), so the only randomness is the
bootstrap resampling of which utterances land in the corpus. This is the
large-set gate that R12-B3 recommends for future quantization experiments.

Usage:
  wer_bootstrap.py --a A/results.jsonl --b B/results.jsonl \
      [--subset-ids hist100.txt] [--resamples 10000] [--seed 0] \
      [--label-a "INT8 FFN"] [--label-b "INT4 FFN"]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "error" in row or "word_edits" not in row or "ref_words" not in row:
                continue
            rows[row["id"]] = row
    return rows


def corpus_wer(edits: list[int], refs: list[int]) -> float:
    total_refs = sum(refs)
    return sum(edits) / total_refs if total_refs else 0.0


def paired_bootstrap(
    edits_a: list[int],
    edits_b: list[int],
    refs: list[int],
    resamples: int,
    seed: int,
) -> dict:
    """Resample utterance indices with replacement; corpus WER is recomputed on
    the resampled multiset for both builds (paired: same indices for A and B)."""
    n = len(refs)
    rng = random.Random(seed)
    deltas: list[float] = []
    wer_a_bs: list[float] = []
    wer_b_bs: list[float] = []
    b_worse = 0
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        ra = sum(refs[i] for i in idx)
        ea = sum(edits_a[i] for i in idx)
        eb = sum(edits_b[i] for i in idx)
        wa = ea / ra if ra else 0.0
        wb = eb / ra if ra else 0.0
        d = wb - wa
        deltas.append(d)
        wer_a_bs.append(wa)
        wer_b_bs.append(wb)
        if d > 0:
            b_worse += 1
    deltas.sort()
    lo = deltas[int(0.025 * resamples)]
    hi = deltas[int(0.975 * resamples)]
    point = corpus_wer(edits_b, refs) - corpus_wer(edits_a, refs)
    return {
        "n_utts": n,
        "resamples": resamples,
        "wer_a": corpus_wer(edits_a, refs),
        "wer_b": corpus_wer(edits_b, refs),
        "delta_point": point,
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "delta_mean": statistics.fmean(deltas),
        "p_b_worse": b_worse / resamples,
        "rel_delta": point / corpus_wer(edits_a, refs) if corpus_wer(edits_a, refs) else 0.0,
    }


def edit_stats(edits: list[int]) -> dict:
    s = sorted(edits)
    n = len(s)
    return {
        "total": sum(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "max": s[-1] if s else 0,
        "n_zero": sum(1 for e in s if e == 0),
        "frac_zero": (sum(1 for e in s if e == 0) / n) if n else 0.0,
    }


def fmt_report(title: str, bs: dict, label_a: str, label_b: str) -> str:
    lines = [
        f"## {title} (n={bs['n_utts']} utterances, {bs['resamples']} resamples)",
        f"  corpus WER {label_a}: {bs['wer_a']:.4f}",
        f"  corpus WER {label_b}: {bs['wer_b']:.4f}",
        f"  dWER (B-A) point:  {bs['delta_point']:+.5f}  ({bs['rel_delta']*100:+.2f}% relative)",
        f"  dWER 95% CI:       [{bs['delta_ci_lo']:+.5f}, {bs['delta_ci_hi']:+.5f}]",
        f"  P(B worse):        {bs['p_b_worse']:.4f}",
        f"  CI excludes 0:     {'YES' if (bs['delta_ci_lo'] > 0 or bs['delta_ci_hi'] < 0) else 'NO'}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path, help="build A results.jsonl (baseline, e.g. INT8 FFN)")
    ap.add_argument("--b", required=True, type=Path, help="build B results.jsonl (candidate, e.g. INT4 FFN)")
    ap.add_argument("--subset-ids", type=Path, default=None, help="file of utt ids (one per line) for the continuity subset")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows_a = load_rows(args.a)
    rows_b = load_rows(args.b)
    common = sorted(set(rows_a) & set(rows_b))
    only_a = set(rows_a) - set(rows_b)
    only_b = set(rows_b) - set(rows_a)
    if only_a or only_b:
        print(f"WARNING: {len(only_a)} ids only in A, {len(only_b)} ids only in B; using {len(common)} paired.")

    def vecs(ids: list[str]):
        ea = [int(rows_a[i]["word_edits"]) for i in ids]
        eb = [int(rows_b[i]["word_edits"]) for i in ids]
        rf = [int(rows_a[i]["ref_words"]) for i in ids]  # ref identical across builds
        return ea, eb, rf

    ea, eb, rf = vecs(common)
    full = paired_bootstrap(ea, eb, rf, args.resamples, args.seed)

    print(fmt_report("LARGE SET", full, args.label_a, args.label_b))
    print(f"  per-utt edits {args.label_a}: {edit_stats(ea)}")
    print(f"  per-utt edits {args.label_b}: {edit_stats(eb)}")
    print(f"  ref words total: {sum(rf)}")

    result = {"large": full,
              "edit_stats_a": edit_stats(ea),
              "edit_stats_b": edit_stats(eb),
              "ref_words": sum(rf)}

    if args.subset_ids and args.subset_ids.is_file():
        want = [l.strip() for l in args.subset_ids.read_text().splitlines() if l.strip()]
        sub = [i for i in want if i in rows_a and i in rows_b]
        missing = [i for i in want if i not in (set(rows_a) & set(rows_b))]
        sea, seb, srf = vecs(sub)
        subbs = paired_bootstrap(sea, seb, srf, args.resamples, args.seed)
        print()
        print(fmt_report(f"CONTINUITY SUBSET ({len(sub)}/{len(want)} ids present)", subbs, args.label_a, args.label_b))
        if missing:
            print(f"  (missing {len(missing)} subset ids)")
        print(f"  per-utt edits {args.label_a}: {edit_stats(sea)}")
        print(f"  per-utt edits {args.label_b}: {edit_stats(seb)}")
        result["subset"] = subbs
        result["subset_edit_stats_a"] = edit_stats(sea)
        result["subset_edit_stats_b"] = edit_stats(seb)

    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
