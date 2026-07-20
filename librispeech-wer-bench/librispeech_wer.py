#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path


PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”‘’—–…")
DEFAULT_DATASET_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"


def levenshtein(a: list[str], b: list[str]) -> int:
    dp = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, bv in enumerate(b, start=1):
            old = dp[j]
            if av == bv:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_items(dataset_dir: Path) -> list[tuple[str, Path, str]]:
    items: list[tuple[str, Path, str]] = []
    for transcript_file in sorted(dataset_dir.rglob("*.trans.txt")):
        chapter_dir = transcript_file.parent
        with transcript_file.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise SystemExit(f"Bad transcript line {transcript_file}:{line_no}: {line!r}")
                utt_id, reference = parts
                audio_path = chapter_dir / f"{utt_id}.flac"
                if not audio_path.exists():
                    raise SystemExit(f"Missing audio for transcript id {utt_id}: {audio_path}")
                items.append((utt_id, audio_path, reference))
    return items


def has_librispeech_items(dataset_dir: Path) -> bool:
    return dataset_dir.is_dir() and any(dataset_dir.rglob("*.trans.txt")) and any(dataset_dir.rglob("*.flac"))


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    dest = destination.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if target != dest and dest not in target.parents:
            raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
    tar.extractall(dest)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        pct = downloaded * 100.0 / total_size
        print(f"\rDownloading {url}: {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print("")


def prepare_dataset(args: argparse.Namespace) -> None:
    if has_librispeech_items(args.dataset):
        return
    if not args.download_dataset:
        raise SystemExit(
            f"Dataset directory not found or incomplete: {args.dataset}\n"
            "Pass --download-dataset to fetch LibriSpeech dev-clean automatically."
        )

    args.dataset.mkdir(parents=True, exist_ok=True)
    archive_name = Path(args.dataset_url).name or "dataset.tar.gz"
    archive_path = args.download_cache / archive_name
    if not archive_path.exists() or args.force_download:
        print(f"Downloading dataset archive to {archive_path}")
        download_file(args.dataset_url, archive_path)
    else:
        print(f"Using cached dataset archive: {archive_path}")

    with tempfile.TemporaryDirectory(prefix="qwen-librispeech-extract-") as tmp:
        extract_root = Path(tmp)
        print(f"Extracting {archive_path}")
        with tarfile.open(archive_path, "r:*") as tar:
            safe_extract_tar(tar, extract_root)

        candidates = [p for p in extract_root.rglob("*") if has_librispeech_items(p)]
        if not candidates:
            raise RuntimeError(f"No LibriSpeech transcript/audio pairs found in {archive_path}")
        source = min(candidates, key=lambda p: len(p.parts))

        for child in source.iterdir():
            dest = args.dataset / child.name
            if dest.exists():
                continue
            shutil.move(str(child), str(dest))

    if not has_librispeech_items(args.dataset):
        raise RuntimeError(f"Downloaded dataset is incomplete after extraction: {args.dataset}")


def convert_flac_to_wav(ffmpeg: str, flac: Path, wav: Path) -> None:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(flac),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(wav),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {flac}:\n{proc.stderr.strip()}")


def transcribe(args: argparse.Namespace, wav: Path) -> tuple[str, float]:
    cmd = [str(args.binary), "-d", str(args.model_dir), "-i", str(wav), "--silent"]
    if args.threads:
        cmd.extend(["-t", str(args.threads)])
    if args.mode == "segmented":
        cmd.extend(["-S", str(args.segment_sec)])
    elif args.mode == "streaming":
        cmd.append("--stream")
    if getattr(args, "prompt", None):
        cmd.extend(["--prompt", args.prompt])

    start = time.perf_counter()
    proc = run_cmd(cmd)
    wall_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(f"qwen-asr failed for {wav}:\n{proc.stderr.strip()}")
    return proc.stdout.strip(), wall_ms


def score(reference: str, hypothesis: str) -> dict[str, object]:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))
    word_edits = levenshtein(ref_words, hyp_words)
    char_edits = levenshtein(ref_chars, hyp_chars)
    return {
        "reference_norm": ref_norm,
        "hypothesis_norm": hyp_norm,
        "ref_words": len(ref_words),
        "word_edits": word_edits,
        "wer": word_edits / max(len(ref_words), 1),
        "ref_chars": len(ref_chars),
        "char_edits": char_edits,
        "cer": char_edits / max(len(ref_chars), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run qwen-asr WER evaluation on a LibriSpeech/Mini LibriSpeech directory."
    )
    parser.add_argument("--dataset", default="dev-clean-2", type=Path, help="LibriSpeech split directory")
    parser.add_argument("--binary", default="target/release/qwen-asr", type=Path, help="qwen-asr binary")
    parser.add_argument("--model-dir", default="qwen3-asr-0.6b", type=Path, help="Qwen ASR model directory")
    parser.add_argument("--output-dir", default="bench/wer-results", type=Path, help="Directory for JSONL/summary output")
    parser.add_argument("--label", default="", help="Run label; default is timestamp")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N utterances")
    parser.add_argument("--threads", type=int, default=0, help="Pass -t N to qwen-asr")
    parser.add_argument("--mode", choices=("offline", "segmented", "streaming"), default="offline")
    parser.add_argument("--segment-sec", type=int, default=30, help="Segment length for --mode segmented")
    parser.add_argument("--prompt", default="", help="Pass --prompt TEXT to qwen-asr (e.g. to A/B a system prompt)")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    parser.add_argument("--download-dataset", action="store_true", help="Download and extract LibriSpeech dev-clean if --dataset is missing")
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL, help="Dataset .tar.gz URL used with --download-dataset")
    parser.add_argument("--download-cache", type=Path, default=Path("librispeech-wer-bench/download-cache"), help="Where dataset archives are cached")
    parser.add_argument("--force-download", action="store_true", help="Redownload dataset archive even if it exists in the cache")
    args = parser.parse_args()

    prepare_dataset(args)
    if not args.binary.is_file():
        raise SystemExit(f"Binary not found: {args.binary}; run cargo build --release first")
    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_dir}")
    if shutil.which(args.ffmpeg) is None:
        raise SystemExit("ffmpeg not found on PATH; install ffmpeg to convert LibriSpeech FLAC files")

    items = find_items(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No LibriSpeech transcript/audio pairs found under {args.dataset}")

    label = args.label or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    total_ref_words = 0
    total_word_edits = 0
    total_ref_chars = 0
    total_char_edits = 0
    macro_wer_sum = 0.0
    macro_cer_sum = 0.0
    failures: list[dict[str, str]] = []

    print(f"Dataset: {args.dataset}")
    print(f"Items: {len(items)}")
    print(f"Mode: {args.mode}")
    print(f"Results: {run_dir}")

    with tempfile.TemporaryDirectory(prefix="qwen-librispeech-wer-") as tmp, jsonl_path.open(
        "w", encoding="utf-8"
    ) as out:
        tmp_dir = Path(tmp)
        for idx, (utt_id, flac, reference) in enumerate(items, start=1):
            print(f"[{idx}/{len(items)}] {utt_id}", flush=True)
            wav = tmp_dir / f"{utt_id}.wav"
            try:
                convert_flac_to_wav(args.ffmpeg, flac, wav)
                hypothesis, wall_ms = transcribe(args, wav)
                metrics = score(reference, hypothesis)
                total_ref_words += int(metrics["ref_words"])
                total_word_edits += int(metrics["word_edits"])
                total_ref_chars += int(metrics["ref_chars"])
                total_char_edits += int(metrics["char_edits"])
                macro_wer_sum += float(metrics["wer"])
                macro_cer_sum += float(metrics["cer"])
                row = {
                    "id": utt_id,
                    "audio": str(flac),
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wall_ms": round(wall_ms, 1),
                    **metrics,
                }
            except Exception as exc:
                failures.append({"id": utt_id, "audio": str(flac), "error": str(exc)})
                row = {"id": utt_id, "audio": str(flac), "reference": reference, "error": str(exc)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    ok_count = len(items) - len(failures)
    summary = {
        "label": label,
        "dataset": str(args.dataset),
        "mode": args.mode,
        "items": len(items),
        "ok": ok_count,
        "failed": len(failures),
        "corpus_wer": total_word_edits / max(total_ref_words, 1),
        "macro_wer": macro_wer_sum / max(ok_count, 1),
        "word_edits": total_word_edits,
        "ref_words": total_ref_words,
        "corpus_cer": total_char_edits / max(total_ref_chars, 1),
        "macro_cer": macro_cer_sum / max(ok_count, 1),
        "char_edits": total_char_edits,
        "ref_chars": total_ref_chars,
        "results_jsonl": str(jsonl_path),
        "failures": failures,
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print("")
    print(f"OK: {ok_count}/{len(items)}")
    print(f"Corpus WER: {summary['corpus_wer']:.4f}")
    print(f"Macro WER:  {summary['macro_wer']:.4f}")
    print(f"Corpus CER: {summary['corpus_cer']:.4f}")
    print(f"Summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
