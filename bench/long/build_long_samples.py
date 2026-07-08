#!/usr/bin/env python3
"""Deterministic long-audio sample builder for the q-asr long-audio benchmark.

Concatenates LibriSpeech dev-clean-2 utterances into long test WAVs so the
long-audio optimization track (F27 shared-weights split -> F28 parallel segment
transcription) has something to measure. Everything here is tooling: it does not
touch any library or CLI code path used in production.

Determinism contract
--------------------
* Utterances are taken from ``librispeech-wer-bench/dev-clean-2`` in a FIXED
  order: sorted ascending by their POSIX relative FLAC path
  (equivalently, by ``<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac``).
* A fixed silence gap (default 0.5 s) is inserted *between* utterances (never a
  leading or trailing gap).
* Utterances are appended in that fixed order until the accumulated audio +
  gaps first reaches the target duration; the utterance that crosses the
  threshold is included, so the final duration is slightly over target and is a
  pure function of the target/gap and the dataset. No randomness anywhere.
* Audio is rendered to 16 kHz mono signed-16-bit PCM WAV via ffmpeg, matching
  the converter used by ``librispeech_wer.py`` so scores stay comparable.

For each sample the builder writes:
* ``manifests/<name>.txt`` -- committed manifest: header comment lines
  (``#key value``) documenting order/gap/target/duration/md5, then one utterance
  id per line, in order. This is the reproducibility record.
* ``samples/<name>.wav`` -- the rendered long WAV (gitignored; rebuilt here).
* ``samples/<name>.ref.txt`` -- sidecar reference transcript: the utterances'
  reference texts joined with single spaces, in the same order (gitignored;
  rebuilt here).

Rebuilding is idempotent: same dataset + same flags => identical manifest and
identical WAV md5.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
WER_BENCH_DIR = PROJECT_DIR / "librispeech-wer-bench"

# Reuse the existing LibriSpeech harness so utterance discovery and the FLAC->WAV
# conversion stay identical to the short-utterance gate.
sys.path.insert(0, str(WER_BENCH_DIR))
from librispeech_wer import convert_flac_to_wav, find_items  # noqa: E402

# Sample targets: (name, target_seconds).
DEFAULT_SAMPLES = [
    ("long-2min", 120.0),
    ("long-10min", 600.0),
]


def ffprobe_duration(ffprobe: str, path: Path) -> float:
    out = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        text=True,
    )
    return float(out.strip())


def make_silence_wav(ffmpeg: str, seconds: float, dest: Path) -> None:
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", f"{seconds:.6f}",
        "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        str(dest),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg silence generation failed:\n{proc.stderr.strip()}")


def concat_wavs(ffmpeg: str, parts: list[Path], dest: Path) -> None:
    """Loss-preserving concat of identically-formatted WAVs via the demuxer."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        list_path = Path(fh.name)
        for p in parts:
            # ffconcat list format: escape single quotes.
            escaped = str(p).replace("'", "'\\''")
            fh.write(f"file '{escaped}'\n")
    try:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            str(dest),
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{proc.stderr.strip()}")
    finally:
        list_path.unlink(missing_ok=True)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_utterances(items, gap: float, target: float):
    """Return the ordered sublist of items whose audio+gaps first reach target.

    ``items`` must already be in the fixed deterministic order. Durations are the
    raw FLAC durations reported by ffprobe (the rendered WAV is the same content
    resampled to 16 kHz mono, so total duration matches within resampling noise).
    """
    chosen = []
    acc = 0.0
    for idx, item in enumerate(items):
        utt_id, flac, reference, dur = item
        if chosen:
            acc += gap
        acc += dur
        chosen.append(item)
        if acc >= target:
            break
    return chosen, acc


def build_sample(args, name: str, target: float, items) -> dict:
    ffmpeg = args.ffmpeg
    ffprobe = args.ffprobe
    chosen, est_dur = select_utterances(items, args.gap, target)

    samples_dir = SCRIPT_DIR / "samples"
    manifests_dir = SCRIPT_DIR / "manifests"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    wav_path = samples_dir / f"{name}.wav"
    ref_path = samples_dir / f"{name}.ref.txt"
    manifest_path = manifests_dir / f"{name}.txt"

    with tempfile.TemporaryDirectory(prefix=f"long-build-{name}-") as tmp:
        tmp_dir = Path(tmp)
        silence = tmp_dir / "gap.wav"
        make_silence_wav(ffmpeg, args.gap, silence)

        parts: list[Path] = []
        for i, (utt_id, flac, reference, dur) in enumerate(chosen):
            seg = tmp_dir / f"{i:04d}_{utt_id}.wav"
            convert_flac_to_wav(ffmpeg, flac, seg)
            if i > 0:
                parts.append(silence)
            parts.append(seg)

        concat_wavs(ffmpeg, parts, wav_path)

    real_dur = ffprobe_duration(ffprobe, wav_path)
    reference_text = " ".join(item[2].strip() for item in chosen)
    ref_path.write_text(reference_text + "\n", encoding="utf-8")

    wav_md5 = md5_file(wav_path)

    header = [
        "# q-asr long-audio benchmark manifest (deterministic; do not hand-edit)",
        f"# name {name}",
        f"# dataset librispeech-wer-bench/dev-clean-2",
        f"# order sorted_ascending_by_relative_flac_path",
        f"# gap_seconds {args.gap}",
        f"# target_seconds {target}",
        f"# utterances {len(chosen)}",
        f"# audio_seconds {real_dur:.3f}",
        f"# wav_md5 {wav_md5}",
        f"# ref_words {len(reference_text.split())}",
    ]
    lines = header + [item[0] for item in chosen]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{name}] {len(chosen)} utts, {real_dur:.1f}s, wav_md5={wav_md5}")
    print(f"        wav={wav_path}")
    print(f"        ref={ref_path}")
    print(f"        manifest={manifest_path}")
    return {
        "name": name,
        "utterances": len(chosen),
        "audio_seconds": real_dur,
        "wav_md5": wav_md5,
        "wav": str(wav_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=WER_BENCH_DIR / "dev-clean-2",
                        help="LibriSpeech split directory (default: dev-clean-2)")
    parser.add_argument("--gap", type=float, default=0.5,
                        help="Silence gap in seconds between utterances (default: 0.5)")
    parser.add_argument("--ffmpeg", default="/opt/homebrew/bin/ffmpeg", help="ffmpeg executable")
    parser.add_argument("--ffprobe", default="/opt/homebrew/bin/ffprobe", help="ffprobe executable")
    parser.add_argument("--only", default="", help="Comma-separated sample names to build (default: all)")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.dataset}")

    # Discover all utterances, then impose the fixed order: sort ascending by the
    # POSIX relative FLAC path. find_items() returns (utt_id, flac_path, reference).
    raw = find_items(args.dataset)
    items = []
    for utt_id, flac, reference in raw:
        rel = flac.resolve().relative_to(args.dataset.resolve()).as_posix()
        items.append((utt_id, flac, reference, rel))
    items.sort(key=lambda t: t[3])
    # Attach durations (ffprobe) in the fixed order.
    ordered = []
    for utt_id, flac, reference, rel in items:
        dur = ffprobe_duration(args.ffprobe, flac)
        ordered.append((utt_id, flac, reference, dur))

    wanted = set(s.strip() for s in args.only.split(",") if s.strip())
    summaries = []
    for name, target in DEFAULT_SAMPLES:
        if wanted and name not in wanted:
            continue
        summaries.append(build_sample(args, name, target, ordered))

    print("")
    print("Built samples:")
    for s in summaries:
        print(f"  {s['name']}: {s['utterances']} utts, {s['audio_seconds']:.1f}s, md5={s['wav_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
