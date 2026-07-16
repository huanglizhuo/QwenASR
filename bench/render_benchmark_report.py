#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "rust-original": "#4C78A8",
    "rust-current": "#F58518",
    "c-antirez": "#54A24B",
    "second-state-mlx": "#B279A2",
    "mlx-audio": "#E45756",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_check_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


def memory_gb() -> str:
    raw = safe_check_output(["sysctl", "-n", "hw.memsize"])
    if raw and raw != "unknown":
        try:
            return str(round(int(raw) / (1024 ** 3), 1))
        except ValueError:
            pass
    return "unknown"


def load_system_info(path: Path | None) -> dict[str, str]:
    defaults = {
        "cpu_brand": safe_check_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_cores": safe_check_output(["sysctl", "-n", "hw.physicalcpu"]),
        "logical_cores": safe_check_output(["sysctl", "-n", "hw.logicalcpu"]),
        "memory_gb": memory_gb(),
        "arch": safe_check_output(["uname", "-m"]),
        "macos_version": safe_check_output(["sw_vers", "-productVersion"]),
        "rustc_version": safe_check_output(["rustc", "--version"]).splitlines()[0]
        if safe_check_output(["rustc", "--version"]) != "unknown"
        else "unknown",
    }
    if path and path.exists():
        try:
            data = load_json(path)
            return {**defaults, **data}
        except Exception:
            pass
    return defaults


def pick_result(items: list[dict], impl: str, accelerate: bool, mode: str = "offline") -> dict | None:
    for item in items:
        if item.get("impl") == impl and item.get("accelerate") == accelerate and item.get("mode") == mode:
            return item
    return None


def chart_rows_from_summary(items: list[dict], baseline_ref: str, current_ref: str) -> list[dict]:
    labels = {
        "rust-original": f"qwen-asr first\n{baseline_ref}",
        "rust-current": f"qwen-asr latest\n{current_ref}",
        "c-antirez": "pure C\nupstream",
        "second-state-mlx": "second-state\nMLX GPU",
        "mlx-audio": "mlx-audio\nPython MLX",
    }
    rows = []
    for impl, label in labels.items():
        item = next(
            (
                candidate
                for candidate in items
                if candidate.get("impl") == impl
                and candidate.get("mode") == "offline"
                and candidate.get("run_ok")
            ),
            None,
        )
        if not item or not item.get("run_ok"):
            continue
        rows.append(
            {
                "impl": impl,
                "label": label,
                "total_ms": float(item["total_ms"]),
                "realtime_factor": float(item["realtime_factor"]),
                "inference_mean_ms": float(item["inference_mean_ms"]) if item.get("inference_mean_ms") is not None else None,
                "inference_best_ms": float(item["inference_best_ms"]) if item.get("inference_best_ms") is not None else None,
                "wall_clock_ms": float(item["wall_clock_ms"]) if item.get("wall_clock_ms") is not None else None,
                "wall_clock_realtime_factor": float(item["wall_clock_realtime_factor"]) if item.get("wall_clock_realtime_factor") is not None else None,
                "wall_clock_mean_ms": float(item["wall_clock_mean_ms"]) if item.get("wall_clock_mean_ms") is not None else None,
                "wall_clock_best_ms": float(item["wall_clock_best_ms"]) if item.get("wall_clock_best_ms") is not None else None,
                "commit": item.get("commit"),
            }
        )
    # Sort by median inference latency ascending (fastest first) so charts read left-to-right by speed.
    rows.sort(key=lambda r: r["total_ms"])
    return rows


def nice_upper_bound(values: list[float]) -> float:
    vmax = max(values)
    if vmax <= 0:
        return 1.0
    padded = vmax * 1.15
    magnitude = 10 ** math.floor(math.log10(padded))
    scaled = padded / magnitude
    for candidate in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if scaled <= candidate:
            return candidate * magnitude
    return 10 * magnitude


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f}" if value >= 100 else f"{value:.2f}"


def render_bar_chart(rows: list[dict], metric: str, ylabel: str, title: str, subtitle: str, output_path: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [row[metric] for row in rows]
    colors = [COLORS[row["impl"]] for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=200)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ymax = nice_upper_bound(values) * 1.05
    ax.set_ylim(0, ymax)

    for bar, value in zip(bars, values):
        if metric == "total_ms":
            label = f"{value:,.0f} ms" if value >= 100 else f"{value:.2f} ms"
        else:
            label = f"{value:.2f}x"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_markdown_report(
    report_dir: Path,
    chart_dir: Path,
    summary_items: list[dict],
    baseline_ref: str,
    current_ref: str,
    model_dir: str,
    input_file: str,
    runs: str,
    modes: str,
    system_info: dict[str, str],
) -> str:
    rows = chart_rows_from_summary(summary_items, baseline_ref, current_ref)
    by_impl = {row["impl"]: row for row in rows}

    required = ("rust-original", "rust-current", "c-antirez", "second-state-mlx", "mlx-audio")
    for impl in required:
        if impl not in by_impl:
            raise SystemExit(f"Missing benchmark result for '{impl}'")

    # Build table sorted by median inference latency (fastest first).
    impl_labels = {
        "rust-original": ("qwen-asr (first)", baseline_ref),
        "rust-current": ("qwen-asr (latest)", current_ref),
        "c-antirez": ("pure C upstream", by_impl["c-antirez"].get("commit") or "-"),
        "second-state-mlx": ("second-state MLX GPU", by_impl["second-state-mlx"].get("commit") or "-"),
        "mlx-audio": ("mlx-audio Python MLX", by_impl["mlx-audio"].get("commit") or "-"),
    }
    table = [
        (impl_labels[row["impl"]][0], impl_labels[row["impl"]][1], row)
        for row in rows
    ]

    latest_ms = by_impl["rust-current"]["total_ms"]
    original_ms = by_impl["rust-original"]["total_ms"]
    c_ms = by_impl["c-antirez"]["total_ms"]
    second_state_ms = by_impl["second-state-mlx"]["total_ms"]
    mlx_audio_ms = by_impl["mlx-audio"]["total_ms"]

    unified_latency_rel = os.path.relpath(chart_dir / "benchmark-unified-latency.png", report_dir)
    unified_rtf_rel = os.path.relpath(chart_dir / "benchmark-unified-rtf.png", report_dir)

    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Offline benchmark on the same input WAV and model across five implementations.")
    lines.append(f"- qwen-asr first: `{baseline_ref}`.")
    lines.append(f"- qwen-asr latest: `{current_ref}`.")
    lines.append("- Upstream C: `antirez/qwen-asr`.")
    lines.append("- GPU baselines: `second-state/qwen3_asr_rs` MLX and `mlx-audio` Python MLX.")
    lines.append("- Implementations are benchmarked sequentially, not in parallel; each round is a standalone process invocation.")
    lines.append("- Primary metric is median inference time across standalone rounds for every implementation.")
    lines.append("- qwen-asr and pure C use their internal inference timers. MLX-based implementations are timed after model load with explicit GPU synchronization.")
    lines.append("- macOS Accelerate enabled for qwen-asr and pure C where applicable.")
    lines.append("- Wall-clock time is retained as a secondary metric.")
    lines.append(f"- Standalone rounds per target: `{runs}`.")
    lines.append(f"- Modes requested: `{modes}`.")
    lines.append("- Results in the table and charts are sorted by median inference latency (fastest leftmost).")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- CPU: `{system_info.get('cpu_brand', 'unknown')}`")
    lines.append(f"- Cores: `{system_info.get('physical_cores', 'unknown')} physical / {system_info.get('logical_cores', 'unknown')} logical`")
    lines.append(f"- Memory: `{system_info.get('memory_gb', 'unknown')} GB`")
    lines.append(f"- Machine arch: `{system_info.get('arch', 'unknown')}`")
    lines.append(f"- macOS: `{system_info.get('macos_version', 'unknown')}`")
    lines.append(f"- Rustc: `{system_info.get('rustc_version', 'unknown')}`")
    lines.append(f"- Model dir: `{model_dir}`")
    lines.append(f"- Input file: `{input_file}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Implementation | Commit | Median inference ms | Mean ms | Best ms | RTF |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, commit, row in table:
        commit_text = f"`{commit}`" if commit != "-" else "-"
        total_ms = row["total_ms"]
        rtf = row["realtime_factor"]
        total_text = fmt_ms(total_ms)
        mean_text = fmt_ms(row.get("inference_mean_ms"))
        best_text = fmt_ms(row.get("inference_best_ms"))
        lines.append(f"| {label} | {commit_text} | `{total_text}` | `{mean_text}` | `{best_text}` | `{rtf:.2f}x` |")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Wall-clock timing</summary>")
    lines.append("")
    lines.append("| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, commit, row in table:
        commit_text = f"`{commit}`" if commit != "-" else "-"
        wall_ms = row.get("wall_clock_ms")
        wall_rtf = row.get("wall_clock_realtime_factor")
        wall_text = fmt_ms(wall_ms)
        wall_mean_text = fmt_ms(row.get("wall_clock_mean_ms"))
        wall_best_text = fmt_ms(row.get("wall_clock_best_ms"))
        wall_rtf_text = "N/A" if wall_rtf is None else f"{wall_rtf:.2f}x"
        lines.append(f"| {label} | {commit_text} | `{wall_text}` | `{wall_mean_text}` | `{wall_best_text}` | `{wall_rtf_text}` |")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append(f"![Unified latency]({unified_latency_rel})")
    lines.append("")
    lines.append(f"![Unified realtime factor]({unified_rtf_rel})")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append(f"- qwen-asr latest `{current_ref}` is `{original_ms / latest_ms:.2f}x` the speed of qwen-asr first `{baseline_ref}`.")
    lines.append(f"- qwen-asr latest `{current_ref}` is `{c_ms / latest_ms:.2f}x` faster than the upstream pure C implementation.")
    lines.append(f"- qwen-asr latest `{current_ref}` is `{second_state_ms / latest_ms:.2f}x` faster than second-state MLX GPU by inference latency.")
    lines.append(f"- qwen-asr latest `{current_ref}` is `{mlx_audio_ms / latest_ms:.2f}x` faster than mlx-audio Python MLX by inference latency.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark report and charts.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--root-report", required=True)
    parser.add_argument("--charts-dir", required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--current-ref", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--system-info", default="")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    report_path = Path(args.report)
    root_report_path = Path(args.root_report)
    charts_dir = Path(args.charts_dir)

    summary_items = load_json(summary_path)
    system_info = load_system_info(Path(args.system_info) if args.system_info else None)

    rows = chart_rows_from_summary(summary_items, args.baseline_ref, args.current_ref)
    impls = {row["impl"] for row in rows}
    missing = {"rust-original", "rust-current", "c-antirez", "second-state-mlx", "mlx-audio"} - impls
    if missing:
        raise SystemExit(f"Missing benchmark results for: {', '.join(sorted(missing))}. Got: {', '.join(sorted(impls))}")

    render_bar_chart(
        rows,
        "total_ms",
        "Latency (ms)",
        "Offline ASR Benchmark on macOS",
        "Median inference time, lower is better",
        charts_dir / "benchmark-unified-latency.png",
    )
    render_bar_chart(
        rows,
        "realtime_factor",
        "Realtime Factor (x)",
        "Offline ASR Benchmark on macOS",
        "Higher is better",
        charts_dir / "benchmark-unified-rtf.png",
    )

    # Keep the docs copy of the charts in sync so comparison.md can use local paths.
    docs_charts_dir = Path("docs/benchmarks/charts")
    docs_charts_dir.mkdir(parents=True, exist_ok=True)
    for chart_name in ("benchmark-unified-latency.png", "benchmark-unified-rtf.png"):
        shutil.copy2(charts_dir / chart_name, docs_charts_dir / chart_name)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = build_markdown_report(
        report_path.parent,
        charts_dir,
        summary_items,
        args.baseline_ref,
        args.current_ref,
        args.model_dir,
        args.input_file,
        args.runs,
        args.modes,
        system_info,
    )
    report_path.write_text(report_content, encoding="utf-8")
    root_report_content = build_markdown_report(
        root_report_path.parent,
        charts_dir,
        summary_items,
        args.baseline_ref,
        args.current_ref,
        args.model_dir,
        args.input_file,
        args.runs,
        args.modes,
        system_info,
    )
    root_report_path.write_text(root_report_content, encoding="utf-8")


if __name__ == "__main__":
    main()
