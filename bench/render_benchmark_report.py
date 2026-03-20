#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "rust-before": "#4C78A8",
    "rust-current": "#F58518",
    "c-antirez": "#54A24B",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_check_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


def pick_result(items: list[dict], impl: str, accelerate: bool, mode: str = "offline") -> dict | None:
    for item in items:
        if item.get("impl") == impl and item.get("accelerate") == accelerate and item.get("mode") == mode:
            return item
    return None


def chart_rows_from_summary(items: list[dict], baseline_ref: str, current_ref: str) -> list[dict]:
    mapping = [
        ("rust-before", f"before auto research\n{baseline_ref}"),
        ("rust-current", f"after auto research\n{current_ref}"),
        ("c-antirez", "pure C\nupstream"),
    ]
    rows = []
    for impl, label in mapping:
        item = pick_result(items, impl, True, "offline")
        if not item or not item.get("run_ok"):
            continue
        rows.append(
            {
                "impl": impl,
                "label": label,
                "total_ms": float(item["total_ms"]),
                "realtime_factor": float(item["realtime_factor"]),
                "commit": item.get("commit"),
            }
        )
    return rows


def nice_upper_bound(values: list[float]) -> float:
    vmax = max(values)
    if vmax <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(vmax))
    scaled = vmax / magnitude
    if scaled <= 1.5:
        nice = 2
    elif scaled <= 3:
        nice = 4
    elif scaled <= 7:
        nice = 8
    else:
        nice = 10
    return nice * magnitude


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
    fig.savefig(output_path, bbox_inches="tight")
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
) -> str:
    accel_rows = chart_rows_from_summary(summary_items, baseline_ref, current_ref)
    accel_by_impl = {row["impl"]: row for row in accel_rows}

    for impl in ("rust-before", "rust-current", "c-antirez"):
        if impl not in accel_by_impl:
            raise SystemExit(f"Missing Accelerate result for '{impl}' — expected all three implementations to succeed")

    accel_before = accel_by_impl["rust-before"]
    accel_current = accel_by_impl["rust-current"]
    accel_c = accel_by_impl["c-antirez"]

    accel_table = [
        ("before auto research", baseline_ref, accel_before["total_ms"], accel_before["realtime_factor"]),
        ("after auto research", current_ref, accel_current["total_ms"], accel_current["realtime_factor"]),
        ("pure C upstream", "-", accel_c["total_ms"], accel_c["realtime_factor"]),
    ]

    before_ms = accel_before["total_ms"]
    after_ms = accel_current["total_ms"]
    c_ms = accel_c["total_ms"]
    speedup_vs_before = before_ms / after_ms
    speedup_vs_c = c_ms / after_ms

    accel_latency_rel = os.path.relpath(chart_dir / "benchmark-accelerate-latency.png", report_dir)
    accel_rtf_rel = os.path.relpath(chart_dir / "benchmark-accelerate-rtf.png", report_dir)

    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Offline benchmark on the same input WAV and model across three implementations.")
    lines.append(f"- Rust baseline: `before auto research {baseline_ref}`.")
    lines.append(f"- Rust optimized: `after auto research {current_ref}`.")
    lines.append("- Upstream baseline: `antirez/qwen-asr` pure C implementation.")
    lines.append("- macOS Accelerate enabled.")
    lines.append(f"- Runs per target: `{runs}`.")
    lines.append(f"- Modes requested: `{modes}`.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Machine arch: `{safe_check_output(['uname', '-m'])}`")
    lines.append(f"- macOS: `{safe_check_output(['sw_vers', '-productVersion'])}`")
    lines.append(f"- Model dir: `{model_dir}`")
    lines.append(f"- Input file: `{input_file}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Implementation | Commit | Total ms | RTF |")
    lines.append("|---|---:|---:|---:|")
    for label, commit, total_ms, rtf in accel_table:
        commit_text = f"`{commit}`" if commit != "-" else "-"
        total_text = f"{total_ms:,.0f}" if total_ms >= 100 else f"{total_ms:.2f}"
        lines.append(f"| {label} | {commit_text} | `{total_text}` | `{rtf:.2f}x` |")
    lines.append("")
    lines.append(f"![Accelerate latency]({accel_latency_rel})")
    lines.append("")
    lines.append(f"![Accelerate realtime factor]({accel_rtf_rel})")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append(f"- With Accelerate enabled, `after auto research {current_ref}` is `{speedup_vs_before:.2f}x` faster than `before auto research {baseline_ref}`.")
    lines.append(f"- With Accelerate enabled, `after auto research {current_ref}` is `{speedup_vs_c:.2f}x` faster than the upstream pure C implementation.")
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
    args = parser.parse_args()

    summary_path = Path(args.summary)
    report_path = Path(args.report)
    root_report_path = Path(args.root_report)
    charts_dir = Path(args.charts_dir)

    summary_items = load_json(summary_path)

    accel_rows = chart_rows_from_summary(summary_items, args.baseline_ref, args.current_ref)
    accel_impls = {row["impl"] for row in accel_rows}
    missing = {"rust-before", "rust-current", "c-antirez"} - accel_impls
    if missing:
        raise SystemExit(f"Missing Accelerate results for: {', '.join(sorted(missing))}. Got: {', '.join(sorted(accel_impls))}")

    render_bar_chart(
        accel_rows,
        "total_ms",
        "Latency (ms)",
        "Offline ASR Benchmark on macOS",
        "Accelerate enabled, lower is better",
        charts_dir / "benchmark-accelerate-latency.png",
    )
    render_bar_chart(
        accel_rows,
        "realtime_factor",
        "Realtime Factor (x)",
        "Offline ASR Benchmark on macOS",
        "Accelerate enabled, higher is better",
        charts_dir / "benchmark-accelerate-rtf.png",
    )

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
    )
    root_report_path.write_text(root_report_content, encoding="utf-8")


if __name__ == "__main__":
    main()
