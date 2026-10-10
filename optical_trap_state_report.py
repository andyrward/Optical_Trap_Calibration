#!/usr/bin/env python3
"""CLI utility for summarizing contiguous state runs in a trap trace."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

if not sys.platform.startswith("win") and not sys.stdout.isatty():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from optical_trap_toolbox import (
    StateRun,
    build_state_report,
    export_state_report_csv,
    export_state_report_json,
    load_normalized_trace,
    plot_state_vs_time,
)


def format_state_report(report: dict) -> str:
    summary = report.get("summary", [])
    lines: list[str] = ["State report"]
    if not summary:
        lines.append("No state runs detected.")
        return "\n".join(lines)

    lines.append(
        f"{'state_code':>10} {'label':<28} {'conf':<8} {'runs':>4} {'samples':>8} {'dur(s)':>9} {'frac':>7}"
    )
    for item in summary:
        lines.append(
            f"{item['state_code']:>10} {item['label']:<28} {item['confidence']:<8} "
            f"{item['run_count']:>4} {item['total_samples']:>8} {item['total_duration_s']:>9.3f} {item['fraction_of_trace']:>7.3f}"
        )

    lines.append("")
    lines.append("Run boundaries:")
    for run in report.get("state_runs", []):
        start_idx = run["start_idx"]
        end_idx = run["end_idx"]
        state_code = run["state_code"]
        start_time = run.get("start_time_s")
        end_time = run.get("end_time_s")
        if start_time is None:
            time_desc = f"idx {start_idx}-{end_idx}"
        else:
            time_desc = f"idx {start_idx}-{end_idx}; time {start_time:.3f}s to {end_time:.3f}s"
        lines.append(f"  state {state_code}: {time_desc}")
    return "\n".join(lines)


def _save_plot(trace: dict, report: dict, path: str | Path):
    runs = [StateRun(int(item["start_idx"]), int(item["end_idx"]), int(item["state_code"])) for item in report["state_runs"]]
    fig = plot_state_vs_time(trace, runs)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize contiguous state runs in an optical trap trace.")
    parser.add_argument("--file", required=True, help="Path to a trace file to load and summarize.")
    parser.add_argument("--csv", help="Optional CSV output path for per-state summary.")
    parser.add_argument("--json", help="Optional JSON output path for the full report.")
    parser.add_argument(
        "--plot",
        nargs="?",
        const="state_report.png",
        default=None,
        help="Optional path to save a state-vs-time plot. Defaults to 'state_report.png' when enabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        trace = load_normalized_trace(args.file)
        report = build_state_report(trace)
        print(format_state_report(report))
        if args.csv:
            export_state_report_csv(report["summary"], args.csv)
            print(f"CSV export written to {args.csv}")
        if args.json:
            export_state_report_json(report["summary"], args.json)
            print(f"JSON export written to {args.json}")
        if args.plot:
            _save_plot(trace, report, args.plot)
            print(f"Plot written to {args.plot}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
