#!/usr/bin/env python3
"""Oscillator-mode analyzer for optical-trap tidy DataFrames.

For each oscillator cycle reports:
* Period and individual ramp durations.
* Voltage and extension at the top and bottom of each ramp.
* Extension vs voltage for ``pull_out`` and ``pull_in`` phases.
* Mean and spread of extension and force during each ``hold``.
* Peak trap-1 force during each cycle.

Saves a summary CSV and a multi-panel figure per run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optical_trap_tidy import (
    load_tidy,
    find_routine_segments,
    label_cycles,
    apply_valid_mask,
    compute_force_reference,
    _expand_paths,
)


# ---------------------------------------------------------------------------
# Per-cycle statistics
# ---------------------------------------------------------------------------


def _cycle_stats(df_cycle: pd.DataFrame) -> dict[str, Any]:
    """Compute statistics for a single oscillator cycle."""
    stats: dict[str, Any] = {}

    # Drop invalid samples (edge spikes, glitches) before any statistics
    if "valid" in df_cycle.columns:
        df_cycle = df_cycle[df_cycle["valid"].astype(bool)]

    # sub-phase frames
    pull_out = df_cycle[df_cycle["phase"] == "pull_out"]
    pull_in = df_cycle[df_cycle["phase"] == "pull_in"]
    holds = df_cycle[df_cycle["phase"] == "hold"]

    # timing
    t = df_cycle["time_s"].to_numpy(dtype=float)
    stats["start_time_s"] = float(t[0]) if len(t) else float("nan")
    stats["end_time_s"] = float(t[-1]) if len(t) else float("nan")
    stats["period_s"] = stats["end_time_s"] - stats["start_time_s"]

    # ramp durations
    if len(pull_out):
        t_po = pull_out["time_s"].to_numpy(dtype=float)
        stats["pull_out_duration_s"] = float(t_po[-1] - t_po[0])
    else:
        stats["pull_out_duration_s"] = float("nan")

    if len(pull_in):
        t_pi = pull_in["time_s"].to_numpy(dtype=float)
        stats["pull_in_duration_s"] = float(t_pi[-1] - t_pi[0])
    else:
        stats["pull_in_duration_s"] = float("nan")

    # voltage and extension at ramp extremes
    for col_suffix, phase_df in [("pull_out", pull_out), ("pull_in", pull_in)]:
        if len(phase_df):
            stats[f"voltage_start_{col_suffix}_v"] = float(phase_df["voltage_v"].iloc[0])
            stats[f"voltage_end_{col_suffix}_v"] = float(phase_df["voltage_v"].iloc[-1])
            stats[f"extension_start_{col_suffix}_nm"] = float(phase_df["extension_nm"].iloc[0])
            stats[f"extension_end_{col_suffix}_nm"] = float(phase_df["extension_nm"].iloc[-1])
        else:
            for k in (
                f"voltage_start_{col_suffix}_v",
                f"voltage_end_{col_suffix}_v",
                f"extension_start_{col_suffix}_nm",
                f"extension_end_{col_suffix}_nm",
            ):
                stats[k] = float("nan")

    # hold statistics
    hold_groups = []
    if len(holds):
        # split into contiguous hold runs
        h_idx = holds.index.to_numpy()
        breaks = np.where(np.diff(h_idx) > 1)[0] + 1
        hold_groups = [holds.loc[grp] for grp in np.split(h_idx, breaks)]

    for i, hg in enumerate(hold_groups[:2]):  # at most 2 holds per cycle
        ext_h = hg["extension_nm"].to_numpy(dtype=float)
        force_h = hg["force_pn"].to_numpy(dtype=float)
        v_h = hg["voltage_v"].to_numpy(dtype=float)
        sp = hg["setpoint_v"].to_numpy(dtype=float)
        suffix = f"hold{i + 1}"
        stats[f"{suffix}_setpoint_v"] = float(np.nanmean(sp)) if len(sp) else float("nan")
        stats[f"{suffix}_extension_mean_nm"] = float(np.nanmean(ext_h)) if len(ext_h) else float("nan")
        stats[f"{suffix}_extension_std_nm"] = float(np.nanstd(ext_h)) if len(ext_h) else float("nan")
        finite_force = force_h[np.isfinite(force_h)]
        stats[f"{suffix}_force_mean_pn"] = float(np.mean(finite_force)) if len(finite_force) else float("nan")
        stats[f"{suffix}_force_std_pn"] = float(np.std(finite_force)) if len(finite_force) else float("nan")

    # peak trap-1 force
    force_all = df_cycle["force_pn"].to_numpy(dtype=float)
    finite = force_all[np.isfinite(force_all)]
    stats["peak_force_pn"] = float(np.max(np.abs(finite))) if len(finite) else float("nan")

    # cycle completeness metadata
    if "cycle_complete" in df_cycle.columns:
        comp = df_cycle["cycle_complete"].dropna()
        stats["cycle_complete"] = bool(comp.astype(bool).all()) if len(comp) else False
    if "cycle_status" in df_cycle.columns:
        st = df_cycle["cycle_status"].dropna()
        stats["cycle_status"] = str(st.iloc[-1]) if len(st) else "unknown"
    if "cycle_partial_reason" in df_cycle.columns:
        reason = df_cycle["cycle_partial_reason"].dropna()
        stats["cycle_partial_reason"] = str(reason.iloc[-1]) if len(reason) else ""
    if "cycle_last_phase" in df_cycle.columns:
        lp = df_cycle["cycle_last_phase"].dropna()
        stats["cycle_last_phase"] = str(lp.iloc[-1]) if len(lp) else ""
    if "cycle_next_state" in df_cycle.columns:
        ns = df_cycle["cycle_next_state"].dropna()
        stats["cycle_next_state"] = int(ns.iloc[-1]) if len(ns) else pd.NA

    return stats


def analyze_oscillator(
    df: pd.DataFrame,
    source: str | None = None,
) -> pd.DataFrame:
    """Compute per-cycle statistics for oscillator mode.

    Parameters
    ----------
    df:
        Tidy DataFrame (after :func:`find_routine_segments` and
        :func:`label_cycles`).
    source:
        Run identifier; defaults to the ``source`` column value.

    Returns
    -------
    DataFrame with one row per cycle.
    """
    if source is None:
        source = str(df["source"].iloc[0]) if "source" in df.columns else "unknown"

    osc = df[df["mode"] == "oscillator"].copy()
    if osc.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    cycle_ids = osc["cycle"].dropna().unique()
    for cid in sorted(cycle_ids):
        df_cycle = osc[osc["cycle"] == cid]
        stats = _cycle_stats(df_cycle)
        stats["source"] = source
        stats["cycle"] = int(cid)
        records.append(stats)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_oscillator(df: pd.DataFrame, stats: pd.DataFrame) -> plt.Figure:
    """Return a multi-panel figure showing the oscillator analysis.

    Parameters
    ----------
    df:
        Tidy DataFrame (oscillator rows).
    stats:
        Per-cycle statistics as returned by :func:`analyze_oscillator`.
    """
    osc = df[df["mode"] == "oscillator"]
    # Drop invalid samples (edge spikes, glitches) before plotting
    if "valid" in osc.columns:
        osc = osc[osc["valid"].astype(bool)]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Panel 1: extension vs time, coloured by phase
    ax = axes[0]
    phase_colours = {"pull_out": "tab:blue", "pull_in": "tab:orange", "hold": "tab:green"}
    for phase_name, colour in phase_colours.items():
        sub = osc[osc["phase"] == phase_name]
        if not sub.empty:
            ax.scatter(
                sub["time_s"],
                sub["extension_nm"],
                s=0.5,
                c=colour,
                label=phase_name,
                rasterized=True,
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Extension (nm)")
    ax.set_title("Extension vs time (oscillator cycles)")
    ax.legend(markerscale=8, loc="upper right")

    # Panel 2: extension vs voltage for pull_out and pull_in
    ax = axes[1]
    for phase_name, colour in [("pull_out", "tab:blue"), ("pull_in", "tab:orange")]:
        sub = osc[osc["phase"] == phase_name]
        if not sub.empty:
            ax.scatter(
                sub["voltage_v"],
                sub["extension_nm"],
                s=0.5,
                c=colour,
                label=phase_name,
                alpha=0.5,
                rasterized=True,
            )
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Extension (nm)")
    ax.set_title("Extension vs voltage (pull phases)")
    ax.legend(markerscale=8, loc="upper right")

    # Panel 3: per-cycle period
    ax = axes[2]
    if not stats.empty and "period_s" in stats.columns:
        ax.bar(stats["cycle"], stats["period_s"], color="tab:purple")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Period (s)")
        ax.set_title("Oscillator cycle period")

    source = str(df["source"].iloc[0]) if "source" in df.columns else ""
    fig.suptitle(f"Oscillator analysis — {source}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Captions below figures (in figure text)
    caption = (
        "Top: extension coloured by phase (blue=pull_out, orange=pull_in, "
        "green=hold).  Middle: extension vs voltage for ramp phases.  "
        "Bottom: per-cycle period."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8, wrap=True)

    return fig


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze oscillator cycles in tidy optical-trap data.")
    parser.add_argument(
        "files",
        nargs="+",
        help=(
            "One or more .dat run files to analyze, or a single folder "
            "(top-level .dat files only; _Voltage.dat and _params.dat skipped)."
        ),
    )
    parser.add_argument("--k-per-mw", type=float, default=None, help="Trap-1 stiffness per mW (pN/nm/mW).")
    parser.add_argument("--cal-file", default=None, help="Path to JSON/YAML calibration file.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write per-run CSV and PNG outputs.",
    )
    parser.add_argument(
        "--dv-dt-threshold",
        type=float,
        default=0.0003,
        help="dV/dt threshold (V/sample) used to classify stalled oscillator ramps as non-moving, while still preserving the cycle.",
    )
    parser.add_argument(
        "--force-ref",
        choices=["high_hold_tail", "fixed"],
        default="high_hold_tail",
        help="Force reference method for oscillator mode (default: high_hold_tail).",
    )
    parser.add_argument(
        "--force-ref-frames",
        type=int,
        default=100,
        help="Number of tail frames for --force-ref high_hold_tail.",
    )
    parser.add_argument(
        "--force-ref-nm",
        type=float,
        default=None,
        help="Fixed reference position for --force-ref fixed (nm).",
    )
    parser.add_argument(
        "--edge-correction",
        action="store_true",
        default=False,
        help="Apply edge-tracking correction before analysis (default: off).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    fpaths = _expand_paths(args.files)
    if not fpaths:
        print("Error: no input files found.", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fpath in fpaths:
        try:
            df = load_tidy(
                fpath,
                k_per_mw=args.k_per_mw,
                cal_file=args.cal_file,
                edge_correction=args.edge_correction,
            )
            df = find_routine_segments(df, mode="oscillator")
            df = label_cycles(df, dv_dt_threshold=args.dv_dt_threshold)
            df = apply_valid_mask(df)

            # Compute force if stiffness is available
            if not df["stiffness_pn_per_nm"].isna().all():
                df = compute_force_reference(
                    df,
                    method=args.force_ref,
                    force_ref_nm=args.force_ref_nm,
                    force_ref_frames=args.force_ref_frames,
                )

            source = df["source"].iloc[0]
            stats = analyze_oscillator(df, source=source)

            if stats.empty:
                print(f"{source}: no oscillator cycles detected.")
                continue

            csv_path = out_dir / f"{source}_oscillator.csv"
            stats.to_csv(csv_path, index=False)
            complete_count = int((stats["cycle_complete"].astype(bool)).sum()) if "cycle_complete" in stats.columns else len(stats)
            partial_count = int((~stats["cycle_complete"].astype(bool)).sum()) if "cycle_complete" in stats.columns else 0
            print(f"{source}: {complete_count} complete cycle(s), {partial_count} partial cycle(s). CSV → {csv_path}")

            fig = plot_oscillator(df, stats)
            fig_path = out_dir / f"{source}_oscillator.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"{source}: figure → {fig_path}")

        except Exception as exc:
            print(f"Error processing {fpath}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
