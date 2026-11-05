"""Post-tidy rupture and Δ-extension analysis helpers.

This module is an explicit analysis layer that consumes tidy per-frame tables
from either oscillator (`load_tidy` + cycle labelling) or puppet (`read_puppet`)
workflows. Loaders stay side-effect-free.

Expected tidy input columns
---------------------------
Required by all routines:
* ``time_s``
* ``phase`` (``pull_out``, ``pull_in``, ``hold``)
* ``force_pn``
* ``extension_nm``

Optional but used when available:
* ``valid`` (invalid frames are excluded by default)
* ``cycle`` (oscillator cycle id)
* ``segment_id`` (puppet contiguous segment id)
* ``voltage_v`` / ``hold_voltage``
* ``source`` / ``mode``

Outputs
-------
Dynamic/static rupture detectors return a shared event schema with columns:
``source``, ``mode``, ``regime``, ``event_type``, ``group_id``,
``analysis_cycle``, ``rupture_time_s``, ``rupture_force_pn``,
``pre_rupture_extension_nm``, ``loading_rate_pn_per_s``, ``lifetime_s``,
``hold_voltage_v``, ``group_col``, ``hold_group_col``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "source",
    "mode",
    "regime",
    "event_type",
    "group_id",
    "analysis_cycle",
    "rupture_time_s",
    "rupture_force_pn",
    "pre_rupture_extension_nm",
    "loading_rate_pn_per_s",
    "lifetime_s",
    "hold_voltage_v",
    "group_col",
    "hold_group_col",
]


def _event_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[EVENT_COLUMNS]


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def infer_analysis_cycle(df: pd.DataFrame) -> pd.Series:
    """Return a cycle-like ID shared by pull_out/pull_in rows.

    Preference order:
    1) Existing ``cycle`` column if populated.
    2) Inferred from pull_out segment starts (works for puppet tidy tables).
    """
    if "cycle" in df.columns and df["cycle"].notna().any():
        return df["cycle"].astype("Int64")

    phase = df["phase"].astype(object).to_numpy()
    cyc = np.full(len(df), pd.NA, dtype=object)
    current = 0
    prev = None
    for i, p in enumerate(phase):
        if p == "pull_out" and prev != "pull_out":
            current += 1
        if current > 0 and p in {"pull_out", "pull_in", "hold"}:
            cyc[i] = current
        prev = p
    return pd.array(cyc, dtype="Int64")


def _find_rupture_index(force: np.ndarray, idx: np.ndarray, threshold_pn: float, n_samples: int) -> int | None:
    """Return absolute index of rupture point in idx, or None."""
    pull_force = force[idx]
    for j in range(len(idx) - 1):
        window_end = min(len(idx), j + n_samples + 1)
        window = pull_force[j:window_end]
        finite = np.isfinite(window)
        if finite.sum() < 2:
            continue
        peak = float(np.nanmax(window[finite]))
        tail = float(window[finite][-1])
        if (peak - tail) >= threshold_pn:
            peak_rel = int(np.flatnonzero(finite & (window == peak))[0])
            return int(idx[j + peak_rel])
    return None


def detect_dynamic_rupture_events(
    df: pd.DataFrame,
    *,
    threshold_pn: float = 5.0,
    n_samples: int = 100,
    cycle_col: str | None = None,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Detect rupture on pull_out trajectories and estimate pre-rupture loading rate."""
    _require_columns(df, ["time_s", "phase", "force_pn", "extension_nm"])
    work = df.copy()
    work["analysis_cycle"] = infer_analysis_cycle(work) if cycle_col is None else work[cycle_col].astype("Int64")
    work_valid = work["valid"].astype(bool) if ("valid" in work.columns and valid_only) else pd.Series(True, index=work.index)

    t = work["time_s"].to_numpy(dtype=float)
    f = work["force_pn"].to_numpy(dtype=float)
    x = work["extension_nm"].to_numpy(dtype=float)
    phase = work["phase"].astype(object).to_numpy()

    records: list[dict[str, Any]] = []
    for cyc in sorted(work["analysis_cycle"].dropna().astype(int).unique()):
        cyc_mask = work["analysis_cycle"].eq(cyc).to_numpy() & (phase == "pull_out") & work_valid.to_numpy()
        idx = np.where(cyc_mask)[0]
        if len(idx) < 2:
            continue
        rupture_i = _find_rupture_index(f, idx, threshold_pn=threshold_pn, n_samples=n_samples)
        if rupture_i is None:
            continue

        pre_mask = idx[idx <= rupture_i]
        pre_mask = pre_mask[np.isfinite(f[pre_mask]) & np.isfinite(t[pre_mask])]
        loading_rate = np.nan
        if len(pre_mask) >= 2:
            loading_rate = float(np.polyfit(t[pre_mask], f[pre_mask], 1)[0])

        records.append(
            {
                "source": work["source"].iloc[0] if "source" in work.columns else "",
                "mode": work["mode"].iloc[0] if "mode" in work.columns else "",
                "regime": "dynamic",
                "event_type": "rupture",
                "group_id": int(cyc),
                "analysis_cycle": int(cyc),
                "rupture_time_s": float(t[rupture_i]),
                "rupture_force_pn": float(f[rupture_i]),
                "pre_rupture_extension_nm": float(x[rupture_i]),
                "loading_rate_pn_per_s": loading_rate,
                "lifetime_s": np.nan,
                "hold_voltage_v": np.nan,
                "group_col": cycle_col or "analysis_cycle",
                "hold_group_col": np.nan,
            }
        )
    return _event_frame(records)


def _default_hold_group(df: pd.DataFrame) -> pd.Series:
    if "segment_id" in df.columns and df["segment_id"].notna().any():
        return df["segment_id"].astype("Int64")
    hold = df["phase"].eq("hold").to_numpy()
    starts = np.zeros(len(df), dtype=np.int64)
    starts[0] = 1 if hold[0] else 0
    starts[1:] = (hold[1:] & ~hold[:-1]).astype(np.int64)
    groups = np.cumsum(starts)
    out = pd.array([g if h else pd.NA for g, h in zip(groups, hold)], dtype="Int64")
    return pd.Series(out, index=df.index)


def detect_static_hold_rupture_events(
    df: pd.DataFrame,
    *,
    threshold_pn: float = 5.0,
    n_samples: int = 100,
    hold_group_col: str | None = None,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Detect rupture during hold phases and report hold lifetime."""
    _require_columns(df, ["time_s", "phase", "force_pn", "extension_nm"])
    work = df.copy()
    work["analysis_cycle"] = infer_analysis_cycle(work)
    work["_hold_group"] = _default_hold_group(work) if hold_group_col is None else work[hold_group_col].astype("Int64")
    work_valid = work["valid"].astype(bool) if ("valid" in work.columns and valid_only) else pd.Series(True, index=work.index)

    t = work["time_s"].to_numpy(dtype=float)
    f = work["force_pn"].to_numpy(dtype=float)
    x = work["extension_nm"].to_numpy(dtype=float)
    hold = work["phase"].eq("hold").to_numpy()

    records: list[dict[str, Any]] = []
    for gid in sorted(work["_hold_group"].dropna().astype(int).unique()):
        hold_mask = hold & work["_hold_group"].eq(gid).to_numpy() & work_valid.to_numpy()
        idx = np.where(hold_mask)[0]
        if len(idx) < 2:
            continue
        rupture_i = _find_rupture_index(f, idx, threshold_pn=threshold_pn, n_samples=n_samples)
        if rupture_i is None:
            continue
        hold_voltage = np.nan
        if "hold_voltage" in work.columns and np.isfinite(work.loc[idx, "hold_voltage"]).any():
            hold_voltage = float(work.loc[idx, "hold_voltage"].dropna().mean())
        elif "voltage_v" in work.columns and np.isfinite(work.loc[idx, "voltage_v"]).any():
            hold_voltage = float(work.loc[idx, "voltage_v"].dropna().mean())

        records.append(
            {
                "source": work["source"].iloc[0] if "source" in work.columns else "",
                "mode": work["mode"].iloc[0] if "mode" in work.columns else "",
                "regime": "static",
                "event_type": "hold_rupture",
                "group_id": int(gid),
                "analysis_cycle": int(work.loc[rupture_i, "analysis_cycle"]) if pd.notna(work.loc[rupture_i, "analysis_cycle"]) else np.nan,
                "rupture_time_s": float(t[rupture_i]),
                "rupture_force_pn": float(f[rupture_i]),
                "pre_rupture_extension_nm": float(x[rupture_i]),
                "loading_rate_pn_per_s": np.nan,
                "lifetime_s": float(t[rupture_i] - t[idx[0]]),
                "hold_voltage_v": hold_voltage,
                "group_col": "analysis_cycle",
                "hold_group_col": hold_group_col or "_hold_group",
            }
        )
    return _event_frame(records)


def _interpolate_extension_at_force(force: np.ndarray, ext: np.ndarray, target_force: float) -> float:
    finite = np.isfinite(force) & np.isfinite(ext)
    if finite.sum() < 2:
        return np.nan
    f = force[finite]
    x = ext[finite]
    order = np.argsort(f)
    f = f[order]
    x = x[order]
    uniq_f, uniq_idx = np.unique(f, return_index=True)
    uniq_x = x[uniq_idx]
    if len(uniq_f) < 2 or target_force < uniq_f.min() or target_force > uniq_f.max():
        return np.nan
    return float(np.interp(target_force, uniq_f, uniq_x))


def add_same_force_extension(
    events: pd.DataFrame,
    df: pd.DataFrame,
    *,
    phase: str = "pull_in",
    group_col: str = "analysis_cycle",
    voltage_v: float | None = None,
    voltage_tolerance_v: float = 0.05,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Add same-force interpolated extension from a later phase (default pull_in)."""
    if len(events) == 0:
        out = events.copy()
        out["same_force_extension_nm"] = pd.Series(dtype=float)
        out["delta_extension_nm"] = pd.Series(dtype=float)
        out["same_force_phase"] = pd.Series(dtype=object)
        return out

    _require_columns(df, ["phase", "force_pn", "extension_nm"])
    work = df.copy()
    if group_col == "analysis_cycle" and "analysis_cycle" not in work.columns:
        work["analysis_cycle"] = infer_analysis_cycle(work)

    out = events.copy()
    same_force_vals: list[float] = []
    for _, ev in out.iterrows():
        if group_col not in ev or pd.isna(ev[group_col]):
            same_force_vals.append(np.nan)
            continue
        mask = work[group_col].eq(ev[group_col]) & work["phase"].eq(phase)
        if "valid" in work.columns and valid_only:
            mask &= work["valid"].astype(bool)
        if voltage_v is not None and "voltage_v" in work.columns:
            mask &= (work["voltage_v"] - voltage_v).abs() <= float(voltage_tolerance_v)
        seg = work.loc[mask]
        if len(seg) < 2:
            same_force_vals.append(np.nan)
            continue
        same_force_vals.append(
            _interpolate_extension_at_force(
                seg["force_pn"].to_numpy(dtype=float),
                seg["extension_nm"].to_numpy(dtype=float),
                float(ev["rupture_force_pn"]),
            )
        )

    out["same_force_extension_nm"] = same_force_vals
    out["delta_extension_nm"] = out["pre_rupture_extension_nm"] - out["same_force_extension_nm"]
    out["same_force_phase"] = phase
    return out


def _adaptive_force_bins(force_values: np.ndarray, target_count: int) -> np.ndarray:
    if len(force_values) == 0:
        return np.array([], dtype=float)
    sorted_force = np.sort(force_values)
    target = max(1, int(target_count))
    edges = [sorted_force[0]]
    for start in range(target, len(sorted_force), target):
        edges.append(sorted_force[start - 1])
    edges.append(sorted_force[-1])
    bins = np.array(sorted(set(float(v) for v in edges)), dtype=float)
    if len(bins) == 1:
        bins = np.array([bins[0], bins[0] + 1e-9], dtype=float)
    return bins


def summarize_dynamic_delta_extension(
    df: pd.DataFrame,
    *,
    force_bin_target_count: int = 400,
    cycle_col: str | None = None,
    force_min_pn: float | None = None,
    force_max_pn: float | None = None,
    valid_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Δ-extension in adaptive force bins (pull_out − pull_in)."""
    _require_columns(df, ["phase", "force_pn", "extension_nm"])
    work = df.copy()
    work["analysis_cycle"] = infer_analysis_cycle(work) if cycle_col is None else work[cycle_col].astype("Int64")
    if "valid" in work.columns and valid_only:
        work = work[work["valid"].astype(bool)]
    work = work[work["phase"].isin(["pull_out", "pull_in"])].copy()
    work = work[np.isfinite(work["force_pn"]) & np.isfinite(work["extension_nm"])]
    if work.empty:
        empty_cycle = pd.DataFrame(columns=[
            "analysis_cycle",
            "force_bin",
            "force_bin_low_pn",
            "force_bin_high_pn",
            "extension_pull_out_nm",
            "extension_pull_in_nm",
            "delta_extension_nm",
            "mean_force_pn",
        ])
        empty_summary = pd.DataFrame(columns=[
            "force_bin",
            "force_bin_low_pn",
            "force_bin_high_pn",
            "n_cycles",
            "mean_force_pn",
            "delta_extension_mean_nm",
            "delta_extension_sd_nm",
        ])
        return empty_cycle, empty_summary

    pull_out = work[work["phase"] == "pull_out"].copy()
    if force_min_pn is None:
        force_min_pn = float(np.nanmin(pull_out["force_pn"]))
    if force_max_pn is None:
        force_max_pn = float(np.nanmax(pull_out["force_pn"]))
    work = work[(work["force_pn"] >= force_min_pn) & (work["force_pn"] <= force_max_pn)].copy()
    pull_out = pull_out[(pull_out["force_pn"] >= force_min_pn) & (pull_out["force_pn"] <= force_max_pn)]
    if pull_out.empty:
        return summarize_dynamic_delta_extension(
            pd.DataFrame(columns=df.columns),
            force_bin_target_count=force_bin_target_count,
            cycle_col=cycle_col,
            force_min_pn=force_min_pn,
            force_max_pn=force_max_pn,
            valid_only=valid_only,
        )

    bins = _adaptive_force_bins(pull_out["force_pn"].to_numpy(dtype=float), force_bin_target_count)
    work["force_bin"] = pd.cut(
        work["force_pn"],
        bins=bins,
        include_lowest=True,
        right=True,
        labels=False,
    )
    work = work[work["force_bin"].notna()].copy()
    work["force_bin"] = work["force_bin"].astype(int)
    if work.empty:
        return summarize_dynamic_delta_extension(
            pd.DataFrame(columns=df.columns),
            force_bin_target_count=force_bin_target_count,
            cycle_col=cycle_col,
            force_min_pn=force_min_pn,
            force_max_pn=force_max_pn,
            valid_only=valid_only,
        )

    phase_means = (
        work.groupby(["analysis_cycle", "force_bin", "phase"], dropna=True)["extension_nm"]
        .mean()
        .unstack("phase")
        .reset_index()
    )
    if "pull_out" not in phase_means.columns:
        phase_means["pull_out"] = np.nan
    if "pull_in" not in phase_means.columns:
        phase_means["pull_in"] = np.nan
    phase_means = phase_means.dropna(subset=["pull_out", "pull_in"])
    if phase_means.empty:
        return summarize_dynamic_delta_extension(
            pd.DataFrame(columns=df.columns),
            force_bin_target_count=force_bin_target_count,
            cycle_col=cycle_col,
            force_min_pn=force_min_pn,
            force_max_pn=force_max_pn,
            valid_only=valid_only,
        )

    force_means = (
        work.groupby(["analysis_cycle", "force_bin"], dropna=True)["force_pn"]
        .mean()
        .rename("mean_force_pn")
        .reset_index()
    )
    per_cycle = phase_means.merge(force_means, on=["analysis_cycle", "force_bin"], how="left")
    per_cycle["delta_extension_nm"] = per_cycle["pull_out"] - per_cycle["pull_in"]
    per_cycle = per_cycle.rename(
        columns={
            "pull_out": "extension_pull_out_nm",
            "pull_in": "extension_pull_in_nm",
        }
    )
    per_cycle["force_bin_low_pn"] = per_cycle["force_bin"].map(lambda b: float(bins[int(b)]))
    per_cycle["force_bin_high_pn"] = per_cycle["force_bin"].map(lambda b: float(bins[int(b) + 1]))
    per_cycle = per_cycle[
        [
            "analysis_cycle",
            "force_bin",
            "force_bin_low_pn",
            "force_bin_high_pn",
            "extension_pull_out_nm",
            "extension_pull_in_nm",
            "delta_extension_nm",
            "mean_force_pn",
        ]
    ].reset_index(drop=True)

    summary = (
        per_cycle.groupby("force_bin", dropna=False)
        .agg(
            force_bin_low_pn=("force_bin_low_pn", "first"),
            force_bin_high_pn=("force_bin_high_pn", "first"),
            n_cycles=("analysis_cycle", "nunique"),
            mean_force_pn=("mean_force_pn", "mean"),
            delta_extension_mean_nm=("delta_extension_nm", "mean"),
            delta_extension_sd_nm=("delta_extension_nm", "std"),
        )
        .reset_index()
        .sort_values("force_bin")
        .reset_index(drop=True)
    )
    return per_cycle, summary


def average_pull_in_trace(
    df: pd.DataFrame,
    *,
    cycle_col: str | None = None,
    n_points: int = 200,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Stack and average pull_in trajectories in force-space."""
    _require_columns(df, ["phase", "force_pn", "extension_nm"])
    work = df.copy()
    work["analysis_cycle"] = infer_analysis_cycle(work) if cycle_col is None else work[cycle_col].astype("Int64")
    if "valid" in work.columns and valid_only:
        work = work[work["valid"].astype(bool)]
    work = work[work["phase"] == "pull_in"].copy()
    work = work[np.isfinite(work["force_pn"]) & np.isfinite(work["extension_nm"])]
    if work.empty:
        return pd.DataFrame(columns=["force_pn", "extension_mean_nm", "extension_sd_nm", "n_cycles"])

    force_ranges: list[tuple[int, float, float]] = []
    for cyc, grp in work.groupby("analysis_cycle", dropna=True):
        if len(grp) < 2:
            continue
        force_ranges.append((int(cyc), float(grp["force_pn"].min()), float(grp["force_pn"].max())))
    if not force_ranges:
        return pd.DataFrame(columns=["force_pn", "extension_mean_nm", "extension_sd_nm", "n_cycles"])

    f_low = max(v[1] for v in force_ranges)
    f_high = min(v[2] for v in force_ranges)
    if not np.isfinite(f_low) or not np.isfinite(f_high) or f_high <= f_low:
        return pd.DataFrame(columns=["force_pn", "extension_mean_nm", "extension_sd_nm", "n_cycles"])

    grid = np.linspace(f_low, f_high, max(2, int(n_points)))
    stacked: list[np.ndarray] = []
    for cyc, grp in work.groupby("analysis_cycle", dropna=True):
        interp = _interpolate_extension_at_force(
            grp["force_pn"].to_numpy(dtype=float),
            grp["extension_nm"].to_numpy(dtype=float),
            target_force=float(grid[0]),
        )
        if not np.isfinite(interp):
            continue
        force_arr = grp["force_pn"].to_numpy(dtype=float)
        ext_arr = grp["extension_nm"].to_numpy(dtype=float)
        finite = np.isfinite(force_arr) & np.isfinite(ext_arr)
        order = np.argsort(force_arr[finite])
        f = force_arr[finite][order]
        x = ext_arr[finite][order]
        uf, ui = np.unique(f, return_index=True)
        ux = x[ui]
        if len(uf) < 2:
            continue
        stacked.append(np.interp(grid, uf, ux))
    if not stacked:
        return pd.DataFrame(columns=["force_pn", "extension_mean_nm", "extension_sd_nm", "n_cycles"])

    arr = np.vstack(stacked)
    return pd.DataFrame(
        {
            "force_pn": grid,
            "extension_mean_nm": np.nanmean(arr, axis=0),
            "extension_sd_nm": np.nanstd(arr, axis=0, ddof=1 if arr.shape[0] > 1 else 0),
            "n_cycles": np.full(len(grid), arr.shape[0], dtype=int),
        }
    )
