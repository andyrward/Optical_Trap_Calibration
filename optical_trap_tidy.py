#!/usr/bin/env python3
"""Tidy per-sample dataset builder and routine-detection for optical-trap runs.

Public API
----------
load_tidy(path, k_mW=None, cal_file=None, k_per_mw=None) -> pd.DataFrame
    Load a single run as a tidy DataFrame with one row per sample.

find_routine_segments(df) -> pd.DataFrame
    Annotate ``mode`` column using the registered mode detectors.

label_cycles(df, dv_dt_threshold=0.1, smooth_window=50) -> pd.DataFrame
    Annotate ``cycle``, ``phase``, and ``setpoint_v`` columns.

apply_valid_mask(df, spike_threshold_nm=1000.0) -> pd.DataFrame
    Add / update the ``valid`` boolean column.

load_many(paths, k_mW=None, cal_file=None, k_per_mw=None) -> pd.DataFrame
    Concatenate several runs into one tidy table.

load_calibration(cal_file=None, k_per_mw_flag=None) -> dict
    Load calibration from a JSON/YAML file or a CLI flag value.

Mode registry
-------------
Modes are detected via small spec classes registered in ``MODE_REGISTRY``.  To
add a new mode: subclass ``ModeSpec``, implement ``detect`` and
``label_cycles_segment``, then call ``MODE_REGISTRY.register(MyMode())``.

Voltage sign convention
-----------------------
On this instrument, **falling voltage increases the tether extension magnitude**
(trap 1 moves away from trap 2).  Therefore:

* ``pull_out`` – voltage is falling (dV/dt < −threshold), extension magnitude
  is increasing.
* ``pull_in``  – voltage is rising  (dV/dt > +threshold), extension magnitude
  is decreasing.
* ``hold``     – |dV/dt| ≤ threshold.
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from optical_trap_toolbox import (
    load_normalized_trace,
    segment_state_runs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSCILLATOR_RAMP_STATE = 4
OSCILLATOR_HOLD_STATE = 3
FORCE_EXTENSION_STATE = 27

# Minimum number of samples a run must have to be considered real (not a
# single-sample glitch during pattern matching).
_MIN_PATTERN_SAMPLES = 2

# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def load_calibration(cal_file: str | Path | None = None, k_per_mw_flag: float | None = None) -> dict[str, Any]:
    """Load trap stiffness calibration.

    Priority: *k_per_mw_flag* (CLI ``--k-per-mw``) overrides *cal_file*.

    Parameters
    ----------
    cal_file:
        Path to a JSON (or YAML) calibration file.  Required keys:
        ``k_trap1_pn_per_nm_per_mw``.  Optional keys:
        ``k_trap2_pn_per_nm_per_mw``, ``source_file``, ``date``,
        ``fit_method``.
    k_per_mw_flag:
        Scalar stiffness for trap 1 (pN/nm/mW), overrides the file.

    Returns
    -------
    dict with keys ``k_trap1``, ``k_trap2`` (may be NaN), and ``calibration_id``.
    """
    cal: dict[str, Any] = {
        "k_trap1": float("nan"),
        "k_trap2": float("nan"),
        "calibration_id": "none",
    }

    if cal_file is not None:
        path = Path(cal_file)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            warnings.warn(f"Could not read calibration file {path}: {exc}")
            return cal

        # Support JSON directly; YAML support if PyYAML is present.
        data: dict[str, Any] = {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore[import]
                data = yaml.safe_load(text)
            except Exception:
                warnings.warn(f"Could not parse calibration file {path}: not JSON or YAML")
                return cal

        cal["k_trap1"] = float(data.get("k_trap1_pn_per_nm_per_mw", float("nan")))
        cal["k_trap2"] = float(data.get("k_trap2_pn_per_nm_per_mw", float("nan")))
        source = data.get("source_file", path.stem)
        date = data.get("date", "")
        method = data.get("fit_method", "")
        cal["calibration_id"] = f"{source}_{date}_{method}".strip("_")
        cal.update({k: v for k, v in data.items() if k not in cal})

    if k_per_mw_flag is not None:
        cal["k_trap1"] = float(k_per_mw_flag)
        cal["calibration_id"] = f"cli_k_per_mw={k_per_mw_flag}"

    return cal


# ---------------------------------------------------------------------------
# Tidy loader
# ---------------------------------------------------------------------------


def load_tidy(
    path: str | Path,
    k_mW: float | None = None,
    cal_file: str | Path | None = None,
    k_per_mw: float | None = None,
    *,
    source: str | None = None,
) -> pd.DataFrame:
    """Load a single run as a tidy per-sample DataFrame.

    Parameters
    ----------
    path:
        Path to the ``.dat`` file (7-column format expected for tether runs).
    k_mW:
        Trap-1 stiffness per mW (pN/nm/mW).  Deprecated alias for
        *k_per_mw*; if both are supplied, *k_per_mw* wins.
    cal_file:
        Path to a JSON/YAML calibration file.
    k_per_mw:
        Stiffness per mW; overrides *cal_file* and *k_mW*.
    source:
        Run identifier stored in the ``source`` column.  Defaults to the
        file stem.

    Returns
    -------
    pd.DataFrame with the tidy schema (one row per raw sample).

    Notes
    -----
    **Force and displacement reference**: trap-1 force requires the bead
    displacement from the trap centre.  For 7-column tether files, the raw
    data contains only absolute bead positions, not displacements.  Without a
    separately measured trap-centre reference, ``force_pn`` is left as NaN.
    Calibration files in the 6-column format include an explicit force column,
    which is preserved as-is.
    """
    path = Path(path)
    if source is None:
        source = path.stem

    # --- resolve stiffness ---------------------------------------------------
    effective_k = k_per_mw if k_per_mw is not None else k_mW
    cal = load_calibration(cal_file=cal_file, k_per_mw_flag=effective_k)
    k1 = cal["k_trap1"]
    cal_id = cal["calibration_id"]

    if np.isnan(k1):
        warnings.warn(
            f"{source}: no stiffness calibration supplied. "
            "'stiffness_pn_per_nm' and 'force_pn' will be NaN.",
            stacklevel=2,
        )

    # --- load raw trace -------------------------------------------------------
    trace = load_normalized_trace(path)
    n = trace["time_s"].size

    # --- build tidy columns --------------------------------------------------
    trap1 = trace.get("position_1_nm", np.full(n, np.nan))
    trap2 = trace.get("position_2_nm", np.full(n, np.nan))
    ext = trace.get("extension_nm", np.full(n, np.nan))
    power = trace.get("power_mw", np.full(n, np.nan))
    voltage = trace.get("voltage_v", np.full(n, np.nan))
    state_code = trace.get("state_code", np.full(n, np.nan))
    act_state = trace.get("actuation_state_code", np.zeros(n))

    # Stiffness: k_mW × power
    if not np.isnan(k1):
        stiffness = k1 * power
    else:
        stiffness = np.full(n, np.nan)

    # Force: stiffness × displacement from trap-centre.
    # For 7-column files the trap-centre reference is unknown; leave as NaN.
    force = trace.get("force_pn", np.full(n, np.nan))
    # If force_pn came from the 6-column format it is already set; otherwise
    # it remains NaN (trap-centre reference unavailable for 7-column files).

    df = pd.DataFrame(
        {
            "source": source,
            "sample_idx": np.arange(n, dtype=np.int64),
            "time_s": trace["time_s"],
            "trap1_pos_nm": trap1,
            "trap2_pos_nm": trap2,
            "extension_nm": ext,
            "power_mw": power,
            "voltage_v": voltage,
            "stiffness_pn_per_nm": stiffness,
            "force_pn": force,
            "state_code": state_code.astype(float),
            "actuation_state_code": act_state.astype(float),
            "calibration_id": cal_id,
            # placeholders filled by downstream functions
            "mode": "other",
            "cycle": pd.NA,
            "phase": pd.NA,
            "setpoint_v": pd.NA,
            "valid": True,
        }
    )
    df["cycle"] = df["cycle"].astype(pd.Int64Dtype())

    return df


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------


class ModeSpec(abc.ABC):
    """Abstract base for mode detectors.

    Subclasses implement:
    - ``name``: str label used in the ``mode`` column.
    - ``detect(state_codes, actuation_codes, time)``: returns a list of
      (start_idx, end_idx) tuples (inclusive) for each detected segment.
    - ``label_cycles_segment(df_seg, dv_dt_threshold, smooth_window)``:
      returns df_seg with ``cycle``, ``phase``, ``setpoint_v`` filled in.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def detect(
        self,
        state_codes: np.ndarray,
        actuation_codes: np.ndarray,
        time: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Return list of (start_idx, end_idx) inclusive index pairs."""
        ...

    @abc.abstractmethod
    def label_cycles_segment(
        self,
        df_seg: pd.DataFrame,
        dv_dt_threshold: float,
        smooth_window: int,
    ) -> pd.DataFrame:
        ...


class _ModeRegistry:
    def __init__(self) -> None:
        self._specs: list[ModeSpec] = []

    def register(self, spec: ModeSpec) -> None:
        self._specs.append(spec)

    def specs(self) -> list[ModeSpec]:
        return list(self._specs)


MODE_REGISTRY = _ModeRegistry()


# ---------------------------------------------------------------------------
# Oscillator mode
# ---------------------------------------------------------------------------


class OscillatorMode(ModeSpec):
    """Detect the repeating [state 4, state 3, state 4, state 3] pattern.

    Each complete quad of (ramp, short-hold, ramp, long-hold) is one cycle.

    Detection strategy
    ------------------
    1. Build state runs, skipping single-sample glitches (runs of length 1
       with state codes other than 3 or 4).
    2. Find consecutive sub-sequences of state-3/4 runs.
    3. Within each such block, scan for quads [4, 3, 4, 3].
    4. Return the sample-index range of each contiguous run of quads as a
       segment.
    """

    @property
    def name(self) -> str:
        return "oscillator"

    def detect(
        self,
        state_codes: np.ndarray,
        actuation_codes: np.ndarray,
        time: np.ndarray,
    ) -> list[tuple[int, int]]:
        all_runs = segment_state_runs(state_codes)

        # Filter: keep runs with state 3 or 4, skip tiny glitch runs of other
        # states.  A run of state 3/4 with only 1 sample is kept (it may be a
        # genuine very-short hold at high sample rate).
        def _is_osc_run(r) -> bool:
            return r.state_code in (OSCILLATOR_RAMP_STATE, OSCILLATOR_HOLD_STATE)

        def _is_glitch(r) -> bool:
            return (r.end_idx - r.start_idx) < _MIN_PATTERN_SAMPLES

        filtered: list = []
        for r in all_runs:
            if _is_osc_run(r):
                filtered.append(r)
            elif _is_glitch(r):
                # single-sample non-osc run: treat as a transparent glitch,
                # don't break the consecutive-3/4 run.
                continue
            else:
                # real break: emit a sentinel
                filtered.append(None)

        # Split filtered list at sentinels → candidate blocks of 3/4 runs
        blocks: list[list] = []
        current: list = []
        for item in filtered:
            if item is None:
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(item)
        if current:
            blocks.append(current)

        segments: list[tuple[int, int]] = []
        for block in blocks:
            codes = [r.state_code for r in block]
            # Find complete non-overlapping quads [4, 3, 4, 3]
            quad_start_positions: list[int] = []
            i = 0
            while i <= len(codes) - 4:
                if codes[i : i + 4] == [4, 3, 4, 3]:
                    quad_start_positions.append(i)
                    i += 4
                else:
                    i += 1
            if not quad_start_positions:
                continue
            # Within a block all quads are drawn from consecutive 3/4 runs;
            # merge them all into one segment.
            seg_start = block[quad_start_positions[0]].start_idx
            seg_end = block[quad_start_positions[-1] + 3].end_idx
            segments.append((seg_start, seg_end))

        return segments

    def label_cycles_segment(
        self,
        df_seg: pd.DataFrame,
        dv_dt_threshold: float,
        smooth_window: int,
    ) -> pd.DataFrame:
        df_seg = df_seg.copy()
        voltage = df_seg["voltage_v"].to_numpy(dtype=float)
        phase_arr = _voltage_to_phase(voltage, dv_dt_threshold, smooth_window)

        # Assign cycles: each transition from a hold-at-high-V back to pull_out
        # starts a new cycle.  We detect cycles as runs of the 4-phase pattern.
        cycle_arr = _assign_oscillator_cycles(phase_arr)

        df_seg["phase"] = phase_arr
        df_seg["cycle"] = pd.array(cycle_arr, dtype=pd.Int64Dtype())
        df_seg["setpoint_v"] = _setpoint_voltages(df_seg["voltage_v"].to_numpy(), phase_arr)
        return df_seg


MODE_REGISTRY.register(OscillatorMode())


# ---------------------------------------------------------------------------
# Force-extension mode
# ---------------------------------------------------------------------------


class ForceExtensionMode(ModeSpec):
    """Detect contiguous runs of state 27 (force-extension pulls)."""

    @property
    def name(self) -> str:
        return "force_extension"

    def detect(
        self,
        state_codes: np.ndarray,
        actuation_codes: np.ndarray,
        time: np.ndarray,
    ) -> list[tuple[int, int]]:
        runs = segment_state_runs(state_codes, state_code=FORCE_EXTENSION_STATE)
        return [(r.start_idx, r.end_idx) for r in runs]

    def label_cycles_segment(
        self,
        df_seg: pd.DataFrame,
        dv_dt_threshold: float,
        smooth_window: int,
    ) -> pd.DataFrame:
        df_seg = df_seg.copy()
        voltage = df_seg["voltage_v"].to_numpy(dtype=float)
        phase_arr = _voltage_to_phase(voltage, dv_dt_threshold, smooth_window)
        cycle_arr = _assign_fe_cycles(phase_arr)
        df_seg["phase"] = phase_arr
        df_seg["cycle"] = pd.array(cycle_arr, dtype=pd.Int64Dtype())
        df_seg["setpoint_v"] = _setpoint_voltages(df_seg["voltage_v"].to_numpy(), phase_arr)
        return df_seg


MODE_REGISTRY.register(ForceExtensionMode())


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _smooth_gradient(voltage: np.ndarray, smooth_window: int) -> np.ndarray:
    """Return a rolling-endpoint voltage trend (V/sample).

    At each sample *i*, the trend is the mean voltage change per sample over
    the preceding *smooth_window* samples::

        trend[i] = (V[i] − V[max(0, i−window)]) / span

    This is robust to the stepped (staircase) voltage waveform produced by
    MATLAB on this instrument, where per-sample differences are dominated by
    step-quantisation noise even after box-car smoothing.

    Typical values at ~1400 Hz:
    * Oscillator ramp (≈2.54 V/s):  −0.0018 V/sample
    * Hold (0 V/s):                   0.0000 V/sample
    """
    n = voltage.size
    if n < 2:
        return np.zeros(n, dtype=float)
    window = max(2, min(smooth_window, n - 1))
    v = np.asarray(voltage, dtype=float)
    # Vectorized rolling-endpoint: trend[i] = (v[i] - v[max(0, i-window)]) / span
    idx = np.arange(n, dtype=np.intp)
    starts = np.maximum(0, idx - window)
    spans = idx - starts
    spans = np.where(spans == 0, 1, spans)
    trend = (v[idx] - v[starts]) / spans
    return trend


def _voltage_to_phase(
    voltage: np.ndarray,
    dv_dt_threshold: float,
    smooth_window: int,
) -> np.ndarray:
    """Map voltage to phase labels for each sample.

    Voltage sign convention (instrument-specific):
    * falling V (dV/dt < −threshold) → ``pull_out``
    * rising  V (dV/dt > +threshold) → ``pull_in``
    * |dV/dt| ≤ threshold            → ``hold``
    """
    dv = _smooth_gradient(voltage, smooth_window)
    phase = np.where(
        dv < -dv_dt_threshold,
        "pull_out",
        np.where(dv > dv_dt_threshold, "pull_in", "hold"),
    )
    return phase


def _assign_oscillator_cycles(phase: np.ndarray) -> np.ndarray:
    """Number oscillator cycles.

    A new cycle starts at each ``pull_out`` run that immediately follows a
    ``hold`` (or at the very first ``pull_out``).
    """
    n = len(phase)
    cycle = np.full(n, pd.NA, dtype=object)
    # Find transitions into pull_out from hold (or start)
    is_pull_out = phase == "pull_out"
    # shifted: previous phase
    prev_is_hold_or_start = np.concatenate([[True], (phase[:-1] == "hold")])
    new_cycle_starts = is_pull_out & prev_is_hold_or_start

    if not new_cycle_starts.any():
        return cycle

    # Assign cycle numbers: cumsum of new_cycle_starts
    cycle_nums = np.cumsum(new_cycle_starts)
    # Only assign within the range where we've started at least one cycle
    started = np.where(new_cycle_starts)[0][0]
    cycle[started:] = cycle_nums[started:]
    return cycle


def _assign_fe_cycles(phase: np.ndarray) -> np.ndarray:
    """Number force-extension pulls.  Each force-extension run is one pull.

    Since ``force_extension`` is detected per-run (each state-27 run is its
    own segment), all samples in the segment get cycle 1.
    """
    cycle = np.ones(len(phase), dtype=object)
    return cycle


def _setpoint_voltages(voltage: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Return voltage setpoint for ``hold`` samples, NaN otherwise."""
    setpoint = np.where(phase == "hold", voltage, np.nan).astype(float)
    return setpoint


# ---------------------------------------------------------------------------
# find_routine_segments
# ---------------------------------------------------------------------------


def find_routine_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate the ``mode`` column using all registered mode detectors.

    Raises a warning (not an error) when a sample matches more than one mode;
    the first-registered mode wins and a ``mode_conflict`` note is logged.

    Parameters
    ----------
    df:
        Tidy DataFrame (as returned by :func:`load_tidy`).

    Returns
    -------
    New DataFrame with ``mode`` column updated.
    """
    df = df.copy()
    n = len(df)
    mode_arr = np.full(n, "other", dtype=object)
    conflict_arr = np.zeros(n, dtype=bool)

    for spec in MODE_REGISTRY.specs():
        state_codes = df["state_code"].to_numpy(dtype=float)
        actuation_codes = df["actuation_state_code"].to_numpy(dtype=float)
        time = df["time_s"].to_numpy(dtype=float)
        segments = spec.detect(state_codes, actuation_codes, time)
        for start, end in segments:
            seg_mask = np.zeros(n, dtype=bool)
            seg_mask[start : end + 1] = True
            # Where already assigned to a different mode → conflict
            already_assigned = mode_arr != "other"
            conflict = seg_mask & already_assigned & (mode_arr != spec.name)
            if conflict.any():
                conflict_idxs = np.where(conflict)[0]
                for ci in conflict_idxs[:5]:  # log first 5
                    logger.warning(
                        "Sample %d matches both '%s' and '%s'; keeping '%s'",
                        ci,
                        mode_arr[ci],
                        spec.name,
                        mode_arr[ci],
                    )
            # Assign only to samples not yet assigned
            new_assign = seg_mask & ~already_assigned
            mode_arr[new_assign] = spec.name

    df["mode"] = mode_arr
    return df


# ---------------------------------------------------------------------------
# label_cycles
# ---------------------------------------------------------------------------


def label_cycles(
    df: pd.DataFrame,
    dv_dt_threshold: float = 0.0003,
    smooth_window: int = 200,
) -> pd.DataFrame:
    """Assign ``cycle``, ``phase``, and ``setpoint_v`` to each row.

    For each detected mode segment, the corresponding :class:`ModeSpec`
    ``label_cycles_segment`` method is called.

    Parameters
    ----------
    df:
        Tidy DataFrame after :func:`find_routine_segments`.
    dv_dt_threshold:
        Voltage rate-of-change (V/sample) used to separate ramp from hold.
        Samples with |dV/dt| ≤ this value are labelled ``hold``; those
        outside it are labelled ``pull_out`` or ``pull_in`` depending on sign.
        At the typical 1400 Hz acquisition rate the faster oscillator ramp
        (pull_out) runs at ≈ −0.0018 V/sample and the slower ramp (pull_in)
        at ≈ +0.0006 V/sample; the default 0.0003 V/sample comfortably
        separates both ramps from the zero-noise hold segments.  Adjust for
        different acquisition rates or ramp speeds.
    smooth_window:
        Number of samples over which to compute the rolling-endpoint voltage
        trend.  Larger values give a cleaner separation at the cost of a
        delayed response at phase transitions.

    Returns
    -------
    New DataFrame with ``cycle``, ``phase``, and ``setpoint_v`` filled.
    """
    df = df.copy()
    spec_by_name = {s.name: s for s in MODE_REGISTRY.specs()}

    for mode_name, spec in spec_by_name.items():
        mask = df["mode"] == mode_name
        if not mask.any():
            continue

        # Find contiguous blocks of this mode
        indices = df.index[mask].to_numpy()
        if indices.size == 0:
            continue

        # Split into contiguous runs of row-indices
        breaks = np.where(np.diff(indices) > 1)[0] + 1
        segments = np.split(indices, breaks)

        # Number cycles continuously across segments of the same mode.
        cycle_offset = 0
        for seg_indices in segments:
            seg = df.loc[seg_indices]
            labelled = spec.label_cycles_segment(seg, dv_dt_threshold, smooth_window)
            raw_cycle = labelled["cycle"].to_numpy()
            # Shift cycle numbers by offset (vectorized)
            int_arr = pd.array(raw_cycle, dtype=pd.Int64Dtype())
            finite_mask = ~int_arr.isna()
            if finite_mask.any():
                raw_int = np.where(np.array(finite_mask), raw_cycle.astype(float), np.nan)
                max_in_seg = int(np.nanmax(raw_int)) if not np.all(np.isnan(raw_int)) else 0
                shifted_int = np.where(np.array(finite_mask), raw_int + cycle_offset, np.nan)
                cycle_offset += max_in_seg
                shifted_cycle = pd.array(
                    [int(x) if not np.isnan(x) else pd.NA for x in shifted_int],
                    dtype=pd.Int64Dtype(),
                )
            else:
                shifted_cycle = pd.array(raw_cycle, dtype=pd.Int64Dtype())
            df.loc[seg_indices, "cycle"] = shifted_cycle.to_numpy()
            df.loc[seg_indices, "phase"] = labelled["phase"].values
            df.loc[seg_indices, "setpoint_v"] = labelled["setpoint_v"].values

    return df


# ---------------------------------------------------------------------------
# valid mask
# ---------------------------------------------------------------------------


def apply_valid_mask(
    df: pd.DataFrame,
    spike_threshold_nm: float = 1000.0,
) -> pd.DataFrame:
    """Add / update the ``valid`` boolean column.

    Two classes of invalid samples are detected:

    1. **State glitches**: single-sample state runs (a state code that appears
       in only one consecutive sample, surrounded by different state codes).
    2. **Extension spikes**: jumps in ``extension_nm`` larger than
       *spike_threshold_nm* between adjacent samples.  Both the sample that
       caused the jump and the one that follows are flagged (edge spikes
       typically appear as positive then negative artefacts at ramp edges).
    """
    df = df.copy()
    valid = df["valid"].to_numpy(dtype=bool).copy()

    # 1. Single-sample state glitches
    state_codes = df["state_code"].to_numpy(dtype=float)
    all_runs = segment_state_runs(state_codes)
    for run in all_runs:
        if run.start_idx == run.end_idx:
            # single sample
            row_pos = run.start_idx
            if row_pos < len(valid):
                valid[row_pos] = False

    # 2. Extension spikes
    ext = df["extension_nm"].to_numpy(dtype=float)
    dext = np.abs(np.diff(ext, prepend=ext[0]))
    spike_mask = dext > spike_threshold_nm
    # flag the sample at the jump and the following one
    valid[spike_mask] = False
    shifted = np.roll(spike_mask, 1)
    shifted[0] = False
    valid[shifted] = False

    df["valid"] = valid
    return df


# ---------------------------------------------------------------------------
# load_many
# ---------------------------------------------------------------------------


def load_many(
    paths: list[str | Path],
    k_mW: float | None = None,
    cal_file: str | Path | None = None,
    k_per_mw: float | None = None,
) -> pd.DataFrame:
    """Load and label several runs; concatenate into one tidy DataFrame.

    Parameters
    ----------
    paths:
        List of ``.dat`` file paths.
    k_mW, cal_file, k_per_mw:
        Stiffness calibration forwarded to :func:`load_tidy`.

    Returns
    -------
    Concatenated DataFrame with all runs.
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = load_tidy(p, k_mW=k_mW, cal_file=cal_file, k_per_mw=k_per_mw)
        df = find_routine_segments(df)
        df = label_cycles(df)
        df = apply_valid_mask(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-run summary helper
# ---------------------------------------------------------------------------


def summarize_run(df: pd.DataFrame) -> dict[str, Any]:
    """Return a dict with pull counts and oscillator cycle counts for one run."""
    source = df["source"].iloc[0] if "source" in df.columns else "unknown"

    n_fe_pulls = (
        df[df["mode"] == "force_extension"]["cycle"]
        .dropna()
        .astype(int)
        .max()
        if (df["mode"] == "force_extension").any()
        else 0
    )
    n_osc_cycles = (
        df[df["mode"] == "oscillator"]["cycle"]
        .dropna()
        .astype(int)
        .max()
        if (df["mode"] == "oscillator").any()
        else 0
    )

    return {
        "source": source,
        "n_force_extension_pulls": int(n_fe_pulls) if n_fe_pulls else 0,
        "n_oscillator_cycles": int(n_osc_cycles) if n_osc_cycles else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tidy per-sample dataset from optical-trap run files and "
            "print a per-run summary."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more .dat run files to process.",
    )
    parser.add_argument(
        "--k-per-mw",
        type=float,
        default=None,
        help="Trap-1 stiffness per mW (pN/nm/mW).  Overrides --cal-file.",
    )
    parser.add_argument(
        "--cal-file",
        default=None,
        help="Path to a JSON/YAML calibration file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the tidy output file.  Extension determines format: "
            ".csv (default) or .parquet."
        ),
    )
    parser.add_argument(
        "--dv-dt-threshold",
        type=float,
        default=0.0003,
        help="dV/dt threshold (V/sample) for hold/ramp phase discrimination.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for fpath in args.files:
        try:
            df = load_tidy(fpath, k_per_mw=args.k_per_mw, cal_file=args.cal_file)
            df = find_routine_segments(df)
            df = label_cycles(df, dv_dt_threshold=args.dv_dt_threshold)
            df = apply_valid_mask(df)
            frames.append(df)
            summaries.append(summarize_run(df))
        except Exception as exc:
            print(f"Error processing {fpath}: {exc}", file=sys.stderr)
            return 1

    # print per-run summary: pulls first, then oscillator cycles
    print(f"{'Source':<30} {'FE_pulls':>8} {'Osc_cycles':>10}")
    print("-" * 52)
    for s in summaries:
        print(f"{s['source']:<30} {s['n_force_extension_pulls']:>8} {s['n_oscillator_cycles']:>10}")

    if args.output and frames:
        all_df = pd.concat(frames, ignore_index=True)
        out = Path(args.output)
        if out.suffix == ".parquet":
            all_df.to_parquet(out, index=False)
        else:
            all_df.to_csv(out, index=False)
        print(f"\nTidy output written to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
