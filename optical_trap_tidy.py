#!/usr/bin/env python3
"""Tidy per-sample dataset builder and routine-detection for optical-trap runs.

Public API
----------
load_tidy(path, k_mW=None, cal_file=None, k_per_mw=None, *, source=None) -> pd.DataFrame
    Load a single run as a tidy DataFrame with one row per sample.
    ``k_per_mw`` (or the deprecated alias ``k_mW``) sets trap-1 stiffness in
    pN/nm/mW; pass ``cal_file`` for a JSON/YAML calibration file instead.

find_routine_segments(df) -> pd.DataFrame
    Annotate ``mode`` column using the registered mode detectors.

label_cycles(df, dv_dt_threshold=0.0003, smooth_window=200) -> pd.DataFrame
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
    STATE_LABELS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSCILLATOR_RAMP_STATE = 4
OSCILLATOR_HOLD_STATE = 3
# Tether-detection state codes
TETHER_PULL_STATE = 29          # force-extension pull-out
TETHER_RELAX_STATE = 26         # pull-to-reference / relax
TETHER_REFERENCE_STATE = 27     # measure-reference hold
TETHER_APPROACH_STATE = 24      # approach (no tether)
TETHER_TOUCH_STATE = 25         # touch (bead–bead contact)
TETHER_POST_BREAK_STATE = 23    # post-break / retry
TETHER_STOP_STATE = 30          # stop-tether-detection
# Set of all states belonging to the tether-detection workflow
TETHER_DETECTION_STATES: frozenset[int] = frozenset(
    {TETHER_POST_BREAK_STATE, TETHER_APPROACH_STATE, TETHER_TOUCH_STATE,
     TETHER_RELAX_STATE, TETHER_REFERENCE_STATE, TETHER_PULL_STATE,
     TETHER_STOP_STATE}
)
# Legacy alias – kept for any external callers that import this name.
# The canonical pull state is the actual force-extension pull (29), not the
# reference hold (27).
FORCE_EXTENSION_STATE = TETHER_PULL_STATE

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
    edge_correction: bool = False,
    edge_correction_kwargs: dict | None = None,
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

    # Raw edge columns (only present for 7-column files)
    edge1_raw = trace.get("edge1_raw_nm", np.full(n, np.nan))
    edge2_raw = trace.get("edge2_raw_nm", np.full(n, np.nan))
    edge3_raw = trace.get("edge3_raw_nm", np.full(n, np.nan))
    edge4_raw = trace.get("edge4_raw_nm", np.full(n, np.nan))

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
            # Raw edge columns (NaN for 6-column files)
            "edge1_raw_nm": edge1_raw,
            "edge2_raw_nm": edge2_raw,
            "edge3_raw_nm": edge3_raw,
            "edge4_raw_nm": edge4_raw,
            # placeholders filled by downstream functions
            "mode": "other",
            "cycle": pd.NA,
            "phase": pd.NA,
            "setpoint_v": pd.NA,
            "cycle_complete": pd.NA,
            "cycle_status": pd.NA,
            "cycle_partial_reason": pd.NA,
            "cycle_last_phase": pd.NA,
            "cycle_next_state": pd.NA,
            "valid": True,
            # tether-detection columns (filled by label_cycles / compute_force_reference)
            "fe_attempt": pd.NA,
            "outcome": pd.NA,
            "force_ref_nm": np.nan,
            "force_ref_method": pd.NA,
            # linking columns (filled by optical_trap_tethers)
            "tether_id": pd.NA,
            "bead": pd.NA,
            "tether_number": pd.NA,
            "processing_id": pd.NA,
        }
    )
    df["cycle"] = df["cycle"].astype(pd.Int64Dtype())
    df["cycle_complete"] = df["cycle_complete"].astype(pd.BooleanDtype())
    df["cycle_next_state"] = df["cycle_next_state"].astype(pd.Int64Dtype())
    df["fe_attempt"] = df["fe_attempt"].astype(pd.Int64Dtype())

    # Optional edge-tracking correction
    if edge_correction:
        from optical_trap_edge_correction import correct_edge_tracking
        kwargs = edge_correction_kwargs or {}
        df = correct_edge_tracking(df, **kwargs)

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
            sample_count = r.end_idx - r.start_idx + 1
            return sample_count < _MIN_PATTERN_SAMPLES

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
                if [int(c) for c in codes[i : i + 4]] == [4, 3, 4, 3]:
                    quad_start_positions.append(i)
                    i += 4
                else:
                    i += 1
            # Keep the whole contiguous 3/4 run block so partial cycles at the
            # beginning/end are retained and can be flagged downstream.
            # If no full quad is present, still keep the block as a
            # partial-only oscillator segment.
            seg_start = block[0].start_idx
            seg_end = block[-1].end_idx
            segments.append((seg_start, seg_end))

        return segments

    def label_cycles_segment(
        self,
        df_seg: pd.DataFrame,
        dv_dt_threshold: float,
        smooth_window: int,
    ) -> pd.DataFrame:
        """Assign oscillator phases from the state column.

        * state 3  → ``"hold"``
        * state 4  → ``"pull_out"`` (falling voltage) or ``"pull_in"``
          (rising voltage); direction is determined from the overall voltage
          trend of each contiguous state-4 run, not per-sample dV/dt.
        * Any other state inside the segment is assigned ``"unknown"`` and
          a warning is logged with the affected times.
        """
        df_seg = df_seg.copy()
        n = len(df_seg)
        state = df_seg["state_code"].to_numpy(dtype=float)
        voltage = df_seg["voltage_v"].to_numpy(dtype=float)
        time_s = df_seg["time_s"].to_numpy(dtype=float)

        phase_arr = np.empty(n, dtype=object)

        # First pass: assign phase from state code
        for i in range(n):
            s = int(state[i]) if np.isfinite(state[i]) else -1
            if s == OSCILLATOR_HOLD_STATE:      # 3
                phase_arr[i] = "hold"
            elif s == OSCILLATOR_RAMP_STATE:    # 4
                phase_arr[i] = "ramp"           # temporary; filled below
            else:
                phase_arr[i] = "unknown"

        # Second pass: for each contiguous state-4 run, determine direction
        # from the overall voltage change (first → last sample of the run).
        # A state-4 run that never moves (or moves less than the threshold) is
        # treated as a stalled ramp rather than a valid pull_in, and is kept in
        # the cycle to preserve end-of-file partial information.
        diff = np.diff((state == OSCILLATOR_RAMP_STATE).astype(np.int8), prepend=0, append=0)
        ramp_starts = np.where(diff == 1)[0]
        ramp_ends = np.where(diff == -1)[0]
        for rs, re in zip(ramp_starts, ramp_ends):
            v_run = voltage[rs:re]
            if len(v_run) >= 2:
                delta_v = float(v_run[-1] - v_run[0])
                if abs(delta_v) <= max(abs(float(dv_dt_threshold)), 1e-12):
                    direction = "ramp_stalled"
                else:
                    direction = "pull_out" if v_run[-1] < v_run[0] else "pull_in"
            else:
                direction = "ramp_stalled"  # single-sample run: stalled/default
            phase_arr[rs:re] = direction

        # Warn about any unknown states
        unknown_mask = phase_arr == "unknown"
        if unknown_mask.any():
            unknown_times = time_s[unknown_mask]
            unknown_states = state[unknown_mask]
            unique_unknown = sorted(
                set(int(s) for s in unknown_states if np.isfinite(s))
            )
            logger.warning(
                "Oscillator segment: %d frame(s) with unknown state code(s) %s at "
                "times %.3f … %.3f s. Treating as 'hold'.",
                int(unknown_mask.sum()),
                unique_unknown,
                float(unknown_times[0]),
                float(unknown_times[-1]),
            )
            # Treat unknown states as hold for cycle numbering purposes
            phase_arr[unknown_mask] = "hold"

        # Assign cycles and setpoint
        cycle_arr = _assign_oscillator_cycles(phase_arr)

        df_seg["phase"] = phase_arr
        df_seg["cycle"] = pd.array(cycle_arr, dtype=pd.Int64Dtype())
        df_seg["setpoint_v"] = _setpoint_voltages(df_seg["voltage_v"].to_numpy(), phase_arr)
        return _annotate_oscillator_cycles(df_seg)


MODE_REGISTRY.register(OscillatorMode())


# ---------------------------------------------------------------------------
# Tether-detection mode
# ---------------------------------------------------------------------------


class TetherDetectionMode(ModeSpec):
    """Detect contiguous runs of tether-detection workflow states.

    States in scope: 23 (post_break), 24 (approach), 25 (touch),
    26 (pull_to_reference / relax), 27 (measure_reference), 29 (pull_out),
    30 (stop_tether_detection).

    Segment detection
    -----------------
    A segment is one or more contiguous tether-detection state runs, with
    single-sample glitches of other states skipped (same rule as
    OscillatorMode).

    Phase labelling (label_cycles_segment)
    ----------------------------------------
    * state 29 → ``pull_out``; each state-29 run starts a new ``fe_attempt``.
    * state 26 immediately after a state-29 run → ``relax``; same attempt.
    * state 27 → ``reference``; the fe_attempt of the nearest following pull
      is assigned, so references precede their pulls.
    * state 24 → ``approach``
    * state 25 → ``touch``
    * state 23 → ``post_break``
    * state 30 → ``stop``

    ``outcome`` is set on state-29 rows only:
    * ``survived``  – next tether-detection state after 29 is 26.
    * ``broke``     – next tether-detection state after 29 is 23 and force
                      exceeds the no-tether threshold at some point.
    * ``no_tether`` – next tether-detection state after 29 is 23 and force
                      never exceeds the threshold (force is NaN at this stage,
                      so the label is refined by
                      :func:`label_tether_detection_outcomes`).
    * ``unknown``   – anything else.
    """

    @property
    def name(self) -> str:
        return "tether_detection"

    def detect(
        self,
        state_codes: np.ndarray,
        actuation_codes: np.ndarray,
        time: np.ndarray,
    ) -> list[tuple[int, int]]:
        all_runs = segment_state_runs(state_codes)

        def _is_td_run(r) -> bool:
            return int(r.state_code) in TETHER_DETECTION_STATES

        def _is_glitch(r) -> bool:
            return (r.end_idx - r.start_idx + 1) < _MIN_PATTERN_SAMPLES

        filtered: list = []
        for r in all_runs:
            if _is_td_run(r):
                filtered.append(r)
            elif _is_glitch(r):
                continue
            else:
                filtered.append(None)

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
            segments.append((block[0].start_idx, block[-1].end_idx))
        return segments

    def label_cycles_segment(
        self,
        df_seg: pd.DataFrame,
        dv_dt_threshold: float,
        smooth_window: int,
    ) -> pd.DataFrame:
        df_seg = df_seg.copy()
        states = df_seg["state_code"].to_numpy(dtype=float)

        # Build phase array based on state code
        _STATE_TO_PHASE: dict[int, str] = {
            TETHER_PULL_STATE: "pull_out",
            TETHER_RELAX_STATE: "relax",
            TETHER_REFERENCE_STATE: "reference",
            TETHER_APPROACH_STATE: "approach",
            TETHER_TOUCH_STATE: "touch",
            TETHER_POST_BREAK_STATE: "post_break",
            TETHER_STOP_STATE: "stop",
        }
        n = len(df_seg)
        phase_arr = np.array(
            [_STATE_TO_PHASE.get(int(s), f"unknown_state_{int(s)}") for s in states],
            dtype=object,
        )
        df_seg["phase"] = phase_arr

        # fe_attempt: increments at each new state-29 run; shared with
        # immediately following state-26 (relax).
        state_runs = segment_state_runs(states)
        # Index offset from df_seg.index[0] to array index
        idx0 = df_seg.index[0]
        fe_attempt_arr = np.full(n, pd.NA, dtype=object)
        attempt = 0
        # Track which array positions belong to which state run
        run_codes = []
        run_slices = []
        for r in state_runs:
            lo = r.start_idx
            hi = r.end_idx + 1
            run_codes.append(int(r.state_code))
            run_slices.append((lo, hi))

        # Assign fe_attempt: each state-29 run increments the counter.
        # The following state-26 run gets the same attempt number.
        for i, (code, (lo, hi)) in enumerate(zip(run_codes, run_slices)):
            if code == TETHER_PULL_STATE:
                attempt += 1
                fe_attempt_arr[lo:hi] = attempt
                # Check next run: if it's state 26 → relax for same attempt
                if i + 1 < len(run_codes) and run_codes[i + 1] == TETHER_RELAX_STATE:
                    lo2, hi2 = run_slices[i + 1]
                    fe_attempt_arr[lo2:hi2] = attempt
                    df_seg.iloc[lo2:hi2, df_seg.columns.get_loc("phase")] = "relax"

        # State-27 runs get the fe_attempt of the *next* (upcoming) pull.
        # Walk forward to find the next pull attempt for each reference run.
        _next_by_position = np.full(n, pd.NA, dtype=object)
        current_next = pd.NA
        for i in range(len(run_codes) - 1, -1, -1):
            code, (lo, hi) = run_codes[i], run_slices[i]
            if code == TETHER_PULL_STATE:
                current_next = int(fe_attempt_arr[lo])
            _next_by_position[lo:hi] = current_next
        for i, (code, (lo, hi)) in enumerate(zip(run_codes, run_slices)):
            if code == TETHER_REFERENCE_STATE:
                fe_attempt_arr[lo:hi] = _next_by_position[lo:hi]

        # Convert to Int64 array
        int64_arr = pd.array(
            [int(x) if x is not pd.NA and x is not None else pd.NA
             for x in fe_attempt_arr],
            dtype=pd.Int64Dtype(),
        )
        df_seg["fe_attempt"] = int64_arr

        # outcome: set on pull_out rows based on the next state-detection run
        outcome_arr = np.full(n, pd.NA, dtype=object)
        for i, (code, (lo, hi)) in enumerate(zip(run_codes, run_slices)):
            if code == TETHER_PULL_STATE:
                # Find next tether-detection state run (skip non-TD states)
                next_td_code = None
                for j in range(i + 1, len(run_codes)):
                    if run_codes[j] in TETHER_DETECTION_STATES:
                        next_td_code = run_codes[j]
                        break
                if next_td_code == TETHER_RELAX_STATE:
                    outcome_str = "survived"
                elif next_td_code == TETHER_POST_BREAK_STATE:
                    # broke or no_tether: resolved after force is computed
                    outcome_str = "broke_or_no_tether"
                else:
                    outcome_str = "unknown"
                outcome_arr[lo:hi] = outcome_str
        df_seg["outcome"] = outcome_arr

        # setpoint_v: NaN for all (force-extension has no held setpoint)
        df_seg["setpoint_v"] = np.nan
        # cycle: use fe_attempt as the cycle number for compatibility
        df_seg["cycle"] = int64_arr

        return df_seg


MODE_REGISTRY.register(TetherDetectionMode())


# Keep a public alias so external code can still refer to ForceExtensionMode.
# It maps to the same tether_detection behaviour.
ForceExtensionMode = TetherDetectionMode


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
    # Find transitions into the first visible part of a cycle. In partial EOF
    # records, a stalled ramp following a hold can be the start of the last
    # partial cycle even before a full pull_out is observed.
    is_pull_out = phase == "pull_out"
    is_ramp_stalled = phase == "ramp_stalled"
    prev_is_hold_or_start = np.concatenate([[True], (phase[:-1] == "hold")])
    new_cycle_starts = (is_pull_out | is_ramp_stalled) & prev_is_hold_or_start

    if not new_cycle_starts.any():
        return cycle

    # Assign cycle numbers: cumsum of new_cycle_starts
    cycle_nums = np.cumsum(new_cycle_starts)
    # Only assign within the range where we've started at least one cycle
    started = np.where(new_cycle_starts)[0][0]
    cycle[started:] = cycle_nums[started:]
    return cycle


def _compress_phases(phase_values: np.ndarray) -> list[str]:
    """Return contiguous phase values as a compact sequence."""
    if len(phase_values) == 0:
        return []
    seq: list[str] = [str(phase_values[0])]
    for value in phase_values[1:]:
        sval = str(value)
        if sval != seq[-1]:
            seq.append(sval)
    return seq


def _count_hold_runs(phase_values: np.ndarray) -> int:
    """Count contiguous hold runs in a phase vector."""
    if len(phase_values) == 0:
        return 0
    is_hold = np.array(phase_values == "hold", dtype=bool)
    edges = np.diff(is_hold.astype(np.int8), prepend=0, append=0)
    starts = np.count_nonzero(edges == 1)
    return int(starts)


def _annotate_oscillator_cycles(df_seg: pd.DataFrame) -> pd.DataFrame:
    """Add per-cycle completeness metadata for one oscillator segment."""
    df_seg = df_seg.copy()
    n = len(df_seg)
    cycle = df_seg["cycle"].to_numpy(dtype=object)
    phase = df_seg["phase"].to_numpy(dtype=object)
    state = df_seg["state_code"].to_numpy(dtype=float)

    cycle_complete = np.full(n, pd.NA, dtype=object)
    cycle_status = np.full(n, pd.NA, dtype=object)
    cycle_reason = np.full(n, pd.NA, dtype=object)
    cycle_last_phase = np.full(n, pd.NA, dtype=object)
    cycle_next_state = np.full(n, pd.NA, dtype=object)

    unique_cycles = sorted(
        set(int(c) for c in cycle if c is not pd.NA and c is not None)
    )
    for cid in unique_cycles:
        idx = np.where(np.array([c is not pd.NA and c is not None and int(c) == cid for c in cycle]))[0]
        if idx.size == 0:
            continue

        phase_vals = phase[idx]
        seq = _compress_phases(phase_vals)
        hold_runs = _count_hold_runs(phase_vals)
        has_pull_out = "pull_out" in set(seq)
        has_pull_in = "pull_in" in set(seq)
        last_phase = str(seq[-1]) if seq else ""

        # A stalled ramp is not a valid pull_in, but it can still be part of an
        # end-of-file partial cycle. Treat such cycles as incomplete to keep the
        # frame assigned to the partial cycle rather than leaving it orphaned.
        is_complete = has_pull_out and has_pull_in and hold_runs >= 2 and last_phase == "hold"

        cycle_complete[idx] = bool(is_complete)
        cycle_last_phase[idx] = last_phase if last_phase else pd.NA

        end_i = int(idx[-1])
        next_state = pd.NA
        if end_i + 1 < n:
            next_state = state[end_i + 1]
            cycle_next_state[idx] = int(next_state) if np.isfinite(next_state) else pd.NA
        else:
            cycle_next_state[idx] = pd.NA

        if is_complete:
            cycle_status[idx] = "complete"
            continue

        missing: list[str] = []
        if not has_pull_out:
            missing.append("pull_out")
        if not has_pull_in:
            missing.append("pull_in")
        if hold_runs < 2 or last_phase != "hold":
            missing.append("final_hold")

        if next_state is pd.NA or next_state is None:
            cycle_status[idx] = "partial_end_of_file"
            cycle_reason[idx] = "stalled_ramp_at_eof" if "ramp_stalled" in set(seq) and not has_pull_in else "missing final_hold"
        else:
            cycle_status[idx] = "partial_truncated"
            cycle_reason[idx] = (
                f"missing {', '.join(missing)}" if missing else "incomplete phase pattern"
            )
            if "ramp_stalled" in set(seq) and not has_pull_in:
                cycle_reason[idx] = "stalled_ramp_at_eof"

    df_seg["cycle_complete"] = pd.array(cycle_complete, dtype=pd.BooleanDtype())
    df_seg["cycle_status"] = cycle_status
    df_seg["cycle_partial_reason"] = cycle_reason
    df_seg["cycle_last_phase"] = cycle_last_phase
    df_seg["cycle_next_state"] = pd.array(cycle_next_state, dtype=pd.Int64Dtype())
    return df_seg


def _warn_unassigned_oscillator_frames(df: pd.DataFrame) -> None:
    """Warn when state 3/4 frames are left without a cycle number."""
    mask = df["state_code"].isin([3, 4]) & df["cycle"].isna()
    if not mask.any():
        return

    idx = df.index[mask].to_numpy()
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    ranges = np.split(idx, breaks)
    spans: list[str] = []
    for rng in ranges:
        t0 = float(df.loc[rng[0], "time_s"])
        t1 = float(df.loc[rng[-1], "time_s"])
        spans.append(f"{t0:.3f}-{t1:.3f}s")
    logger.warning(
        "Oscillator: %d state-3/4 frame(s) were left without any cycle. Time range(s): %s",
        int(mask.sum()),
        ", ".join(spans),
    )


def _finalize_oscillator_cycle_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Finalize oscillator cycle numbering and completeness metadata."""
    df = df.copy()
    osc_mask = (df["mode"] == "oscillator") & df["cycle"].notna()
    if not osc_mask.any():
        _warn_unassigned_oscillator_frames(df)
        return df

    osc_idx = df.index[osc_mask].to_numpy()
    osc_cycles = df.loc[osc_idx, "cycle"].astype(int).to_numpy()
    osc_complete = df.loc[osc_idx, "cycle_complete"].astype("boolean").to_numpy()

    # Renumber oscillator cycles by start time so the cycle search remains in
    # chronological order even when the raw file IDs are sparse/non-monotonic.
    cycle_first_idx: dict[int, int] = {}
    for i, cid in zip(osc_idx, osc_cycles):
        cycle_first_idx.setdefault(int(cid), int(i))
    ordered_old_ids = sorted(cycle_first_idx.keys(), key=lambda c: cycle_first_idx[c])
    remap = {old: new for new, old in enumerate(ordered_old_ids, start=1)}
    df.loc[osc_mask, "cycle"] = pd.array([remap[int(c)] for c in osc_cycles], dtype=pd.Int64Dtype())

    # Fill next-state and end-of-file partial status after remapping.
    updated_cycles = sorted(df.loc[osc_mask, "cycle"].dropna().astype(int).unique().tolist())
    for cid in updated_cycles:
        cyc_rows = df[(df["mode"] == "oscillator") & (df["cycle"] == cid)]
        if cyc_rows.empty:
            continue
        end_idx = int(cyc_rows.index[-1])
        next_state = pd.NA
        if end_idx + 1 < len(df):
            ns = float(df.loc[end_idx + 1, "state_code"])
            if np.isfinite(ns):
                next_state = int(ns)
        df.loc[cyc_rows.index, "cycle_next_state"] = next_state
        is_complete = bool(cyc_rows["cycle_complete"].dropna().astype(bool).all()) if cyc_rows["cycle_complete"].notna().any() else False
        if not is_complete:
            trailing_states = df.iloc[end_idx + 1 :]["state_code"]
            if next_state is pd.NA:
                eof_partial = True
            elif int(next_state) == 2:
                eof_partial = False
            else:
                eof_partial = not trailing_states.isin([3, 4]).any()
            status = "partial_end_of_file" if eof_partial else "partial_truncated"
            df.loc[cyc_rows.index, "cycle_status"] = status
            t0 = float(cyc_rows["time_s"].iloc[0])
            t1 = float(cyc_rows["time_s"].iloc[-1])
            last_phase = cyc_rows["cycle_last_phase"].iloc[-1]
            reason = cyc_rows["cycle_partial_reason"].iloc[-1]
            logger.warning(
                "Oscillator partial cycle %d: %.3f-%.3f s (%s; last_phase=%s; next_state=%s; reason=%s)",
                int(cid),
                t0,
                t1,
                status,
                last_phase,
                "EOF" if eof_partial else str(int(next_state)) if next_state is not pd.NA else "EOF",
                reason if pd.notna(reason) else "unknown",
            )

    _warn_unassigned_oscillator_frames(df)
    return df


def _assign_fe_cycles(phase: np.ndarray) -> np.ndarray:
    """Number force-extension pulls.  Each force-extension run is one pull.

    Since ``force_extension`` is detected per-run (each state-27 run is its
    own segment), all samples in the segment get cycle 1.
    """
    cycle = np.ones(len(phase), dtype=np.intp)
    return cycle


def _setpoint_voltages(voltage: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Return voltage setpoint for ``hold`` samples, NaN otherwise."""
    setpoint = np.where(phase == "hold", voltage, np.nan).astype(float)
    return setpoint


# ---------------------------------------------------------------------------
# find_routine_segments
# ---------------------------------------------------------------------------


def find_routine_segments(
    df: pd.DataFrame,
    mode: str | None = None,
) -> pd.DataFrame:
    """Annotate the ``mode`` column using the requested detector set.

    Parameters
    ----------
    df:
        Tidy DataFrame (as returned by :func:`load_tidy`).
    mode:
        Optional mode name to restrict detection. Supported values are
        ``"oscillator"``, ``"tether_detection"``, and ``None`` for the
        default all-detector behaviour.

    Returns
    -------
    New DataFrame with ``mode`` column updated.
    """
    df = df.copy()
    n = len(df)
    mode_arr = np.full(n, "other", dtype=object)

    specs = MODE_REGISTRY.specs()
    if mode is not None:
        specs = [spec for spec in specs if spec.name == mode]

    for spec in specs:
        state_codes = df["state_code"].to_numpy(dtype=float)
        actuation_codes = df["actuation_state_code"].to_numpy(dtype=float)
        time = df["time_s"].to_numpy(dtype=float)
        segments = spec.detect(state_codes, actuation_codes, time)
        for start, end in segments:
            seg_mask = np.zeros(n, dtype=bool)
            seg_mask[start : end + 1] = True
            already_assigned = mode_arr != "other"
            conflict = seg_mask & already_assigned & (mode_arr != spec.name)
            if conflict.any():
                conflict_idxs = np.where(conflict)[0]
                for ci in conflict_idxs[:5]:
                    logger.warning(
                        "Sample %d matches both '%s' and '%s'; keeping '%s'",
                        ci,
                        mode_arr[ci],
                        spec.name,
                        mode_arr[ci],
                    )
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
                raw_int = np.array(int_arr, dtype=float)
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
            for extra_col in (
                "cycle_complete",
                "cycle_status",
                "cycle_partial_reason",
                "cycle_last_phase",
                "cycle_next_state",
            ):
                if extra_col in labelled.columns:
                    df.loc[seg_indices, extra_col] = labelled[extra_col].values
            # Propagate tether-detection columns if the mode set them.
            # fe_attempt uses the same values as cycle for TetherDetectionMode,
            # so apply the same shift that was already computed for cycle.
            if "fe_attempt" in labelled.columns:
                raw_fe = labelled["fe_attempt"].to_numpy(dtype=object)
                if "fe_attempt" in df.columns:
                    existing = df["fe_attempt"].dropna()
                    fe_offset = int(existing.max()) if len(existing) > 0 else 0
                else:
                    fe_offset = 0
                shifted_fe = pd.array(
                    [
                        int(v) + fe_offset if v is not pd.NA and v is not None else pd.NA
                        for v in raw_fe
                    ],
                    dtype=pd.Int64Dtype(),
                )
                df.loc[seg_indices, "fe_attempt"] = shifted_fe.to_numpy()
            if "outcome" in labelled.columns:
                df.loc[seg_indices, "outcome"] = labelled["outcome"].values

    if (df["mode"] == "oscillator").any():
        df = _finalize_oscillator_cycle_metadata(df)

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
    # flag the sample at the jump and the following one (no wrap-around)
    valid[spike_mask] = False
    shifted = np.zeros(len(spike_mask), dtype=bool)
    shifted[1:] = spike_mask[:-1]
    valid[shifted] = False

    df["valid"] = valid
    return df


# ---------------------------------------------------------------------------
# Force-from-reference helpers
# ---------------------------------------------------------------------------


def compute_force_reference(
    df: pd.DataFrame,
    method: str = "measure_ref",
    force_ref_nm: float | None = None,
    force_ref_frames: int = 100,
) -> pd.DataFrame:
    """Compute ``force_pn`` from a trap-1 position reference.

    Sign convention: tether tension is positive.  Under tension the bead in
    trap 1 is displaced *toward* the tether (toward higher ``trap1_pos_nm``
    on the ``tether1_overstretch`` data), so force is positive when
    ``trap1_pos_nm > ref_nm``.

    Parameters
    ----------
    df:
        Tidy DataFrame (after :func:`find_routine_segments` and
        :func:`label_cycles`). Must have ``stiffness_pn_per_nm``,
        ``trap1_pos_nm``, ``phase``, ``fe_attempt``, and ``mode`` columns.
    method:
        ``"measure_ref"``   – use the mean trap1_pos_nm of the nearest
                              preceding state-27 (measure_reference) hold for
                              each pull-out attempt.  Default for tether-
                              detection mode.
        ``"high_hold_tail"`` – use the mean of the last *force_ref_frames*
                              valid samples of each cycle's high-voltage hold
                              phase.  Default for oscillator mode.
        ``"fixed"``         – use a single fixed ``force_ref_nm`` value for
                              all rows.
    force_ref_nm:
        Used only with method ``"fixed"``.  In nm.
    force_ref_frames:
        Number of frames from the tail of the high-voltage hold to average.
        Used only with method ``"high_hold_tail"``.

    Returns
    -------
    DataFrame with updated ``force_pn``, ``force_ref_nm`` and
    ``force_ref_method`` columns.
    """
    df = df.copy()
    trap1 = df["trap1_pos_nm"].to_numpy(dtype=float)
    stiff = df["stiffness_pn_per_nm"].to_numpy(dtype=float)
    n = len(df)

    ref_col = np.full(n, np.nan)
    ref_method_col = np.full(n, method, dtype=object)
    ref_source_col = np.full(n, pd.NA, dtype=object)

    if method == "fixed":
        if force_ref_nm is None:
            warnings.warn("method='fixed' requires force_ref_nm; leaving force_pn as NaN.", stacklevel=2)
        else:
            ref_col[:] = force_ref_nm
            ref_source_col[:] = "fixed"

    elif method == "measure_ref":
        # For each fe_attempt, find the nearest preceding state-27 segment
        # and use its mean trap1_pos_nm as the reference.
        state_codes = df["state_code"].to_numpy(dtype=float)
        fe_attempts = df["fe_attempt"].to_numpy(dtype=object)

        # Build a mapping: fe_attempt → ref_nm from the nearest preceding ref
        ref_nm_by_attempt: dict[int, float] = {}
        ref_pos: list[tuple[int, float]] = []  # (last_row_idx, mean_trap1)
        for run in segment_state_runs(state_codes):
            if int(run.state_code) == TETHER_REFERENCE_STATE:
                valid_mask = df["valid"].to_numpy(dtype=bool)
                seg_trap1 = trap1[run.start_idx:run.end_idx + 1]
                seg_valid = valid_mask[run.start_idx:run.end_idx + 1]
                seg_finite = seg_trap1[seg_valid & np.isfinite(seg_trap1)]
                if len(seg_finite):
                    ref_pos.append((run.end_idx, float(np.mean(seg_finite))))

        # For each attempt, find the nearest preceding reference (last ref_pos
        # entry with end_idx < first row of this attempt).
        for run in segment_state_runs(state_codes):
            if int(run.state_code) == TETHER_PULL_STATE:
                fe_val = fe_attempts[run.start_idx]
                if fe_val is pd.NA or fe_val is None:
                    continue
                attempt_num = int(fe_val)
                if attempt_num in ref_nm_by_attempt:
                    continue
                # Find the nearest preceding reference (robust: take max ref_end_idx < start)
                preceding = [
                    (idx, mean) for idx, mean in ref_pos if idx < run.start_idx
                ]
                if preceding:
                    best_ref = max(preceding, key=lambda x: x[0])[1]
                else:
                    best_ref = None
                if best_ref is not None:
                    ref_nm_by_attempt[attempt_num] = best_ref

        # Apply reference to all rows of each attempt
        for i in range(n):
            fe = fe_attempts[i]
            if fe is not pd.NA and fe is not None:
                a = int(fe)
                if a in ref_nm_by_attempt:
                    ref_col[i] = ref_nm_by_attempt[a]
                    ref_source_col[i] = "measure_ref"

    elif method == "high_hold_tail":
        cycles = df["cycle"].to_numpy(dtype=object)
        phase_col = df["phase"].to_numpy(dtype=object)
        valid_mask = df["valid"].to_numpy(dtype=bool)
        voltage = df["voltage_v"].to_numpy(dtype=float)
        hold_phase_mask = np.array(
            [p == "hold" if (p is not pd.NA and p is not None) else False for p in phase_col],
            dtype=bool,
        )
        unique_cycles = sorted(
            set(int(c) for c in cycles if c is not pd.NA and c is not None)
        )
        cycle_time_by_id: dict[int, float] = {}
        for cid in unique_cycles:
            cmask = np.array(
                [c is not pd.NA and c is not None and c == cid for c in cycles],
                dtype=bool,
            )
            if np.any(cmask):
                cycle_time_by_id[int(cid)] = float(df.loc[cmask, "time_s"].iloc[0])

        ordered_cycle_ids = sorted(unique_cycles, key=lambda cid: cycle_time_by_id.get(int(cid), np.inf))
        last_complete_ref = np.nan
        # Voltage of the high-hold in the last complete cycle (set dynamically,
        # no hard-coded voltage values).  Used to detect partial cycles that
        # never reached the high-hold phase.
        last_complete_high_hold_voltage = np.nan
        for cid in ordered_cycle_ids:
            cmask = np.array(
                [c is not pd.NA and c is not None and c == cid for c in cycles],
                dtype=bool,
            )
            cycle_complete = (
                bool(df.loc[cmask, "cycle_complete"].dropna().astype(bool).all())
                if "cycle_complete" in df.columns and df.loc[cmask, "cycle_complete"].notna().any()
                else False
            )
            hold_mask = cmask & hold_phase_mask & valid_mask
            ref_val = np.nan
            source = pd.NA

            cycle_top_voltage = np.nan
            if hold_mask.any():
                cycle_top_voltage = float(np.nanmax(voltage[hold_mask]))

            high_voltage = np.nan
            if hold_mask.any():
                h_idx = np.where(hold_mask)[0]
                hold_breaks = np.where(np.diff(h_idx) > 1)[0] + 1
                hold_groups = np.split(h_idx, hold_breaks)
                hold_groups_sorted = sorted(
                    hold_groups,
                    key=lambda g: float(np.nanmean(voltage[g])),
                    reverse=True,
                )
                high_hold_idx = hold_groups_sorted[0]
                high_voltage = float(np.nanmean(voltage[high_hold_idx]))
                # For incomplete cycles: only accept the own hold as a reference
                # if it is at least as high as the last complete cycle's high-hold
                # voltage.  A partial cycle whose only hold is at a lower voltage
                # never reached the high-hold phase and must use the previous
                # complete cycle's reference instead.
                own_hold_ok = (
                    np.isfinite(high_voltage)
                    and np.isfinite(cycle_top_voltage)
                    and high_voltage >= cycle_top_voltage
                    and (
                        cycle_complete
                        or not np.isfinite(last_complete_high_hold_voltage)
                        or high_voltage >= last_complete_high_hold_voltage
                    )
                )
                if own_hold_ok:
                    tail_idx = high_hold_idx[-force_ref_frames:]
                    ref_val = float(np.nanmean(trap1[tail_idx]))
                    source = "own_high_hold"

            if (not np.isfinite(ref_val) or source is pd.NA) and np.any(cmask):
                cycle_tail = np.where(cmask)[0][-force_ref_frames:]
                if len(cycle_tail):
                    top_hold = (phase_col[cycle_tail] == "hold") & np.isfinite(voltage[cycle_tail])
                    if np.isfinite(cycle_top_voltage):
                        top_hold = top_hold & (voltage[cycle_tail] >= cycle_top_voltage)
                    # For incomplete cycles, do not accept a "relaxed end" hold
                    # that is below the last complete cycle's high-hold voltage.
                    if (
                        not cycle_complete
                        and np.isfinite(last_complete_high_hold_voltage)
                        and np.isfinite(cycle_top_voltage)
                        and cycle_top_voltage < last_complete_high_hold_voltage
                    ):
                        top_hold[:] = False
                    if np.any(top_hold):
                        tail_idx = cycle_tail[top_hold]
                        ref_val = float(np.nanmean(trap1[tail_idx[-force_ref_frames:]]))
                        source = "own_relaxed_end"

            if (not np.isfinite(ref_val) or source is pd.NA) and np.isfinite(last_complete_ref):
                ref_val = float(last_complete_ref)
                source = (
                    "prev_complete_high_hold"
                    if not cycle_complete
                    else "previous_cycle"
                )

            if np.isfinite(ref_val):
                ref_col[cmask] = ref_val
                ref_source_col[cmask] = source
            if cycle_complete and np.isfinite(ref_val):
                last_complete_ref = ref_val
                if source == "own_high_hold" and np.isfinite(high_voltage):
                    last_complete_high_hold_voltage = high_voltage

    # Compute force: positive when trap1 displaced toward higher nm (tension)
    force = (trap1 - ref_col) * stiff
    finite_force = np.isfinite(force)
    df["force_pn"] = np.where(finite_force, force, np.nan)
    df["force_ref_nm"] = ref_col
    df["force_ref_method"] = ref_method_col
    df["force_ref_source"] = ref_source_col

    return df


def flag_low_force_cycles(
    df: pd.DataFrame,
    n_sigma: float = 3.0,
) -> pd.DataFrame:
    """Flag oscillator cycles whose hold force is implausibly low.

    A cycle is marked ``force_suspect = True`` when its mean hold force at a
    given hold voltage is more than *n_sigma* standard deviations below the
    mean of *complete* cycles at the same hold voltage (within ±0.1 V).  The
    high-voltage reference hold (where force is near zero by definition) is
    excluded from the comparison.

    Call this **after** :func:`compute_force_reference`.

    Parameters
    ----------
    df:
        Tidy DataFrame with ``force_pn`` already populated.
    n_sigma:
        Sensitivity of the low-force check (default 3.0).

    Returns
    -------
    DataFrame with a new boolean ``force_suspect`` column (one value per row;
    all rows within a cycle share the same flag).
    """
    df = df.copy()
    df["force_suspect"] = False

    osc_mask = df["mode"] == "oscillator"
    if not osc_mask.any():
        return df

    osc = df[osc_mask]
    if "force_pn" not in osc.columns or osc["force_pn"].isna().all():
        return df

    if (osc["phase"] == "hold").sum() == 0:
        return df

    # ------------------------------------------------------------------ #
    # Step 1: collect per-hold-group stats for complete cycles, keyed by  #
    #         rounded voltage (0.1 V bins).                                #
    # ------------------------------------------------------------------ #
    complete_hold_force_by_v: dict[float, list[float]] = {}
    cycle_complete_col = osc["cycle_complete"] if "cycle_complete" in osc.columns else None

    for cid in osc["cycle"].dropna().unique():
        cyc = osc[osc["cycle"] == cid]
        if cycle_complete_col is not None:
            comp_vals = cyc["cycle_complete"].dropna()
            is_complete = bool(comp_vals.astype(bool).all()) if len(comp_vals) else False
        else:
            is_complete = True  # treat all as complete if column absent
        if not is_complete:
            continue
        cyc_holds = cyc[cyc["phase"] == "hold"]
        if cyc_holds.empty:
            continue
        h_idx = cyc_holds.index.to_numpy()
        hold_breaks = np.where(np.diff(h_idx) > 1)[0] + 1
        for grp_idx in np.split(h_idx, hold_breaks):
            grp = osc.loc[grp_idx]
            mean_v = round(float(grp["voltage_v"].mean()), 1)
            finite_f = grp["force_pn"].dropna()
            if len(finite_f) == 0:
                continue
            complete_hold_force_by_v.setdefault(mean_v, []).append(float(finite_f.mean()))

    if not complete_hold_force_by_v:
        return df

    # ------------------------------------------------------------------ #
    # Step 2: identify the reference-hold voltage bin (the one whose mean  #
    #         force across complete cycles is closest to zero).  This hold  #
    #         is the zero-force baseline and is excluded from the check.    #
    # ------------------------------------------------------------------ #
    ref_v_key = min(
        complete_hold_force_by_v,
        key=lambda v: abs(float(np.mean(complete_hold_force_by_v[v]))),
    )

    # Build reference statistics (mean ± std) for non-reference hold bins
    ref_stats: dict[float, tuple[float, float]] = {}
    for v, forces in complete_hold_force_by_v.items():
        if v == ref_v_key:
            continue
        arr = np.array(forces, dtype=float)
        ref_stats[v] = (float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0)

    if not ref_stats:
        return df

    # ------------------------------------------------------------------ #
    # Step 3: for each cycle flag holds whose force is implausibly low.   #
    # ------------------------------------------------------------------ #
    suspect_cycle_ids: set = set()
    for cid in osc["cycle"].dropna().unique():
        cyc = osc[osc["cycle"] == cid]
        cyc_holds = cyc[cyc["phase"] == "hold"]
        if cyc_holds.empty:
            continue
        h_idx = cyc_holds.index.to_numpy()
        hold_breaks = np.where(np.diff(h_idx) > 1)[0] + 1
        for grp_idx in np.split(h_idx, hold_breaks):
            grp = osc.loc[grp_idx]
            mean_v = round(float(grp["voltage_v"].mean()), 1)
            if mean_v not in ref_stats:
                continue  # skip the reference-hold bin and any untracked bins
            finite_f = grp["force_pn"].dropna()
            if len(finite_f) == 0:
                continue
            mean_f = float(finite_f.mean())
            ref_mean, ref_std = ref_stats[mean_v]
            threshold = n_sigma * ref_std if ref_std > 0 else 1.0
            if mean_f < ref_mean - threshold:
                suspect_cycle_ids.add(cid)

    # Write flag back to every row of the flagged cycles
    if suspect_cycle_ids:
        flagged_rows = osc["cycle"].isin(suspect_cycle_ids)
        df.loc[osc[flagged_rows].index, "force_suspect"] = True

    return df


def label_tether_detection_outcomes(
    df: pd.DataFrame,
    no_tether_threshold_pn: float = 3.0,
) -> pd.DataFrame:
    """Refine ``outcome`` for tether-detection attempts using computed force.

    This function resolves the ``"broke_or_no_tether"`` label (set by
    :class:`TetherDetectionMode`) into ``"broke"`` or ``"no_tether"`` based
    on whether the peak force during the pull ever exceeded
    *no_tether_threshold_pn*.

    Call this **after** :func:`compute_force_reference`.

    Parameters
    ----------
    df:
        Tidy DataFrame (after force computation).
    no_tether_threshold_pn:
        Minimum peak force (pN) during a pull that distinguishes a real
        tether (``"broke"``) from no tether (``"no_tether"``).

    Returns
    -------
    DataFrame with ``outcome`` refined.
    """
    df = df.copy()
    if "outcome" not in df.columns:
        return df

    outcome = df["outcome"].to_numpy(dtype=object).copy()
    force = df["force_pn"].to_numpy(dtype=float)
    fe_attempts = df["fe_attempt"].to_numpy(dtype=object)

    # Find unique attempts with "broke_or_no_tether"
    bn_mask = np.array(
        [o == "broke_or_no_tether" if (o is not pd.NA and o is not None) else False
         for o in outcome],
        dtype=bool,
    )
    if not bn_mask.any():
        return df

    outcome_bn_mask = np.array(
        [o == "broke_or_no_tether" if (o is not pd.NA and o is not None) else False for o in outcome],
        dtype=bool,
    )
    for attempt_val in set(fe_attempts[bn_mask]):
        if attempt_val is pd.NA or attempt_val is None:
            continue
        attempt_rows = np.array(
            [i for i, a in enumerate(fe_attempts)
             if a is not pd.NA and a is not None and a == attempt_val],
            dtype=np.intp,
        )
        pull_rows = attempt_rows[outcome_bn_mask[attempt_rows]]
        if len(pull_rows) == 0:
            continue
        pull_force = force[pull_rows]
        finite = pull_force[np.isfinite(pull_force)]
        if len(finite) and np.max(finite) > no_tether_threshold_pn:
            new_outcome = "broke"
        else:
            new_outcome = "no_tether"
        outcome[pull_rows] = new_outcome

    df["outcome"] = outcome
    return df


def extract_fe_curves(
    df: pd.DataFrame,
    outcome: str | None = None,
    tether_id: str | None = None,
) -> pd.DataFrame:
    """Return tidy pull-out and relax curves for tether-detection runs.

    Parameters
    ----------
    df:
        Tidy DataFrame (tether-detection mode, after force computation).
    outcome:
        If given, filter to attempts with this outcome (``"survived"``,
        ``"broke"``, ``"no_tether"``, ``"unknown"``).
    tether_id:
        If given, filter to rows with this tether_id.

    Returns
    -------
    DataFrame with columns ``source``, ``fe_attempt``, ``tether_id``,
    ``phase``, ``time_s``, ``extension_nm``, ``force_pn``, ``voltage_v``.
    """
    mask = df["mode"] == "tether_detection"
    sub = df[mask & df["phase"].isin(["pull_out", "relax"])].copy()

    if outcome is not None:
        # Attempt outcome is stored on pull_out rows; propagate to relax rows
        attempt_outcome: dict[object, str] = {}
        for _, row in sub[sub["phase"] == "pull_out"].iterrows():
            if row["outcome"] is not pd.NA:
                attempt_outcome[row["fe_attempt"]] = str(row["outcome"])
        keep_attempts = {a for a, o in attempt_outcome.items() if o == outcome}
        sub = sub[sub["fe_attempt"].apply(lambda a: a in keep_attempts)]

    if tether_id is not None:
        sub = sub[sub["tether_id"] == tether_id]

    cols = ["source", "fe_attempt", "tether_id", "phase",
            "time_s", "extension_nm", "force_pn", "voltage_v"]
    present = [c for c in cols if c in sub.columns]
    return sub[present].reset_index(drop=True)


def extract_reference_data(
    df: pd.DataFrame,
    phase: str = "approach",
) -> pd.DataFrame:
    """Return no-tether reference data for a given phase.

    Parameters
    ----------
    df:
        Tidy DataFrame (tether-detection mode).
    phase:
        Phase label to extract (default ``"approach"`` for state-24 data).

    Returns
    -------
    DataFrame filtered to the requested phase.
    """
    mask = (df["mode"] == "tether_detection") & (df["phase"] == phase)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rupture detection (Task C.6)
# ---------------------------------------------------------------------------


def detect_rupture_events(
    df: pd.DataFrame,
    *,
    threshold_pn: float = 5.0,
    n_samples: int = 100,
) -> pd.DataFrame:
    """Detect rupture events for each 'broke' pull attempt.

    Scans the ``pull_out`` phase of every broke attempt for the last valid
    sample before the force drops by more than *threshold_pn* within
    *n_samples* consecutive samples.  The frame immediately before the first
    such drop is taken as the rupture point.

    Parameters
    ----------
    df:
        Tidy DataFrame (tether-detection mode, after force computation and
        outcome labelling).
    threshold_pn:
        Force drop required to identify a rupture (pN).
    n_samples:
        Window of samples to check for the drop.

    Returns
    -------
    DataFrame with one row per broke attempt.  Columns:
    ``fe_attempt``, ``rupture_time_s``, ``rupture_force_pn``,
    ``rupture_extension_nm``.
    """
    records: list[dict[str, Any]] = []

    force = df["force_pn"].to_numpy(dtype=float)
    ext = df["extension_nm"].to_numpy(dtype=float) if "extension_nm" in df.columns else np.full(len(df), np.nan)
    t = df["time_s"].to_numpy(dtype=float)
    phase_col = df["phase"].to_numpy(dtype=object)
    outcome_col = df["outcome"].to_numpy(dtype=object) if "outcome" in df.columns else np.full(len(df), None, dtype=object)
    fe_col = df["fe_attempt"].to_numpy(dtype=object)
    valid_col = df["valid"].to_numpy(dtype=bool) if "valid" in df.columns else np.ones(len(df), dtype=bool)

    unique_attempts = sorted(
        set(int(a) for a in fe_col if a is not pd.NA and a is not None)
    )
    for attempt in unique_attempts:
        attempt_mask = np.array(
            [a is not pd.NA and a is not None and int(a) == attempt for a in fe_col],
            dtype=bool,
        )
        # Only process broke attempts
        attempt_outcomes = outcome_col[attempt_mask]
        if not any(
            o is not None and o is not pd.NA and o == "broke"
            for o in attempt_outcomes
        ):
            continue

        phase_mask = np.array(
            [p == "pull_out" if p is not pd.NA and p is not None else False for p in phase_col],
            dtype=bool,
        )
        pull_mask = attempt_mask & phase_mask & valid_col
        idx = np.where(pull_mask)[0]
        if len(idx) < 2:
            continue

        # Search forward for the first window where the force drops by more
        # than threshold_pn within n_samples frames, then record the preceding
        # peak sample instead of the post-break tail.
        pull_force = force[idx]
        rupture_i = None
        for j in range(len(idx) - 1):
            window_end = min(len(idx), j + n_samples + 1)
            window = pull_force[j:window_end]
            finite_mask = np.isfinite(window)
            if finite_mask.sum() < 2:
                continue
            peak_val = np.nanmax(window[finite_mask])
            final_val = window[finite_mask][-1]
            if (peak_val - final_val) >= threshold_pn:
                peak_positions = np.flatnonzero(finite_mask & (window == peak_val))
                rupture_i = idx[j + int(peak_positions[0])]
                break

        if rupture_i is not None:
            records.append(
                {
                    "fe_attempt": attempt,
                    "rupture_time_s": float(t[rupture_i]),
                    "rupture_force_pn": float(force[rupture_i]),
                    "rupture_extension_nm": float(ext[rupture_i]),
                }
            )

    return pd.DataFrame(records)


def extract_attempt_summary(
    df: pd.DataFrame,
    *,
    rupture_threshold_pn: float = 5.0,
    rupture_n_samples: int = 100,
) -> pd.DataFrame:
    """Return a per-attempt summary for tether-detection runs.

    Parameters
    ----------
    df:
        Tidy DataFrame (tether-detection mode, after force computation and
        outcome labelling).
    rupture_threshold_pn, rupture_n_samples:
        Forwarded to :func:`detect_rupture_events`.

    Returns
    -------
    DataFrame with one row per pull attempt.  Columns: ``fe_attempt``,
    ``outcome``, ``peak_force_pn``, ``max_extension_nm``,
    ``rupture_time_s``, ``rupture_force_pn``, ``rupture_extension_nm``,
    ``source``, ``tether_id``.
    """
    td_mask = df["mode"].eq("tether_detection")
    td_df = df[td_mask]
    pull_df = td_df[td_df["phase"].fillna(False).eq("pull_out")]
    if len(pull_df) == 0:
        return pd.DataFrame()

    ruptures = detect_rupture_events(
        td_df,
        threshold_pn=rupture_threshold_pn,
        n_samples=rupture_n_samples,
    )
    rup_by_attempt = ruptures.set_index("fe_attempt") if len(ruptures) else pd.DataFrame()

    source = df["source"].iloc[0] if "source" in df.columns else ""
    tid = df["tether_id"].iloc[0] if "tether_id" in df.columns else ""

    rows = []
    for attempt_val, grp in pull_df.groupby(pull_df["fe_attempt"].dropna().astype(int)):
        valid_grp = grp[grp["valid"].astype(bool)] if "valid" in grp.columns else grp
        outcome = grp["outcome"].iloc[0] if "outcome" in grp.columns else None
        peak_force = float(valid_grp["force_pn"].dropna().max()) if "force_pn" in valid_grp.columns else np.nan
        max_ext = float(valid_grp["extension_nm"].dropna().max()) if "extension_nm" in valid_grp.columns else np.nan

        row: dict[str, Any] = {
            "fe_attempt": int(attempt_val),
            "outcome": outcome,
            "peak_force_pn": peak_force,
            "max_extension_nm": max_ext,
            "rupture_time_s": np.nan,
            "rupture_force_pn": np.nan,
            "rupture_extension_nm": np.nan,
            "source": source,
            "tether_id": tid,
        }
        if len(rup_by_attempt) and attempt_val in rup_by_attempt.index:
            rrow = rup_by_attempt.loc[attempt_val]
            row["rupture_time_s"] = float(rrow["rupture_time_s"])
            row["rupture_force_pn"] = float(rrow["rupture_force_pn"])
            row["rupture_extension_nm"] = float(rrow["rupture_extension_nm"])
        rows.append(row)

    return pd.DataFrame(rows)


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

    # Count tether-detection attempts (state-29 fe_attempts)
    td_mask = df["mode"] == "tether_detection"
    n_fe_pulls = 0
    n_survived = 0
    n_broke = 0
    n_no_tether = 0
    if td_mask.any():
        td_df = df[td_mask]
        pull_rows = td_df[td_df["phase"] == "pull_out"]
        if len(pull_rows):
            n_fe_pulls = int(pull_rows["fe_attempt"].dropna().astype(int).max())
            if "outcome" in pull_rows.columns:
                outcomes = pull_rows.groupby(
                    pull_rows["fe_attempt"].dropna().astype(int)
                )["outcome"].first()
                n_survived = int((outcomes == "survived").sum())
                n_broke = int((outcomes == "broke").sum())
                n_no_tether = int((outcomes == "no_tether").sum())

    n_osc_cycles = 0
    n_complete_osc_cycles = 0
    n_partial_osc_cycles = 0
    if (df["mode"] == "oscillator").any():
        osc = df[df["mode"] == "oscillator"]
        if "cycle_complete" in osc.columns and osc["cycle_complete"].notna().any():
            complete_cycle_ids = (
                osc[osc["cycle_complete"].astype("boolean") == True]["cycle"]  # noqa: E712
                .dropna()
                .astype(int)
                .unique()
            )
            partial_cycle_ids = (
                osc[osc["cycle_complete"].astype("boolean") == False]["cycle"]  # noqa: E712
                .dropna()
                .astype(int)
                .unique()
            )
            n_complete_osc_cycles = len(complete_cycle_ids)
            n_partial_osc_cycles = len(partial_cycle_ids)
            # Back-compat: legacy reporting kept the count of complete cycles
            # only, while the new split counts are available separately.
            n_osc_cycles = n_complete_osc_cycles
        else:
            cycle_ids = osc["cycle"].dropna().astype(int).unique()
            n_osc_cycles = len(cycle_ids)
            n_complete_osc_cycles = n_osc_cycles
            n_partial_osc_cycles = 0

    # Count force_suspect cycles (set by flag_low_force_cycles)
    n_force_suspect_cycles = 0
    if (df["mode"] == "oscillator").any():
        osc_df = df[df["mode"] == "oscillator"]
        if "force_suspect" in osc_df.columns and osc_df["force_suspect"].any():
            suspect_cycle_ids = (
                osc_df[osc_df["force_suspect"].astype(bool)]["cycle"]
                .dropna()
                .astype(int)
                .unique()
            )
            n_force_suspect_cycles = len(suspect_cycle_ids)

    # Count and warn about unknown state codes
    n_unknown_states = 0
    if "state_code" in df.columns:
        sc = df["state_code"].dropna()
        sc_int = sc[sc.apply(lambda x: float(x) == int(float(x)))].astype(int)
        unknown_sc = sc_int[~sc_int.isin(STATE_LABELS.keys())]
        n_unknown_states = len(unknown_sc)
        if n_unknown_states:
            vc = unknown_sc.value_counts()
            logger.warning(
                "%s: unknown state code(s) encountered: %s",
                source,
                ", ".join(f"state {k} ({v} frame(s))" for k, v in vc.items()),
            )

    # Edge-correction summary if present
    edge_summary: dict[str, Any] = {}
    if "edge_flag" in df.columns:
        from optical_trap_edge_correction import summarize_edge_correction
        edge_summary = summarize_edge_correction(df)

    result: dict[str, Any] = {
        "source": source,
        "n_force_extension_pulls": n_fe_pulls,
        "n_tether_attempts": n_fe_pulls,
        "n_survived": n_survived,
        "n_broke": n_broke,
        "n_no_tether": n_no_tether,
        "n_complete_oscillator_cycles": int(n_complete_osc_cycles) if pd.notna(n_complete_osc_cycles) and n_complete_osc_cycles else 0,
        "n_partial_oscillator_cycles": int(n_partial_osc_cycles) if pd.notna(n_partial_osc_cycles) and n_partial_osc_cycles else 0,
        "n_oscillator_cycles": int(n_osc_cycles) if pd.notna(n_osc_cycles) and n_osc_cycles else 0,
        "n_force_suspect_cycles": int(n_force_suspect_cycles),
        "n_unknown_state_frames": n_unknown_states,
    }
    result.update(edge_summary)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expand_paths(raw: list[str]) -> list[Path]:
    """Expand a list of file/folder arguments to individual .dat file paths.

    If an argument is a directory, reads only the top-level .dat files,
    skipping ``*_Voltage.dat`` and ``*_params.dat``.  No recursion.
    """
    paths: list[Path] = []
    for p in raw:
        fp = Path(p)
        if fp.is_dir():
            dat_files = sorted(
                f for f in fp.glob("*.dat")
                if not f.name.endswith("_Voltage.dat")
                and not f.name.endswith("_params.dat")
            )
            if not dat_files:
                print(f"Warning: no .dat files found in directory {fp}", file=sys.stderr)
            paths.extend(dat_files)
        else:
            paths.append(fp)
    return paths


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
        help=(
            "One or more .dat run files to process, or a single folder "
            "(top-level .dat files only; _Voltage.dat and _params.dat skipped)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["oscillator", "tether-detection"],
        required=True,
        help=(
            "Analysis mode (required).  'oscillator' expects states 3/4; "
            "'tether-detection' expects states 23/24/25/26/27/29/30."
        ),
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
        help="dV/dt threshold (V/sample) retained for back-compatibility; no longer used for oscillator phase assignment.",
    )
    parser.add_argument(
        "--force-ref",
        choices=["measure_ref", "high_hold_tail", "fixed"],
        default=None,
        help=(
            "Force reference method.  Default: 'measure_ref' for "
            "tether-detection; 'high_hold_tail' for oscillator."
        ),
    )
    parser.add_argument(
        "--force-ref-nm",
        type=float,
        default=None,
        help="Fixed reference position for --force-ref fixed (nm).",
    )
    parser.add_argument(
        "--force-ref-frames",
        type=int,
        default=100,
        help="Number of tail frames for --force-ref high_hold_tail.",
    )
    parser.add_argument(
        "--edge-correction",
        action="store_true",
        default=False,
        help="Apply edge-tracking correction before analysis (default: off).",
    )
    # Tether / provenance flags (Task C.5)
    parser.add_argument("--date", default=None,
                        help="Experiment date (YYYYMMDD or YYYY-MM-DD) for tether ID generation.")
    parser.add_argument("--bead", type=int, default=None,
                        help="Bead number for tether ID generation.")
    parser.add_argument("--tether-id", default=None,
                        help="Explicit tether ID (overrides --date/--bead auto-generation).")
    parser.add_argument("--link-previous", action="store_true", default=False,
                        help="Attempt to link to the most recent active tether for this folder/bead.")
    parser.add_argument("--target", default=None,
                        help="Output directory for tethers.csv, provenance.json, session manifest.")
    parser.add_argument("--registry", default=None,
                        help="Path to the tether registry CSV file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    fpaths = _expand_paths(args.files)
    if not fpaths:
        print("Error: no input files found.", file=sys.stderr)
        return 1
    if args.link_previous and not args.registry:
        print(
            "Warning: --link-previous was requested but no --registry path was supplied; skipping registry lookup.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ #
    # Resolve tether metadata (Task C.5)                                   #
    # ------------------------------------------------------------------ #
    tether_id: str | None = args.tether_id
    metadata_source: dict[str, str] = {}
    date_val: str | None = None
    bead_val: int | None = None

    if tether_id is None:
        date_val = args.date
        bead_val = args.bead
        is_tty = sys.stdin.isatty()

        if date_val is None and is_tty:
            try:
                date_val = input("Experiment date (YYYYMMDD): ").strip() or None
                metadata_source["date"] = "prompt"
            except EOFError:
                pass
        else:
            metadata_source["date"] = "flag" if date_val is not None else "missing"

        if bead_val is None and is_tty:
            try:
                bv = input("Bead number: ").strip()
                bead_val = int(bv) if bv else None
                metadata_source["bead"] = "prompt"
            except (EOFError, ValueError):
                pass
        else:
            metadata_source["bead"] = "flag" if bead_val is not None else "missing"

    # ------------------------------------------------------------------ #
    # Determine output target directory                                    #
    # ------------------------------------------------------------------ #
    target_dir: Path | None = Path(args.target) if args.target else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    mode_name = {"oscillator": "oscillator", "tether-detection": "tether_detection"}[args.mode]
    for fpath in fpaths:
        try:
            df = load_tidy(
                fpath,
                k_per_mw=args.k_per_mw,
                cal_file=args.cal_file,
                edge_correction=args.edge_correction,
            )
            df = find_routine_segments(df, mode=mode_name)

            # Check that the expected mode was detected
            if not (df["mode"] == mode_name).any():
                print(
                    f"Warning: {fpath}: no '{mode_name}' segments detected. "
                    f"Check that the file contains the expected state codes.",
                    file=sys.stderr,
                )

            df = label_cycles(df, dv_dt_threshold=args.dv_dt_threshold)
            df = apply_valid_mask(df)

            # Determine force reference method
            force_ref_method = args.force_ref
            if force_ref_method is None:
                has_td = (df["mode"] == "tether_detection").any()
                has_osc = (df["mode"] == "oscillator").any()
                if has_td:
                    force_ref_method = "measure_ref"
                elif has_osc:
                    force_ref_method = "high_hold_tail"

            if force_ref_method and not df["stiffness_pn_per_nm"].isna().all():
                df = compute_force_reference(
                    df,
                    method=force_ref_method,
                    force_ref_nm=args.force_ref_nm,
                    force_ref_frames=args.force_ref_frames,
                )
                if force_ref_method in ("measure_ref",):
                    df = label_tether_detection_outcomes(df)

            # ---------------------------------------------------------- #
            # Tether / provenance metadata on rows (Task C.5)            #
            # ---------------------------------------------------------- #
            file_tether_id = tether_id
            if file_tether_id is None and args.link_previous and bead_val is not None:
                from optical_trap_tethers import TetherRegistry
                if args.registry:
                    reg = TetherRegistry(args.registry)
                    folder_path = str(Path(fpath).parent)
                    file_tether_id = reg.link_previous(folder_path, bead_val)
            if file_tether_id is not None:
                df["tether_id"] = file_tether_id
            if bead_val is not None:
                df["bead"] = bead_val
            if date_val is not None:
                df["date"] = date_val

            manifest_entries.append({"source_file": str(fpath), "tether_id": file_tether_id})

            frames.append(df)
            summaries.append(summarize_run(df))

            # ---------------------------------------------------------- #
            # Tether-detection outputs (Task C.7)                         #
            # ---------------------------------------------------------- #
            if mode_name == "tether_detection" and target_dir is not None:
                stem = Path(fpath).stem
                # Attempts CSV
                if "phase" in df.columns and df["phase"].fillna(False).eq("pull_out").any():
                    att_df = extract_attempt_summary(df)
                    att_path = target_dir / f"{stem}_attempts.csv"
                    att_df.to_csv(att_path, index=False)
                    print(f"  Attempts CSV written to {att_path}")

                # Force-extension figure
                if "force_pn" in df.columns and not df["force_pn"].isna().all():
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        fe_curves = extract_fe_curves(df)
                        if len(fe_curves):
                            fig, ax = plt.subplots(figsize=(8, 5))
                            for att_id, grp in fe_curves.groupby("fe_attempt"):
                                po = grp[grp["phase"] == "pull_out"]
                                relax = grp[grp["phase"] == "relax"]
                                ax.plot(po["extension_nm"], po["force_pn"],
                                        color="steelblue", lw=0.8, alpha=0.7,
                                        label="pull_out" if att_id == fe_curves["fe_attempt"].iloc[0] else "")
                                ax.plot(relax["extension_nm"], relax["force_pn"],
                                        color="coral", lw=0.8, alpha=0.7,
                                        label="relax" if att_id == fe_curves["fe_attempt"].iloc[0] else "")
                            ax.set_xlabel("Extension (nm)")
                            ax.set_ylabel("Force (pN)")
                            ax.set_title(f"Force-extension: {stem}")
                            ax.legend(loc="upper left")
                            fig_path = target_dir / f"{stem}_fe_curves.png"
                            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            print(f"  Force-extension figure written to {fig_path}")
                    except Exception as fig_exc:  # noqa: BLE001
                        print(f"  Warning: could not write FE figure: {fig_exc}", file=sys.stderr)

        except Exception as exc:
            print(f"Error processing {fpath}: {exc}", file=sys.stderr)
            return 1

    # print per-run summary
    print(
        f"{'Source':<30} {'FE_attempts':>11} {'Survived':>8} {'Broke':>6} "
        f"{'Complete':>10} {'Partial':>8}"
    )
    print("-" * 89)
    for s in summaries:
        print(
            f"{s['source']:<30} {s['n_tether_attempts']:>11} "
            f"{s['n_survived']:>8} {s['n_broke']:>6} "
            f"{s['n_complete_oscillator_cycles']:>10} {s['n_partial_oscillator_cycles']:>8}"
        )

    # ------------------------------------------------------------------ #
    # Tidy output file                                                     #
    # ------------------------------------------------------------------ #
    if args.output and frames:
        all_df = pd.concat(frames, ignore_index=True)
        out = Path(args.output)
        try:
            if out.suffix == ".parquet":
                all_df.to_parquet(out, index=False)
            else:
                all_df.to_csv(out, index=False)
        except ImportError as exc:
            print(
                f"Error writing {out}: {exc}. Install pyarrow to enable .parquet output.",
                file=sys.stderr,
            )
            return 1
        print(f"\nTidy output written to {out}")

    # ------------------------------------------------------------------ #
    # Provenance and session manifest (Task C.5)                          #
    # ------------------------------------------------------------------ #
    if target_dir is not None and frames:
        try:
            from optical_trap_tethers import (
                make_provenance_entry,
                write_provenance,
                write_session_manifest,
                TetherRegistry,
                generate_tether_id,
            )
            processing_id = str(__import__("uuid").uuid4())

            # Session manifest
            manifest_path = target_dir / "session_manifest.json"
            write_session_manifest(manifest_entries, manifest_path)
            print(f"\nSession manifest written to {manifest_path}")

            # Provenance
            params = {
                "mode": args.mode,
                "k_per_mw": args.k_per_mw,
                "cal_file": str(args.cal_file) if args.cal_file else None,
                "force_ref": args.force_ref,
                "force_ref_frames": args.force_ref_frames,
                "force_ref_nm": args.force_ref_nm,
                "edge_correction": args.edge_correction,
                "metadata_source": metadata_source,
            }
            for fpath in fpaths:
                entry = make_provenance_entry(
                    source_file=str(fpath),
                    parameters=params,
                    tether_ids=[tether_id] if tether_id else [],
                    processing_id=processing_id,
                )
                write_provenance(entry, target_dir / "provenance.json")
            print(f"Provenance written to {target_dir / 'provenance.json'}")

            # Registry
            if args.registry and date_val and bead_val is not None:
                reg = TetherRegistry(args.registry)
                for fpath in fpaths:
                    reg.create_tether(
                        date_val,
                        bead_val,
                        str(Path(fpath).parent),
                        source_file=str(fpath),
                        fe_attempt=None,
                        target=args.target or "",
                        processing_id=processing_id,
                    )
                reg.save()
                print(f"Tether registry updated at {args.registry}")
        except Exception as prov_exc:  # noqa: BLE001
            print(f"Warning: could not write provenance/manifest: {prov_exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
