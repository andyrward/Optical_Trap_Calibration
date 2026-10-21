#!/usr/bin/env python3
"""Edge-tracking correction for dual-trap optical-trap data.

Public API
----------
correct_edge_tracking(df, ...) -> pd.DataFrame
    Detect and correct edge-tracking failures in a tidy DataFrame that
    contains raw edge columns (``edge1_raw_nm`` … ``edge4_raw_nm``).

Column conventions (7-column MATLAB files, 1-indexed)
------------------------------------------------------
* edge1_raw_nm – col2, trap-1 **outer** edge
* edge2_raw_nm – col3, trap-2 **outer** edge
* edge3_raw_nm – col4, trap-1 **interface** edge  (main failure source)
* edge4_raw_nm – col5, trap-2 **interface** edge

Width for trap 1 = edge3 − edge1  (interface − outer).
Width for trap 2 = edge4 − edge2.

Flag values written to ``edge_flag_trap1``, ``edge_flag_trap2``, ``edge_flag``
-------------------------------------------------------------------------------
* ``"none"``          – no correction applied
* ``"A_width_fill"``  – interface edge replaced with outer + ref_width
* ``"B_interp"``      – outer edge linearly interpolated (≤ max_interp_ms)
* ``"B_width_fill"``  – outer edge replaced with interface − ref_width
* ``"unrecoverable"`` – both edges bad; positions set to NaN
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FLAG_NONE = "none"
_FLAG_A = "A_width_fill"
_FLAG_B_INTERP = "B_interp"
_FLAG_B_FILL = "B_width_fill"
_FLAG_UNRECOVERABLE = "unrecoverable"


def _find_run_slices(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end), ...] of True runs in *mask* (end is exclusive)."""
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _grow_runs(
    initial: np.ndarray,
    deviation: np.ndarray,
    low_threshold: float,
    max_grow: int,
) -> np.ndarray:
    """Hysteresis growth: expand each flagged run while deviation > low_threshold.

    Parameters
    ----------
    initial:
        Boolean mask of initially flagged frames.
    deviation:
        Non-negative deviation values used for the low threshold test.
    low_threshold:
        Frames adjacent to a flagged run are added while their deviation
        exceeds this value.
    max_grow:
        Maximum number of frames added on each side of a run.
    """
    n = len(initial)
    grown = initial.copy()
    for start, end in _find_run_slices(initial):
        # Grow left
        count = 0
        i = start - 1
        while i >= 0 and count < max_grow and deviation[i] > low_threshold:
            grown[i] = True
            i -= 1
            count += 1
        # Grow right
        count = 0
        i = end
        while i < n and count < max_grow and deviation[i] > low_threshold:
            grown[i] = True
            i += 1
            count += 1
    return grown


def _pad_runs(mask: np.ndarray, pad: int) -> np.ndarray:
    """Add *pad* frames around each True run."""
    if pad <= 0:
        return mask.copy()
    n = len(mask)
    padded = mask.copy()
    for start, end in _find_run_slices(mask):
        lo = max(0, start - pad)
        hi = min(n, end + pad)
        padded[lo:hi] = True
    return padded


def _merge_close_runs(mask: np.ndarray, merge: int) -> np.ndarray:
    """Fill gaps of ≤ merge frames between adjacent True runs."""
    if merge <= 0:
        return mask.copy()
    n = len(mask)
    merged = mask.copy()
    runs = _find_run_slices(mask)
    for (_, end1), (start2, _) in zip(runs, runs[1:]):
        gap = start2 - end1
        if gap <= merge:
            merged[end1:start2] = True
    return merged


def _ref_width_per_state(
    widths: np.ndarray,
    state_codes: np.ndarray,
    state_guard_nm: float,
) -> np.ndarray:
    """Compute a per-frame reference width using per-state median.

    If a state's median differs from the whole-run median by more than
    *state_guard_nm*, the whole-run median is used for that state.

    Parameters
    ----------
    widths:
        Array of width values (may contain NaN for sentinel frames).
    state_codes:
        Integer state code per frame.
    state_guard_nm:
        Maximum allowed deviation of a state's median from the global median
        before falling back to the global median.
    """
    finite = np.isfinite(widths)
    global_med = float(np.nanmedian(widths[finite])) if finite.any() else 0.0

    ref = np.full(len(widths), global_med, dtype=float)
    for sc in np.unique(state_codes):
        mask = (state_codes == sc) & finite
        if mask.sum() < 3:
            continue
        state_med = float(np.median(widths[mask]))
        if abs(state_med - global_med) <= state_guard_nm:
            ref[state_codes == sc] = state_med
        # else: keep global_med for this state
    return ref


def _detect_bad_frames(
    widths: np.ndarray,
    ref_width: np.ndarray,
    interface_edge: np.ndarray,
    sentinel: float,
    k_mad: float,
    floor_nm: float,
    k_mad_low: float,
    max_grow_frames: int,
    pad_frames: int,
    merge_frames: int,
) -> tuple[np.ndarray, float]:
    """Return (bad_mask, mad_of_widths).

    bad_mask is True for frames whose interface edge is the sentinel OR whose
    |width - ref_width| exceeds the detection threshold, after hysteresis
    growth and optional padding.
    """
    # Exclude sentinel frames from MAD calculation
    sentinel_mask = np.isclose(interface_edge, sentinel, atol=0.01, rtol=0.0)
    finite_width = ~sentinel_mask & np.isfinite(widths) & np.isfinite(ref_width)

    if finite_width.any():
        deviations = np.abs(widths[finite_width] - ref_width[finite_width])
        mad = float(np.median(deviations))
    else:
        mad = 0.0

    high_thresh = max(k_mad * 1.4826 * mad, floor_nm)
    low_thresh = max(k_mad_low * 1.4826 * mad, 50.0)

    deviation_all = np.abs(widths - ref_width)
    deviation_all[~np.isfinite(deviation_all)] = high_thresh + 1.0  # treat NaN as bad

    initial_bad = sentinel_mask | (deviation_all > high_thresh)
    grown = _grow_runs(initial_bad, deviation_all, low_thresh, max_grow_frames)
    if pad_frames > 0:
        grown = _pad_runs(grown, pad_frames)
    if merge_frames > 0:
        grown = _merge_close_runs(grown, merge_frames)

    return grown, mad


# ---------------------------------------------------------------------------
# Per-trap correction
# ---------------------------------------------------------------------------

def _correct_one_trap(
    outer: np.ndarray,
    interface: np.ndarray,
    state_codes: np.ndarray,
    time_s: np.ndarray,
    sentinel: float,
    state_guard_nm: float,
    k_mad: float,
    floor_nm: float,
    k_mad_low: float,
    max_grow_frames: int,
    pad_frames: int,
    max_interp_ms: float,
    merge_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct one trap's edges.

    Bad-frame detection uses the width (interface − outer) deviation, which
    responds to a failure in *either* edge.  After growing and padding, each
    bad frame is classified by comparing the per-edge deviations from their own
    independent per-state references:

    * sentinel on interface → A_width_fill
    * outer further from its own reference than interface → B_interp / B_width_fill
    * otherwise (interface further, or equal) → A_width_fill
    * both sentinel / both large deviation → unrecoverable

    Returns
    -------
    outer_corr, interface_corr, flag_arr
        Corrected edge arrays and per-frame flag strings.
    """
    n = len(outer)
    outer_corr = outer.copy().astype(float)
    interface_corr = interface.copy().astype(float)
    flag_arr = np.full(n, _FLAG_NONE, dtype=object)

    sc = state_codes.astype(int)

    # ------------------------------------------------------------------ #
    # Step 1: per-state references for EACH edge and for the width        #
    # ------------------------------------------------------------------ #
    # Outer reference: exclude obvious sentinel values
    sentinel_outer_mask = np.isclose(outer_corr, sentinel, atol=0.01, rtol=0.0)
    outer_for_ref = outer_corr.copy()
    outer_for_ref[sentinel_outer_mask] = np.nan
    outer_ref = _ref_width_per_state(outer_for_ref, sc, state_guard_nm)

    # Interface reference: exclude sentinel values from the reference calc
    sentinel_iface_mask = np.isclose(interface_corr, sentinel, atol=0.01, rtol=0.0)
    iface_for_ref = interface_corr.copy()
    iface_for_ref[sentinel_iface_mask] = np.nan
    iface_ref = _ref_width_per_state(iface_for_ref, sc, state_guard_nm)

    # Width = interface − outer; per-state reference width
    widths = interface_corr - outer_corr
    ref_width = iface_ref - outer_ref  # per-frame reference width

    # ------------------------------------------------------------------ #
    # Step 2: unified bad-frame detection (width deviation or sentinel)   #
    # ------------------------------------------------------------------ #
    bad_any, _mad = _detect_bad_frames(
        widths, ref_width, interface_corr,
        sentinel, k_mad, floor_nm, k_mad_low,
        max_grow_frames, pad_frames, merge_frames,
    )

    # ------------------------------------------------------------------ #
    # Step 3: classify each bad frame as interface-bad or outer-bad       #
    # ------------------------------------------------------------------ #
    outer_thresh = max(k_mad * 1.4826 * _mad, floor_nm) if _mad > 0 else floor_nm

    outer_dev = np.abs(outer_corr - outer_ref)
    iface_dev = np.abs(interface_corr - iface_ref)

    # outer_bad_point: outer is sentinel OR outer deviates more than threshold
    outer_bad_point = sentinel_outer_mask | (outer_dev > outer_thresh)
    # interface_bad_point: interface is sentinel OR interface deviates more than
    # threshold AND outer is clean (so outer did NOT cause the width deviation)
    iface_bad_point = sentinel_iface_mask | (
        (iface_dev > outer_thresh) & ~outer_bad_point
    )

    # ------------------------------------------------------------------ #
    # Step 4: apply corrections per bad run                               #
    # ------------------------------------------------------------------ #
    for start, end in _find_run_slices(bad_any):
        run_outer_bad = outer_bad_point[start:end]
        run_iface_bad = iface_bad_point[start:end]

        # Duration of this run
        t_start = time_s[start]
        t_end = time_s[min(end - 1, n - 1)]
        duration_ms = (t_end - t_start) * 1000.0

        for i, gi in enumerate(range(start, end)):
            o_bad = run_outer_bad[i]
            a_bad = run_iface_bad[i]

            if o_bad and a_bad:
                # Both edges bad → unrecoverable
                outer_corr[gi] = np.nan
                interface_corr[gi] = np.nan
                flag_arr[gi] = _FLAG_UNRECOVERABLE
            elif a_bad:
                # Interface bad only → A_width_fill
                interface_corr[gi] = outer_corr[gi] + ref_width[gi]
                flag_arr[gi] = _FLAG_A
            else:
                # Outer bad (or width deviation without clear interface failure) →
                # treat as outer-bad; B_interp if short, B_width_fill if long
                if duration_ms <= max_interp_ms:
                    flag_arr[gi] = _FLAG_B_INTERP  # value filled below
                else:
                    outer_corr[gi] = interface_corr[gi] - ref_width[gi]
                    flag_arr[gi] = _FLAG_B_FILL

        # Fill B_interp values for the whole run in one pass
        if _FLAG_B_INTERP in flag_arr[start:end]:
            left_val = outer_corr[start - 1] if start > 0 else np.nan
            # find right bracket outside the bad run
            right_idx = end
            while right_idx < n and flag_arr[right_idx] != _FLAG_NONE:
                right_idx += 1
            right_val = outer_corr[right_idx] if right_idx < n else np.nan

            interp_indices = [gi for gi in range(start, end)
                              if flag_arr[gi] == _FLAG_B_INTERP]
            if interp_indices:
                run_len = len(interp_indices)
                if np.isfinite(left_val) and np.isfinite(right_val):
                    interp_vals = np.linspace(left_val, right_val,
                                              run_len + 2)[1:-1]
                elif np.isfinite(left_val):
                    interp_vals = np.full(run_len, left_val)
                elif np.isfinite(right_val):
                    interp_vals = np.full(run_len, right_val)
                else:
                    interp_vals = np.full(run_len, np.nan)
                for k, gi in enumerate(interp_indices):
                    outer_corr[gi] = interp_vals[k]

    return outer_corr, interface_corr, flag_arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def correct_edge_tracking(
    df: pd.DataFrame,
    sentinel: float = -30.27,
    ref_mode: str = "state",
    state_guard_nm: float = 50.0,
    k_mad: float = 5.0,
    floor_nm: float = 100.0,
    k_mad_low: float = 3.0,
    max_grow_frames: int = 30,
    pad_frames: int = 0,
    max_interp_ms: float = 50.0,
    b_fallback: str = "width",
    win_s: float = 1.0,
    loc_win: int = 101,
    merge_frames: int = 0,
) -> pd.DataFrame:
    """Detect and correct edge-tracking failures.

    Parameters
    ----------
    df:
        Tidy DataFrame containing ``edge1_raw_nm`` … ``edge4_raw_nm``,
        ``state_code``, ``time_s``, ``trap1_pos_nm``, ``trap2_pos_nm``,
        and ``extension_nm``.  If the raw edge columns are absent (e.g. for
        6-column files), the DataFrame is returned unchanged.
    sentinel:
        Sentinel value that marks a failed interface-edge read.
    ref_mode:
        ``"state"`` (default) – use per-state median width as the reference.
        Any other value – use the global run median.
    state_guard_nm:
        Maximum deviation of a state's width median from the global median
        before falling back to the global median.
    k_mad:
        Multiplier for the detection threshold: high = max(k_mad × 1.4826 × MAD, floor_nm).
    floor_nm:
        Minimum detection threshold (nm).
    k_mad_low:
        Multiplier for the hysteresis growth threshold: max(k_mad_low × 1.4826 × MAD, 50 nm).
    max_grow_frames:
        Maximum number of frames added on each side during hysteresis growth.
    pad_frames:
        Additional frames added around each flagged run after hysteresis.
    max_interp_ms:
        Maximum duration (ms) of an outer-edge error for B_interp; longer
        errors use B_width_fill.
    b_fallback:
        Fallback strategy for long outer-edge errors: ``"width"`` (default)
        sets outer = interface − ref_width.
    win_s:
        Window size in seconds used by the prototype's local-reference
        computation.  Reserved for future use; not currently used.
    loc_win:
        Local-window size in samples used by the prototype.  Reserved for
        future use; not currently used.
    merge_frames:
        Merge adjacent flagged runs separated by ≤ this many frames.

    Returns
    -------
    DataFrame with corrected ``trap1_pos_nm``, ``trap2_pos_nm``,
    ``extension_nm``, and new columns ``edge1_raw_nm`` … ``edge4_raw_nm``
    (raw, unchanged), ``edge_flag_trap1``, ``edge_flag_trap2``, ``edge_flag``.
    Frames that were not flagged are bit-identical to the input.
    """
    EDGE_COLS = ("edge1_raw_nm", "edge2_raw_nm", "edge3_raw_nm", "edge4_raw_nm")
    missing = any(c not in df.columns for c in EDGE_COLS)
    all_na = False
    if not missing:
        edge_values = df.loc[:, EDGE_COLS].to_numpy(dtype=float)
        all_na = np.all(np.isnan(edge_values))
    if missing or all_na:
        # 6-column file, missing edge columns, or all-NaN edge data – pass through unchanged
        logger.debug("correct_edge_tracking: edge columns absent/NaN; skipping.")
        return df

    df = df.copy()
    n = len(df)

    e1 = df["edge1_raw_nm"].to_numpy(dtype=float)  # trap1 outer
    e2 = df["edge2_raw_nm"].to_numpy(dtype=float)  # trap2 outer
    e3 = df["edge3_raw_nm"].to_numpy(dtype=float)  # trap1 interface
    e4 = df["edge4_raw_nm"].to_numpy(dtype=float)  # trap2 interface
    state_codes = df["state_code"].to_numpy(dtype=float)
    time_s = df["time_s"].to_numpy(dtype=float)

    # Use integer state codes; NaN states become 0
    sc_int = np.where(np.isfinite(state_codes), state_codes.astype(int), 0)

    # If ref_mode is not "state", force the whole-run median (treat all as state 0)
    if ref_mode != "state":
        sc_int = np.zeros(n, dtype=int)

    # --- Trap 1 ---
    e1_corr, e3_corr, flag_t1 = _correct_one_trap(
        outer=e1, interface=e3,
        state_codes=sc_int, time_s=time_s,
        sentinel=sentinel,
        state_guard_nm=state_guard_nm,
        k_mad=k_mad, floor_nm=floor_nm, k_mad_low=k_mad_low,
        max_grow_frames=max_grow_frames, pad_frames=pad_frames,
        max_interp_ms=max_interp_ms, merge_frames=merge_frames,
    )

    # --- Trap 2 ---
    e2_corr, e4_corr, flag_t2 = _correct_one_trap(
        outer=e2, interface=e4,
        state_codes=sc_int, time_s=time_s,
        sentinel=sentinel,
        state_guard_nm=state_guard_nm,
        k_mad=k_mad, floor_nm=floor_nm, k_mad_low=k_mad_low,
        max_grow_frames=max_grow_frames, pad_frames=pad_frames,
        max_interp_ms=max_interp_ms, merge_frames=merge_frames,
    )

    # --- Update positions ---
    trap1_new = (e1_corr + e3_corr) / 2.0
    trap2_new = (e2_corr + e4_corr) / 2.0
    ext_new = trap1_new - trap2_new

    df["trap1_pos_nm"] = trap1_new
    df["trap2_pos_nm"] = trap2_new
    df["extension_nm"] = ext_new

    # --- Composite flag ---
    flag_overall = np.full(n, _FLAG_NONE, dtype=object)
    for i in range(n):
        t1 = flag_t1[i]
        t2 = flag_t2[i]
        if t1 == _FLAG_UNRECOVERABLE or t2 == _FLAG_UNRECOVERABLE:
            flag_overall[i] = _FLAG_UNRECOVERABLE
        elif t1 != _FLAG_NONE:
            flag_overall[i] = t1
        elif t2 != _FLAG_NONE:
            flag_overall[i] = t2

    df["edge_flag_trap1"] = flag_t1
    df["edge_flag_trap2"] = flag_t2
    df["edge_flag"] = flag_overall

    # --- Validate: mark unrecoverable frames as invalid ---
    if "valid" in df.columns:
        unrec_mask = flag_overall == _FLAG_UNRECOVERABLE
        valid = df["valid"].to_numpy(dtype=bool).copy()
        valid[unrec_mask] = False
        df["valid"] = valid

    # Log summary
    n_corrected_t1 = int((flag_t1 != _FLAG_NONE).sum())
    n_corrected_t2 = int((flag_t2 != _FLAG_NONE).sum())
    if n_corrected_t1 or n_corrected_t2:
        logger.info(
            "Edge correction: trap1 %d frames, trap2 %d frames (total %d of %d, %.2f%%)",
            n_corrected_t1,
            n_corrected_t2,
            n_corrected_t1 + n_corrected_t2,
            n,
            100.0 * (n_corrected_t1 + n_corrected_t2) / n if n else 0.0,
        )

    return df


def summarize_edge_correction(df: pd.DataFrame) -> dict[str, Any]:
    """Return counts of each correction method per trap and overall.

    Parameters
    ----------
    df:
        DataFrame after :func:`correct_edge_tracking`.

    Returns
    -------
    dict with keys like ``trap1_A_width_fill``, ``trap2_B_interp``, etc.,
    plus ``total_corrected`` and ``fraction_corrected``.
    """
    result: dict[str, Any] = {}
    n = len(df)
    total_corrected = 0

    for trap_label, col in [("trap1", "edge_flag_trap1"), ("trap2", "edge_flag_trap2")]:
        if col not in df.columns:
            continue
        flags = df[col].to_numpy(dtype=object)
        for flag_val in (_FLAG_A, _FLAG_B_INTERP, _FLAG_B_FILL, _FLAG_UNRECOVERABLE):
            count = int((flags == flag_val).sum())
            key = f"{trap_label}_{flag_val}"
            result[key] = count
            if flag_val != _FLAG_NONE:
                total_corrected += count

    result["total_corrected"] = total_corrected
    result["n_frames"] = n
    result["fraction_corrected"] = total_corrected / n if n else 0.0
    return result
