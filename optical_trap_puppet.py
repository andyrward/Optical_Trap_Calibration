"""Puppet-mode optical-trap reader.

Public API
----------
read_puppet(path, k=None, stiffness=None, power=None, ...) -> pd.DataFrame
    Load a puppet-mode run as a tidy per-frame DataFrame.

Protocol
--------
A puppet-mode run drives the trap voltage through programmed hold–ramp cycles.
The voltage source is resolved in this order:

1. A companion ``*_Voltage.dat`` file (``power, voltage`` pairs per sample), if
   present alongside the main ``.dat`` file.
2. Voltage/power columns embedded in the main file (newer instrument format)
   when they are present.

If no voltage source can be found, ``read_puppet`` raises :class:`ValueError`.

Column mapping (matches the MATLAB toolbox)
-------------------------------------------
*  ``trap1_nm``    = (col2 + col4) / 2
*  ``trap2_nm``    = (col3 + col5) / 2
*  ``extension_nm``= trap2 − trap1  (signed; more negative = longer tether)

Sentinel −30.27 in any source column marks the frame ``valid = False``;
the frame is kept in the table.

Phase segmentation
------------------
The voltage trace is smoothed and differentiated to classify each frame as:

* ``hold``     – |dV/dt| ≤ *dv_threshold*
* ``pull_out`` – dV/dt < −*dv_threshold*  (falling V → increasing extension)
* ``pull_in``  – dV/dt > +*dv_threshold*  (rising V → decreasing extension)

Contiguous runs of the same phase are given a monotonically increasing
``segment_id``.  ``hold_voltage`` records the mean voltage of each hold
segment (NaN for ramp segments).

Force
-----
``force_pn = (trap1_nm − force_ref_nm) × k_eff``

*k_eff* is resolved as:

* ``k`` directly (e.g. a provisional ``3.93e-5 pN/nm/mW × power``),
* or ``stiffness × power`` (per-sample if *power* is an array).

For each pull segment the reference ``force_ref_nm`` is the mean ``trap1_nm``
over the immediately preceding hold segment.  The reference is propagated
forward over the entire following ramp and then updated at the next hold.

Rupture detection (optional)
-----------------------------
During ``pull_out`` segments, a rupture is detected when
``|Δtrap1_nm| > rupture_threshold_nm`` between consecutive valid frames.
The ``ruptured`` column is set to ``True`` for the first frame *after* the
drop and all subsequent frames within that pull-out segment.

Output columns
--------------
``frame_idx``, ``time_s``, ``trap1_nm``, ``trap2_nm``, ``extension_nm``,
``voltage_v``, ``power_mw``, ``valid``, ``segment_id``, ``phase``,
``hold_voltage``, ``k_eff``, ``force_ref_nm``, ``force_pn``, ``ruptured``,
``source``
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from optical_trap_toolbox import _load_numeric_table, _load_voltage_power_series, _align_series

# Sentinel value used by the MATLAB data-acquisition software to indicate a
# lost-edge detection event (bead out of field of view).
LOST_EDGE_SENTINEL: float = -30.27

# Tolerance when comparing raw sample values to the sentinel.
_SENTINEL_TOL: float = 0.01


def _flag_sentinel(data: np.ndarray, sentinel: float = LOST_EDGE_SENTINEL, tol: float = _SENTINEL_TOL) -> np.ndarray:
    """Return a boolean array that is True where no column equals *sentinel*."""
    return ~np.any(np.abs(data - sentinel) < tol, axis=1)


def _smooth_diff(v: np.ndarray, window: int) -> np.ndarray:
    """Return the first-difference of a smoothed copy of *v*.

    The smoothing window is clamped so it never exceeds the array length.
    """
    w = max(1, min(window, len(v)))
    vs = uniform_filter1d(v.astype(float), size=w, mode="nearest")
    dv = np.empty_like(vs)
    dv[0] = 0.0
    dv[1:] = np.diff(vs)
    return dv


def _classify_phase(dv: np.ndarray, threshold: float) -> np.ndarray:
    """Classify each sample as 'hold', 'pull_out', or 'pull_in'.

    Returns an object array of strings.
    """
    phase = np.full(len(dv), "hold", dtype=object)
    phase[dv < -threshold] = "pull_out"
    phase[dv > +threshold] = "pull_in"
    return phase


def _build_segments(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign a monotonically increasing segment ID to contiguous same-phase runs.

    Returns
    -------
    segment_id : int array
    segment_phase : object array (parallel to *phase*, giving the canonical phase
        label for each segment determined by the majority label within the run)
    """
    n = len(phase)
    seg_id = np.empty(n, dtype=np.int64)
    boundaries = np.concatenate(([0], np.where(phase[:-1] != phase[1:])[0] + 1, [n]))
    for sid, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        seg_id[start:end] = sid
    return seg_id, phase


def _hold_voltage_array(phase: np.ndarray, voltage: np.ndarray, seg_id: np.ndarray) -> np.ndarray:
    """Return per-sample hold voltage (NaN for ramp samples)."""
    hv = np.full(len(phase), np.nan)
    for sid in np.unique(seg_id):
        mask = seg_id == sid
        if phase[mask][0] == "hold":
            hv[mask] = np.nanmean(voltage[mask])
    return hv


def _assign_force_refs(
    trap1: np.ndarray,
    phase: np.ndarray,
    seg_id: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute per-sample force reference from the preceding hold.

    For each hold segment the reference is the mean trap1 of valid frames.
    That reference is then propagated forward until the next hold's reference
    becomes available.
    """
    n = len(trap1)
    refs = np.full(n, np.nan)
    last_ref = np.nan

    for sid in np.unique(seg_id):
        mask = seg_id == sid
        if phase[mask][0] == "hold":
            valid_t1 = trap1[mask & valid]
            last_ref = np.nanmean(valid_t1) if valid_t1.size > 0 else np.nan
        refs[mask] = last_ref
    return refs


def _detect_rupture(
    trap1: np.ndarray,
    phase: np.ndarray,
    seg_id: np.ndarray,
    valid: np.ndarray,
    threshold_nm: float,
) -> np.ndarray:
    """Set ``ruptured = True`` from the first large drop in trap1 onward within
    each ``pull_out`` segment.
    """
    ruptured = np.zeros(len(trap1), dtype=bool)
    for sid in np.unique(seg_id):
        mask_idx = np.where(seg_id == sid)[0]
        if phase[mask_idx[0]] != "pull_out":
            continue
        # Restrict to valid frames only for computing differences, but flag
        # ALL frames in the segment from the rupture point onward.
        t1 = trap1[mask_idx]
        v_mask = valid[mask_idx]
        dt1 = np.diff(t1)
        # A rupture is a sudden large decrease in trap1
        drops = np.where(v_mask[1:] & (dt1 < -threshold_nm))[0]
        if drops.size > 0:
            first_drop = drops[0] + 1  # index within seg
            ruptured[mask_idx[first_drop:]] = True
    return ruptured


def read_puppet(
    path: str | Path,
    k: float | None = None,
    stiffness: float | None = None,
    power: float | Sequence[float] | None = None,
    *,
    companion_voltage_path: str | Path | None = None,
    force_ref_nm: float | None = None,
    smooth_window: int = 500,
    dv_threshold: float = 1e-4,
    rupture_threshold_nm: float = 500.0,
    source: str | None = None,
) -> pd.DataFrame:
    """Load a puppet-mode optical-trap run as a tidy per-frame DataFrame.

    Parameters
    ----------
    path:
        Path to the ``.dat`` file (7-column two-trap MATLAB format).
    k:
        Effective stiffness in pN/nm (flat scalar).  If provided, overrides
        *stiffness* × *power*.
    stiffness:
        Trap-1 stiffness in pN/nm/mW.  Combined with *power* to give k_eff.
    power:
        Laser power in mW.  Scalar or per-sample array.  If ``None``, the
        companion ``*_Voltage.dat`` file must supply power values.
    companion_voltage_path:
        Explicit path to the companion voltage/power file.  If ``None``, the
        loader looks for ``<stem>_Voltage.dat`` next to *path*.
    force_ref_nm:
        Optional fixed force reference in nm. When supplied, ``force_pn`` is
        computed as ``(trap1_nm - force_ref_nm) * k_eff`` for every row instead
        of using the automatic preceding-hold reference.
    smooth_window:
        Width (samples) of the uniform smoothing kernel applied to the voltage
        trace before differentiation for segment detection.
    dv_threshold:
        |dV/dt| threshold (V/sample) separating holds from ramps.
    rupture_threshold_nm:
        Minimum single-frame drop in ``trap1_nm`` (nm) during a ``pull_out``
        segment that is flagged as a rupture.
    source:
        Run identifier stored in the ``source`` column.  Defaults to the
        file stem.

    Returns
    -------
    pd.DataFrame with one row per raw sample frame.

    Raises
    ------
    ValueError
        If no voltage source (companion file or embedded columns) can be found.
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if source is None:
        source = path.stem

    # ------------------------------------------------------------------
    # 1.  Load raw data
    # ------------------------------------------------------------------
    data = _load_numeric_table(path)
    n_rows, n_cols = data.shape

    if n_cols < 5:
        raise ValueError(
            f"{path.name}: expected at least 5 columns (7-column puppet format); "
            f"got {n_cols}."
        )

    # ------------------------------------------------------------------
    # 2.  Column mapping
    # ------------------------------------------------------------------
    trap1 = (data[:, 1] + data[:, 3]) / 2.0   # (col2 + col4) / 2
    trap2 = (data[:, 2] + data[:, 4]) / 2.0   # (col3 + col5) / 2
    extension = trap2 - trap1                   # signed

    # ------------------------------------------------------------------
    # 3.  Sentinel flagging (−30.27 in any source column)
    # ------------------------------------------------------------------
    # Check columns 1–4 (the four edge detectors) and col 0 (time) is excluded.
    edge_cols = data[:, 1:5]  # cols 2–5 (1-indexed)
    valid = _flag_sentinel(edge_cols)

    # ------------------------------------------------------------------
    # 4.  Voltage / power
    # ------------------------------------------------------------------
    voltage_v: np.ndarray
    power_mw: np.ndarray

    # Try companion *_Voltage.dat first
    if companion_voltage_path is not None:
        volt_path = Path(companion_voltage_path)
    else:
        volt_path = path.with_name(f"{path.stem}_Voltage.dat")

    if volt_path.exists():
        series = _load_voltage_power_series(volt_path)
        if "voltage_v" in series:
            voltage_v = _align_series(series["voltage_v"], n_rows)
        else:
            voltage_v = np.full(n_rows, np.nan)
        if "power_mw" in series:
            power_mw = _align_series(series["power_mw"], n_rows)
        else:
            power_mw = np.full(n_rows, np.nan)
    else:
        # Try embedded voltage from a newer-format main file.
        # In the 7-column format the last two columns are state & actuation;
        # a newer format may use additional columns for power and voltage.
        # We check whether n_cols > 7 and treat col8/col9 as power/voltage.
        if n_cols >= 9:
            power_mw = _align_series(data[:, 7], n_rows)
            voltage_v = _align_series(data[:, 8], n_rows)
        elif n_cols == 8:
            # Single extra column: treat as voltage
            voltage_v = _align_series(data[:, 7], n_rows)
            power_mw = np.full(n_rows, np.nan)
        else:
            voltage_v = np.full(n_rows, np.nan)
            power_mw = np.full(n_rows, np.nan)

    # Apply caller-supplied power override
    if power is not None:
        power_arr = np.asarray(power, dtype=float)
        power_mw = _align_series(power_arr if power_arr.ndim > 0 else np.array([float(power)]), n_rows)

    # Validate: we must have a real voltage trace
    if np.all(np.isnan(voltage_v)):
        raise ValueError(
            f"{path.name}: no voltage source found. "
            "Provide a companion '*_Voltage.dat' file or use a newer-format "
            "main file with embedded voltage columns."
        )

    # ------------------------------------------------------------------
    # 5.  Phase segmentation
    # ------------------------------------------------------------------
    dv = _smooth_diff(voltage_v, smooth_window)
    phase_arr = _classify_phase(dv, dv_threshold)
    seg_id, _ = _build_segments(phase_arr)
    hold_voltage = _hold_voltage_array(phase_arr, voltage_v, seg_id)

    # ------------------------------------------------------------------
    # 6.  Bead-radius correction diagnostics
    # ------------------------------------------------------------------
    # Keep the raw extension intact for auditability.  For a first-pass estimate,
    # compute each bead radius from the median valid outer-to-inner edge spacing.
    valid_mask = valid & np.isfinite(data[:, 1:5]).all(axis=1)
    bead1_outer = data[:, 1]
    bead1_inner = data[:, 3]
    bead2_outer = data[:, 2]
    bead2_inner = data[:, 4]

    bead1_diameter_nm = np.abs(np.median(bead1_inner[valid_mask] - bead1_outer[valid_mask])) if np.any(valid_mask) else np.nan
    bead2_diameter_nm = np.abs(np.median(bead2_inner[valid_mask] - bead2_outer[valid_mask])) if np.any(valid_mask) else np.nan
    bead1_radius_nm = bead1_diameter_nm / 2.0
    bead2_radius_nm = bead2_diameter_nm / 2.0
    extension_corrected = extension - bead1_radius_nm - bead2_radius_nm

    # ------------------------------------------------------------------
    # 7.  Force computation
    # ------------------------------------------------------------------
    # Resolve effective stiffness
    if k is not None:
        k_eff = np.full(n_rows, float(k))
    elif stiffness is not None:
        k_eff = stiffness * power_mw
    else:
        k_eff = np.full(n_rows, np.nan)

    # Dual-zero reference: preceding hold mean, unless a fixed reference is supplied.
    if force_ref_nm is not None:
        force_ref = np.full(n_rows, float(force_ref_nm), dtype=float)
    else:
        force_ref = _assign_force_refs(trap1, phase_arr, seg_id, valid)

    with np.errstate(invalid="ignore"):
        force_pn = (trap1 - force_ref) * k_eff

    # ------------------------------------------------------------------
    # 8.  Rupture detection
    # ------------------------------------------------------------------
    ruptured = _detect_rupture(trap1, phase_arr, seg_id, valid, rupture_threshold_nm)

    # ------------------------------------------------------------------
    # 9.  Build tidy DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "source": source,
            "frame_idx": np.arange(n_rows, dtype=np.int64),
            "time_s": data[:, 0],
            "trap1_nm": trap1,
            "trap2_nm": trap2,
            "extension_nm": extension,
            "extension_nm_corrected": extension_corrected,
            "voltage_v": voltage_v,
            "power_mw": power_mw,
            "valid": valid,
            "segment_id": seg_id,
            "phase": phase_arr,
            "hold_voltage": hold_voltage,
            "bead1_diameter_nm": np.full(n_rows, bead1_diameter_nm, dtype=float),
            "bead2_diameter_nm": np.full(n_rows, bead2_diameter_nm, dtype=float),
            "bead1_radius_nm": np.full(n_rows, bead1_radius_nm, dtype=float),
            "bead2_radius_nm": np.full(n_rows, bead2_radius_nm, dtype=float),
            "k_eff": k_eff,
            "force_ref_nm": force_ref,
            "force_pn": force_pn,
            "ruptured": ruptured,
        }
    )
    return df
