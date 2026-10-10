from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

NORMALIZED_SCHEMA = (
    "time_s",
    "position_1_nm",
    "position_2_nm",
    "extension_nm",
    "force_pn",
    "voltage_v",
    "stiffness_pn_per_nm",
    "state_code",
    "power_mw",
)


STATE_CODE_HINTS: dict[int, str] = {
    3: "oscillator_mode_three",
    4: "oscillator_mode_four",
    21: "tether_attempt_or_contact",
    24: "reference_pull",
    27: "force_extension",
    # MATLAB DualTrap workflows treat state 31 as a second FE-like subtether run.
    31: "force_extension_subtether",
}


@dataclass(frozen=True)
class StateRun:
    start_idx: int
    end_idx: int
    state_code: int


@dataclass(frozen=True)
class ExperimentSegment:
    run: StateRun
    label: str
    confidence: str


def _load_numeric_table(filepath: str | Path) -> np.ndarray:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    errors: list[Exception] = []
    for delimiter in (None, "\t", " ", ","):
        try:
            data = np.loadtxt(path, dtype=float, delimiter=delimiter, ndmin=2)
        except ValueError as exc:
            errors.append(exc)
            continue
        if data.size == 0:
            raise ValueError(f"No data found in file: {path}")
        return data

    detail = "; ".join(str(err) for err in errors)
    raise ValueError(f"Could not parse numeric data in {path}: {detail}")


def _load_voltage_power_series(filepath: str | Path) -> dict[str, np.ndarray]:
    path = Path(filepath)
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    if not text.strip():
        return {}

    # MATLAB exports in this repository are ASCII decimal text with comma/space/tab/newline delimiters.
    normalized = text.replace("\n", ",").replace("\t", ",").replace(" ", ",")
    values = [token for token in normalized.split(",") if token]
    numeric_values: list[float] = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except ValueError:
            continue
    if not numeric_values:
        raise ValueError(f"Voltage file contains no numeric values: {path}")
    numeric = np.array(numeric_values, dtype=float)

    if numeric.size == 1:
        # Single token files are interpreted as scalar trap power only.
        return {"power_mw": numeric}

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    per_line_counts = [
        len([token for token in re.split(r"[,\t ]+", line) if token])
        for line in lines
    ]
    looks_like_two_column_rows = bool(per_line_counts) and all(count == 2 for count in per_line_counts)
    looks_like_single_value_rows = bool(per_line_counts) and all(count == 1 for count in per_line_counts)
    looks_like_single_line_stream = len(lines) == 1 and "," in lines[0]
    alternating_stream = (
        (len(lines) >= 2 and looks_like_two_column_rows)
        or looks_like_single_line_stream
    )
    if alternating_stream:
        if numeric.size % 2 != 0:
            raise ValueError(
                f"Voltage file has an odd number of values ({numeric.size}) and cannot be parsed as paired power/voltage rows: {path}"
            )
        power_values = numeric[0::2]
        voltage_values = numeric[1::2]
        # MATLAB _Voltage.dat streams in this repository follow power,voltage ordering.
        return {"power_mw": power_values, "voltage_v": voltage_values}

    if looks_like_single_value_rows and numeric.size >= 2:
        return {"power_mw": numeric}

    return {"power_mw": numeric}


def _align_series(series: np.ndarray, length: int) -> np.ndarray:
    series = np.asarray(series, dtype=float).reshape(-1)
    if length <= 0:
        return np.array([], dtype=float)
    if series.size == 0:
        return np.full(length, np.nan)
    if series.size == 1:
        return np.full(length, series[0], dtype=float)
    if series.size == length:
        return series

    source_x = np.linspace(0.0, 1.0, num=series.size)
    target_x = np.linspace(0.0, 1.0, num=length)
    return np.interp(target_x, source_x, series)


def load_normalized_trace(
    filepath: str | Path,
    *,
    companion_voltage_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    data = _load_numeric_table(filepath)
    n_rows, n_cols = data.shape
    trace: dict[str, np.ndarray] = {key: np.full(n_rows, np.nan) for key in NORMALIZED_SCHEMA}

    trace["time_s"] = data[:, 0]

    if n_cols == 6:
        # MATLAB stateful layout: [t, z, F, V, k, State]. Position channels are not present.
        trace["extension_nm"] = data[:, 1]
        trace["force_pn"] = data[:, 2]
        trace["voltage_v"] = data[:, 3]
        trace["stiffness_pn_per_nm"] = data[:, 4]
        trace["state_code"] = data[:, 5]
    elif n_cols == 7:
        # MATLAB two-trap layout used by existing calibration code:
        # [time, trap1_a, trap2_a, trap1_b, trap2_b, voltage_or_const, state_or_const]
        # Pairing follows the repo convention: trap1 = mean(col1, col3) and trap2 = mean(col2, col4).
        trace["position_1_nm"] = (data[:, 1] + data[:, 3]) / 2.0
        trace["position_2_nm"] = (data[:, 2] + data[:, 4]) / 2.0
        trace["voltage_v"] = data[:, 5]
        trace["state_code"] = data[:, 6]
        trace["extension_nm"] = trace["position_1_nm"] - trace["position_2_nm"]
    elif n_cols >= 2:
        positions = data[:, 1:]
        if positions.shape[1] == 1:
            trace["position_1_nm"] = positions[:, 0]
        elif positions.shape[1] == 2:
            trace["position_1_nm"] = positions[:, 0]
            trace["position_2_nm"] = positions[:, 1]
            trace["extension_nm"] = positions[:, 0] - positions[:, 1]
        elif positions.shape[1] == 4:
            # Repo convention for dual-trap data keeps the two trap centers interleaved:
            # trap1 = mean(col0, col2), trap2 = mean(col1, col3).
            trap1 = (positions[:, 0] + positions[:, 2]) / 2.0
            trap2 = (positions[:, 1] + positions[:, 3]) / 2.0
            trace["position_1_nm"] = trap1
            trace["position_2_nm"] = trap2
            trace["extension_nm"] = trap1 - trap2
        elif positions.shape[1] == 3:
            # A 3-value position block includes one explicit position pair plus a third axis/track,
            # so the first two columns remain the most stable dual-position representation.
            trace["position_1_nm"] = positions[:, 0]
            trace["position_2_nm"] = positions[:, 1]
            trace["extension_nm"] = positions[:, 0] - positions[:, 1]
        else:
            trace["position_1_nm"] = positions.mean(axis=1)
    else:
        raise ValueError(f"Unsupported column count for {filepath}: {n_cols}")

    data_path = Path(filepath)
    if companion_voltage_path is not None:
        voltage_path = Path(companion_voltage_path)
    else:
        voltage_path = data_path.with_name(f"{data_path.stem}_Voltage.dat")
    if voltage_path.exists():
        series = _load_voltage_power_series(voltage_path)
        for key, values in series.items():
            trace[key] = _align_series(values, n_rows)

    return trace


def segment_state_runs(state_codes: Sequence[float | int], state_code: int | None = None) -> list[StateRun]:
    values = np.asarray(state_codes, dtype=float)
    if values.size == 0:
        return []

    if state_code is not None:
        mask = ~np.isnan(values) & (values == state_code)
        indices = np.where(mask)[0]
        if indices.size == 0:
            return []
        groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        return [StateRun(int(group[0]), int(group[-1]), int(state_code)) for group in groups]

    runs: list[StateRun] = []
    run_start: int | None = None
    run_state: float | None = None
    for idx, value in enumerate(values):
        if np.isnan(value):
            if run_start is not None and run_state is not None:
                runs.append(StateRun(int(run_start), int(idx - 1), int(run_state)))
                run_start = None
                run_state = None
            continue

        if run_start is None:
            run_start = idx
            run_state = float(value)
            continue

        if not np.isnan(run_state) and int(value) != int(run_state):
            runs.append(StateRun(int(run_start), int(idx - 1), int(run_state)))
            run_start = idx
            run_state = float(value)

    if run_start is not None and run_state is not None:
        runs.append(StateRun(int(run_start), int(values.size - 1), int(run_state)))
    return runs


def infer_experiment_segments(state_codes: Sequence[float | int]) -> list[ExperimentSegment]:
    segments: list[ExperimentSegment] = []
    for run in segment_state_runs(state_codes):
        label = STATE_CODE_HINTS.get(run.state_code, f"unknown_state_{run.state_code}")
        confidence = "high" if run.state_code in STATE_CODE_HINTS else "low"
        segments.append(ExperimentSegment(run=run, label=label, confidence=confidence))
    return segments


def plot_state_vs_time(trace: dict[str, np.ndarray], segments: Iterable[StateRun] | None = None):
    """Plot state code versus time with optional segment boundary lines."""
    fig, ax = plt.subplots(figsize=(10, 3))
    time = trace["time_s"]
    state = trace["state_code"].astype(float)
    valid = ~np.isnan(state)
    ax.plot(time[valid], state[valid], lw=1.0)
    for run in segments or []:
        ax.axvline(time[run.start_idx], color="tab:orange", alpha=0.2)
        ax.axvline(time[run.end_idx], color="tab:orange", alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    ax.set_title("State vs Time")
    return fig


def plot_power_voltage(trace: dict[str, np.ndarray]):
    """Plot voltage and power time series from a normalized trace."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    time = trace["time_s"]
    axes[0].plot(time, trace["voltage_v"], lw=1.0)
    axes[0].set_ylabel("Voltage (V)")
    axes[1].plot(time, trace["power_mw"], lw=1.0)
    axes[1].set_ylabel("Power (mW)")
    axes[1].set_xlabel("Time (s)")
    return fig


def plot_force_extension(trace: dict[str, np.ndarray], segments: Iterable[StateRun] | None = None):
    """Plot force-extension data and optional segment overlays."""
    fig, ax = plt.subplots(figsize=(6, 5))
    extension = trace["extension_nm"]
    force = trace["force_pn"]
    valid = ~(np.isnan(extension) | np.isnan(force))
    if np.any(valid):
        ax.plot(extension[valid], force[valid], lw=1.0)
    indices = np.arange(extension.size)
    for run in segments or []:
        run_mask = (indices >= run.start_idx) & (indices <= run.end_idx) & valid
        if np.any(run_mask):
            ax.plot(extension[run_mask], force[run_mask], lw=1.5)
    ax.set_xlabel("Extension (nm)")
    ax.set_ylabel("Force (pN)")
    ax.set_title("Force-Extension")
    return fig


def plot_segment_boundaries(trace: dict[str, np.ndarray], segments: Sequence[StateRun]):
    """Plot contiguous state-run boundaries across time."""
    fig, ax = plt.subplots(figsize=(10, 3))
    time = trace["time_s"]
    if time.size:
        ax.set_xlim(time[0], time[-1])
    for run in segments:
        start_t = time[run.start_idx]
        end_t = time[run.end_idx]
        if np.isclose(start_t, end_t):
            ax.axvline(start_t, alpha=0.2)
        else:
            ax.axvspan(start_t, end_t, alpha=0.2)
        ax.text(
            start_t,
            0.02,
            str(run.state_code),
            rotation=90,
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Segment boundaries")
    return fig
