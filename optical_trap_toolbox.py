from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from optical_trap_calibration import load_voltage_file

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
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return {}

    normalized = text.replace("\n", ",").replace("\t", ",").replace(" ", ",")
    values = [token for token in normalized.split(",") if token]
    try:
        numeric = np.array([float(value) for value in values], dtype=float)
    except ValueError:
        numeric = np.array([load_voltage_file(path)], dtype=float)

    if numeric.size == 0:
        return {}

    if numeric.size == 1:
        return {"power_mw": numeric}

    if numeric.size % 2 == 0:
        return {"power_mw": numeric[0::2], "voltage_v": numeric[1::2]}

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
        trace["extension_nm"] = data[:, 1]
        trace["force_pn"] = data[:, 2]
        trace["voltage_v"] = data[:, 3]
        trace["stiffness_pn_per_nm"] = data[:, 4]
        trace["state_code"] = data[:, 5]
    elif n_cols == 7:
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
            trap1 = (positions[:, 0] + positions[:, 1]) / 2.0
            trap2 = (positions[:, 2] + positions[:, 3]) / 2.0
            trace["position_1_nm"] = trap1
            trace["position_2_nm"] = trap2
            trace["extension_nm"] = trap1 - trap2
        else:
            trace["position_1_nm"] = positions.mean(axis=1)
    else:
        raise ValueError(f"Unsupported column count for {filepath}: {n_cols}")

    data_path = Path(filepath)
    voltage_path = Path(companion_voltage_path) if companion_voltage_path else data_path.with_name(f"{data_path.stem}_Voltage.dat")
    if voltage_path.exists():
        series = _load_voltage_power_series(voltage_path)
        for key, values in series.items():
            trace[key] = _align_series(values, n_rows)

    if np.all(np.isnan(trace["power_mw"])) and voltage_path.exists():
        trace["power_mw"] = np.full(n_rows, load_voltage_file(voltage_path))

    return trace


def segment_state_runs(state_codes: Sequence[float | int], state_code: int | None = None) -> list[StateRun]:
    values = np.asarray(state_codes)
    if values.size == 0:
        return []

    if state_code is not None:
        mask = values == state_code
        indices = np.where(mask)[0]
        if indices.size == 0:
            return []
        gaps = np.where(np.diff(indices) != 1)[0]
        starts = np.insert(indices[gaps + 1], 0, indices[0])
        ends = np.append(indices[gaps], indices[-1])
        return [StateRun(int(start), int(end), int(state_code)) for start, end in zip(starts, ends)]

    runs: list[StateRun] = []
    start = 0
    for idx in range(1, values.size):
        if values[idx] != values[idx - 1]:
            runs.append(StateRun(start, idx - 1, int(values[start])))
            start = idx
    runs.append(StateRun(start, values.size - 1, int(values[start])))
    return runs


def infer_experiment_segments(state_codes: Sequence[float | int]) -> list[ExperimentSegment]:
    segments: list[ExperimentSegment] = []
    for run in segment_state_runs(state_codes):
        label = STATE_CODE_HINTS.get(run.state_code, f"unknown_state_{run.state_code}")
        confidence = "high" if run.state_code in STATE_CODE_HINTS else "low"
        segments.append(ExperimentSegment(run=run, label=label, confidence=confidence))
    return segments


def plot_state_vs_time(trace: dict[str, np.ndarray], segments: Iterable[StateRun] | None = None):
    fig, ax = plt.subplots(figsize=(10, 3))
    time = trace["time_s"]
    state = trace["state_code"]
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
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    time = trace["time_s"]
    axes[0].plot(time, trace["voltage_v"], lw=1.0)
    axes[0].set_ylabel("Voltage (V)")
    axes[1].plot(time, trace["power_mw"], lw=1.0)
    axes[1].set_ylabel("Power (mW)")
    axes[1].set_xlabel("Time (s)")
    return fig


def plot_force_extension(trace: dict[str, np.ndarray], segments: Iterable[StateRun] | None = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    extension = trace["extension_nm"]
    force = trace["force_pn"]
    valid = ~(np.isnan(extension) | np.isnan(force))
    if np.any(valid):
        ax.plot(extension[valid], force[valid], lw=1.0)
    for run in segments or []:
        run_mask = np.arange(extension.size)
        run_mask = (run_mask >= run.start_idx) & (run_mask <= run.end_idx) & valid
        if np.any(run_mask):
            ax.plot(extension[run_mask], force[run_mask], lw=1.5)
    ax.set_xlabel("Extension (nm)")
    ax.set_ylabel("Force (pN)")
    ax.set_title("Force/Extension")
    return fig


def plot_segment_boundaries(trace: dict[str, np.ndarray], segments: Sequence[StateRun]):
    fig, ax = plt.subplots(figsize=(10, 3))
    time = trace["time_s"]
    y = np.zeros_like(time)
    ax.plot(time, y, alpha=0.0)
    for run in segments:
        ax.axvspan(time[run.start_idx], time[run.end_idx], alpha=0.2)
        ax.text(time[run.start_idx], 0.02, str(run.state_code), rotation=90, va="bottom")
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Segment boundaries")
    return fig
