from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import re
from typing import Any, Iterable, Sequence

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
    "actuation_state_code",
    "power_mw",
)

FORCE_EXTENSION_STATE = 29  # state 29 is the actual tether force-extension pull


# ---------------------------------------------------------------------------
# Canonical state-label table.  All known column-6 state codes are listed
# here.  Unknown codes keep the label "unknown_state_N" at call sites.
# To add a new code, add one entry to this dict.
# ---------------------------------------------------------------------------
STATE_LABELS: dict[int, str] = {
    # Oscillator workflow
    3: "oscillator_hold",
    4: "oscillator_ramp",
    # Tether-detection workflow
    23: "post_break",
    24: "approach",
    25: "touch",
    26: "pull_to_reference",
    27: "measure_reference",
    29: "tether_force_extension",
    30: "stop_tether_detection",
    # Background / instrument housekeeping states present in real files
    5: "background_5",
    7: "background_7",
    19: "background_19",
    20: "background_20",
    # Legacy / less common states
    21: "tether_attempt_or_contact",
    31: "force_extension_subtether",
}

# Keep the old alias used by callers that import STATE_CODE_HINTS
STATE_CODE_HINTS: dict[int, str] = STATE_LABELS


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


@dataclass(frozen=True)
class StateRunSummary:
    state_code: int
    label: str
    confidence: str
    run_count: int
    total_samples: int
    total_duration_s: float
    fraction_of_trace: float
    run_starts: tuple[int, ...]
    run_ends: tuple[int, ...]
    run_start_times_s: tuple[float, ...]
    run_end_times_s: tuple[float, ...]


def _resolve_state_label_and_confidence(state_code: int) -> tuple[str, str]:
    label = STATE_CODE_HINTS.get(state_code, f"unknown_state_{state_code}")
    confidence = "high" if state_code in STATE_CODE_HINTS else "low"
    return label, confidence


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


def _coerce_numeric_rows(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = [token for token in re.split(r"[\t, ]+", line) if token]
        if not tokens:
            continue
        try:
            numeric = [float(token) for token in tokens]
        except ValueError:
            continue
        rows.append(numeric)
    return rows


def _legacy_params_result(path: str | Path | None = None) -> dict[str, Any]:
    result = {
        "path": str(Path(path)) if path is not None else None,
        "mode_code": None,
        "code": None,
        "mode": None,
        "mode_name": None,
        "acquisition_mode": None,
        "legacy": True,
        "valid": False,
        "tether_voltages": [],
        "oscillator_voltages": [],
        "oscillator_waits": [],
        "puppet_schedule": [],
        "schedule_rows": [],
        "schedule": [],
    }
    return result


def find_companion_params_path(data_path: str | Path) -> Path | None:
    """Return the companion ``*_params.dat`` next to a ``.dat`` file if present."""
    path = Path(data_path)
    candidate = path.with_name(f"{path.stem if path.suffix else path.name}_params.dat")
    return candidate if candidate.exists() else None


def read_acquisition_params(path: str | Path | None = None) -> dict[str, Any]:
    """Read a companion ``*_params.dat`` file.

    The newer instrument writes the acquisition mode in the first non-empty row,
    usually repeated across four columns, followed by mode-specific values. The
    legacy/blank files used by older datasets contain ``0`` or no schedule data;
    those are treated as compatibility metadata rather than hard failures.

    Format (confirmed from instrument export):
    * mode code row: ``0`` / ``1`` / ``2`` / ``3`` repeated (calibration,
      tether formation, oscillator, puppet)
    * tether formation: one row of ``touch_V, reference_V, tether_V, end_V``
    * oscillator: one row of ``low_force_V, low_wait_s, high_force_V, high_wait_s``
    * puppet: N rows of ``target_V, ramp_rate_or_time, hold_length, step_index``

    Older datasets with blank / ``0`` files are intentionally accepted as legacy
    and leave mode detection to the state-voltage traces.
    """
    if path is None:
        return _legacy_params_result()

    data_path = Path(path)
    if not data_path.exists():
        return _legacy_params_result(data_path)

    try:
        text = data_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    except OSError:
        return _legacy_params_result(data_path)

    rows = _coerce_numeric_rows(text)
    if not rows:
        return _legacy_params_result(data_path)

    first_row = rows[0]
    first_value = float(first_row[0])
    mode_code = int(round(first_value)) if np.isfinite(first_value) else None
    if mode_code not in (0, 1, 2, 3):
        return _legacy_params_result(data_path)

    mode_name = {
        0: "calibration",
        1: "tether_formation",
        2: "oscillator",
        3: "puppet",
    }.get(mode_code)

    schedule_rows: list[list[float]] = []
    for raw_row in rows[1:]:
        if not raw_row:
            continue
        if len(raw_row) < 4:
            continue
        schedule_rows.append([float(value) for value in raw_row[:4]])

    result: dict[str, Any] = {
        "path": str(data_path),
        "mode_code": mode_code,
        "code": mode_code,
        "mode": mode_name,
        "mode_name": mode_name,
        "acquisition_mode": mode_name,
        "legacy": mode_code == 0 or not schedule_rows,
        "valid": True,
        "tether_voltages": [],
        "oscillator_voltages": [],
        "oscillator_waits": [],
        "puppet_schedule": [],
        "schedule_rows": schedule_rows,
        "schedule": schedule_rows,
    }

    if mode_code == 1:
        volts = [float(v) for v in schedule_rows[0][:4]] if schedule_rows else []
        result["tether_voltages"] = volts
    elif mode_code == 2:
        if schedule_rows:
            first = [float(v) for v in schedule_rows[0][:4]]
            result["oscillator_voltages"] = [first[0], first[2]]
            result["oscillator_waits"] = [first[1], first[3]]
    elif mode_code == 3:
        result["puppet_schedule"] = [
            {
                "target_v": float(row[0]),
                "ramp_or_time": float(row[1]),
                "hold_length": float(row[2]),
                "step_index": int(round(float(row[3]))),
            }
            for row in schedule_rows
        ]
        result["schedule"] = result["puppet_schedule"]

    if mode_code == 0:
        result["legacy"] = True
        result["schedule"] = []
        result["schedule_rows"] = []
        result["puppet_schedule"] = []
    return result


def read_companion_params(data_path: str | Path) -> dict[str, Any]:
    """Read a companion ``*_params.dat`` beside a data file, or return legacy metadata."""
    path = Path(data_path)
    params_path = find_companion_params_path(path)
    if params_path is None:
        return _legacy_params_result(path)
    return read_acquisition_params(params_path)


def load_normalized_trace(
    filepath: str | Path,
    *,
    companion_voltage_path: str | Path | None = None,
    companion_params_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    data = _load_numeric_table(filepath)
    n_rows, n_cols = data.shape
    trace: dict[str, np.ndarray] = {key: np.full(n_rows, np.nan) for key in NORMALIZED_SCHEMA}

    trace["time_s"] = data[:, 0]

    if companion_params_path is not None:
        params_path = Path(companion_params_path)
    else:
        params_path = find_companion_params_path(filepath)
    if params_path is not None:
        params = read_acquisition_params(params_path)
        mode_code = params.get("mode_code")
        trace["acquisition_mode"] = np.full(n_rows, params.get("mode") or "unknown", dtype=object)
        trace["acquisition_mode_code"] = np.full(n_rows, mode_code if mode_code is not None else np.nan, dtype=float)
        trace["params_path"] = np.full(n_rows, str(params_path), dtype=object)
        trace["params_legacy"] = np.full(n_rows, bool(params.get("legacy", False)), dtype=bool)
    else:
        trace["acquisition_mode"] = np.full(n_rows, "unknown", dtype=object)
        trace["acquisition_mode_code"] = np.full(n_rows, np.nan, dtype=float)
        trace["params_path"] = np.full(n_rows, "", dtype=object)
        trace["params_legacy"] = np.full(n_rows, True, dtype=bool)

    if n_cols == 6:
        # MATLAB stateful layout: [t, z, F, V, k, State]. Position channels are not present.
        trace["extension_nm"] = data[:, 1]
        trace["force_pn"] = data[:, 2]
        trace["voltage_v"] = data[:, 3]
        trace["stiffness_pn_per_nm"] = data[:, 4]
        trace["state_code"] = data[:, 5]
    elif n_cols == 7:
        # MATLAB two-trap layout:
        # [time, trap1_a, trap2_a, trap1_b, trap2_b, primary_state, actuation_state]
        # Column assignments (1-indexed):
        #   col2 = trap1 outer edge (data[:,1])
        #   col3 = trap2 outer edge (data[:,2])
        #   col4 = trap1 interface edge (data[:,3])
        #   col5 = trap2 interface edge (data[:,4])
        # Pairing follows the repo convention: trap1 = mean(col1, col3) and trap2 = mean(col2, col4).
        # Power/voltage for these files come exclusively from the companion _Voltage.dat.
        trace["edge1_raw_nm"] = data[:, 1]   # trap1 outer edge (col2)
        trace["edge2_raw_nm"] = data[:, 2]   # trap2 outer edge (col3)
        trace["edge3_raw_nm"] = data[:, 3]   # trap1 interface edge (col4)
        trace["edge4_raw_nm"] = data[:, 4]   # trap2 interface edge (col5)
        trace["position_1_nm"] = (data[:, 1] + data[:, 3]) / 2.0
        trace["position_2_nm"] = (data[:, 2] + data[:, 4]) / 2.0
        trace["state_code"] = data[:, 5]
        trace["actuation_state_code"] = data[:, 6]
        trace["extension_nm"] = trace["position_2_nm"] - trace["position_1_nm"]
    elif n_cols >= 2:
        positions = data[:, 1:]
        if positions.shape[1] == 1:
            trace["position_1_nm"] = positions[:, 0]
        elif positions.shape[1] == 2:
            trace["position_1_nm"] = positions[:, 0]
            trace["position_2_nm"] = positions[:, 1]
            trace["extension_nm"] = positions[:, 1] - positions[:, 0]
        elif positions.shape[1] == 4:
            # Repo convention for dual-trap data keeps the two trap centers interleaved:
            # trap1 = mean(col0, col2), trap2 = mean(col1, col3).
            trap1 = (positions[:, 0] + positions[:, 2]) / 2.0
            trap2 = (positions[:, 1] + positions[:, 3]) / 2.0
            trace["position_1_nm"] = trap1
            trace["position_2_nm"] = trap2
            trace["extension_nm"] = trap2 - trap1
        elif positions.shape[1] == 3:
            # A 3-value position block includes one explicit position pair plus a third axis/track,
            # so the first two columns remain the most stable dual-position representation.
            trace["position_1_nm"] = positions[:, 0]
            trace["position_2_nm"] = positions[:, 1]
            trace["extension_nm"] = positions[:, 1] - positions[:, 0]
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
        label, confidence = _resolve_state_label_and_confidence(run.state_code)
        segments.append(ExperimentSegment(run=run, label=label, confidence=confidence))
    return segments


def _run_duration(time: np.ndarray, run: StateRun) -> float:
    """Return duration of a single state run.

    A single-sample run gets one sample period instead of 0 s.
    """
    start = run.start_idx
    end = run.end_idx
    if start == end:
        # Single-sample run: use the gap to the next sample if available,
        # otherwise the gap from the previous sample.
        if end + 1 < time.size:
            return float(time[end + 1] - time[end])
        if start > 0:
            return float(time[start] - time[start - 1])
        return 0.0
    return float(time[end] - time[start])


def summarize_pulls(
    trace: dict[str, np.ndarray],
    pull_state: int = FORCE_EXTENSION_STATE,
) -> dict[str, Any]:
    """Find every contiguous run of *pull_state* and return a summary dict.

    Returns
    -------
    dict with keys:
        n_pulls         – number of contiguous pull runs
        pulls           – list of {start_time_s, end_time_s, duration_s} per pull
        gaps            – list of {states, duration_s} for the inter-pull gaps
    """
    time = np.asarray(trace.get("time_s", np.arange(trace["state_code"].size, dtype=float)))
    all_runs = segment_state_runs(trace["state_code"])
    pull_runs = [r for r in all_runs if r.state_code == pull_state]

    pulls: list[dict[str, Any]] = []
    for r in pull_runs:
        pulls.append(
            {
                "start_time_s": float(time[r.start_idx]) if r.start_idx < time.size else float("nan"),
                "end_time_s": float(time[r.end_idx]) if r.end_idx < time.size else float("nan"),
                "duration_s": _run_duration(time, r),
            }
        )

    gaps: list[dict[str, Any]] = []
    for i in range(len(pull_runs) - 1):
        gap_start_idx = pull_runs[i].end_idx + 1
        gap_end_idx = pull_runs[i + 1].start_idx - 1
        gap_states = [
            r.state_code
            for r in all_runs
            if r.start_idx >= gap_start_idx and r.end_idx <= gap_end_idx
        ]
        gap_duration = 0.0
        if gap_start_idx <= gap_end_idx and gap_end_idx < time.size:
            gap_duration = float(time[gap_end_idx] - time[gap_start_idx])
            if gap_start_idx == gap_end_idx and gap_end_idx + 1 < time.size:
                gap_duration = float(time[gap_end_idx + 1] - time[gap_end_idx])
        gaps.append({"states": gap_states, "duration_s": gap_duration})

    return {"n_pulls": len(pull_runs), "pulls": pulls, "gaps": gaps}


def summarize_state_runs(
    trace: dict[str, np.ndarray],
    *,
    state_runs: Sequence[StateRun] | None = None,
) -> list[StateRunSummary]:
    """Summarize contiguous state runs by code, label, duration, and fraction of trace."""
    time = np.asarray(trace.get("time_s", np.arange(trace["state_code"].size, dtype=float)))
    state_codes = np.asarray(trace["state_code"])
    runs = list(state_runs) if state_runs is not None else segment_state_runs(state_codes)
    if not runs:
        return []

    full_duration = 0.0
    if time.size > 1:
        full_duration = float(time[-1] - time[0])

    summary_by_code: dict[int, dict[str, Any]] = {}
    for run in runs:
        state_code = int(run.state_code)
        label, confidence = _resolve_state_label_and_confidence(state_code)
        duration = 0.0
        if run.start_idx < time.size and run.end_idx < time.size:
            duration = float(time[run.end_idx] - time[run.start_idx])
        entry = summary_by_code.setdefault(
            state_code,
            {
                "state_code": state_code,
                "label": label,
                "confidence": confidence,
                "run_count": 0,
                "total_samples": 0,
                "total_duration_s": 0.0,
                "run_starts": [],
                "run_ends": [],
                "run_start_times_s": [],
                "run_end_times_s": [],
            },
        )
        entry["run_count"] += 1
        sample_count = max(run.end_idx - run.start_idx + 1, 0)
        entry["total_samples"] += sample_count
        entry["total_duration_s"] += duration
        entry["run_starts"].append(run.start_idx)
        entry["run_ends"].append(run.end_idx)
        if time.size > run.start_idx:
            entry["run_start_times_s"].append(float(time[run.start_idx]))
        else:
            entry["run_start_times_s"].append(float("nan"))
        if time.size > run.end_idx:
            entry["run_end_times_s"].append(float(time[run.end_idx]))
        else:
            entry["run_end_times_s"].append(float("nan"))

    results: list[StateRunSummary] = []
    for state_code, entry in sorted(summary_by_code.items()):
        total_samples = int(entry["total_samples"])
        total_duration = float(entry["total_duration_s"])
        fraction = 0.0 if full_duration <= 0.0 else total_duration / full_duration
        results.append(
            StateRunSummary(
                state_code=entry["state_code"],
                label=entry["label"],
                confidence=entry["confidence"],
                run_count=int(entry["run_count"]),
                total_samples=total_samples,
                total_duration_s=total_duration,
                fraction_of_trace=fraction,
                run_starts=tuple(entry["run_starts"]),
                run_ends=tuple(entry["run_ends"]),
                run_start_times_s=tuple(entry["run_start_times_s"]),
                run_end_times_s=tuple(entry["run_end_times_s"]),
            )
        )
    return results


def export_state_report_csv(summary: Sequence[StateRunSummary | dict[str, Any]], path: str | Path):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state_code",
                "label",
                "confidence",
                "run_count",
                "total_samples",
                "total_duration_s",
                "fraction_of_trace",
            ]
        )
        for item in summary:
            if isinstance(item, dict):
                record = item
            else:
                record = {
                    "state_code": item.state_code,
                    "label": item.label,
                    "confidence": item.confidence,
                    "run_count": item.run_count,
                    "total_samples": item.total_samples,
                    "total_duration_s": item.total_duration_s,
                    "fraction_of_trace": item.fraction_of_trace,
                }
            writer.writerow(
                [
                    record["state_code"],
                    record["label"],
                    record["confidence"],
                    record["run_count"],
                    record["total_samples"],
                    record["total_duration_s"],
                    record["fraction_of_trace"],
                ]
            )


def export_state_report_json(summary: Sequence[StateRunSummary | dict[str, Any]], path: str | Path):
    path = Path(path)
    payload = []
    for item in summary:
        if isinstance(item, dict):
            record = item
        else:
            record = {
                "state_code": item.state_code,
                "label": item.label,
                "confidence": item.confidence,
                "run_count": item.run_count,
                "total_samples": item.total_samples,
                "total_duration_s": item.total_duration_s,
                "fraction_of_trace": item.fraction_of_trace,
                "run_starts": item.run_starts,
                "run_ends": item.run_ends,
                "run_start_times_s": item.run_start_times_s,
                "run_end_times_s": item.run_end_times_s,
            }
        payload.append(record)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_state_report(trace: dict[str, np.ndarray], *, include_runs: bool = True) -> dict[str, Any]:
    state_runs = segment_state_runs(trace["state_code"])
    summary = summarize_state_runs(trace, state_runs=state_runs)
    time = np.asarray(trace.get("time_s", np.arange(trace["state_code"].size, dtype=float)))
    details: list[dict[str, Any]] = []
    for run in state_runs:
        start_time = float("nan")
        end_time = float("nan")
        if time.size > run.start_idx:
            start_time = float(time[run.start_idx])
        if time.size > run.end_idx:
            end_time = float(time[run.end_idx])
        label, confidence = _resolve_state_label_and_confidence(run.state_code)
        details.append(
            {
                "start_idx": run.start_idx,
                "end_idx": run.end_idx,
                "state_code": run.state_code,
                "label": label,
                "confidence": confidence,
                "start_time_s": start_time,
                "end_time_s": end_time,
            }
        )
    report: dict[str, Any] = {
        "pulls": summarize_pulls(trace),
        "state_runs": details if include_runs else [],
        "summary": [
            {
                "state_code": item.state_code,
                "label": item.label,
                "confidence": item.confidence,
                "run_count": item.run_count,
                "total_samples": item.total_samples,
                "total_duration_s": item.total_duration_s,
                "fraction_of_trace": item.fraction_of_trace,
                "run_starts": list(item.run_starts),
                "run_ends": list(item.run_ends),
                "run_start_times_s": list(item.run_start_times_s),
                "run_end_times_s": list(item.run_end_times_s),
            }
            for item in summary
        ],
        "total_states": len(summary),
        "total_runs": len(state_runs),
    }
    return report


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
