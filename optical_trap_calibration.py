#!/usr/bin/env python3
"""GUI/CLI optical trap calibration app.

This tool reads the same text data format produced by the MATLAB code, computes
an averaged PSD, and fits the aliased Lorentzian model implemented in
`filtpsdaliased.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy.optimize import curve_fit

import matplotlib

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    matplotlib.use("TkAgg", force=True)
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None
    FigureCanvasTkAgg = None
    Figure = None
    matplotlib.use("Agg")

from filtpsdaliased import filtpsdaliased

DEFAULT_BLOCKS = 64


def load_voltage_file(filepath: str | os.PathLike[str]) -> float:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Voltage file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = path.read_bytes().decode("latin-1").strip()

    if not text:
        raise ValueError(f"Voltage file is empty: {path}")

    if "," in text:
        values = [float(value) for value in text.split(",") if value.strip()]
        if not values:
            raise ValueError(f"Voltage file is empty: {path}")
        return float(values[0])

    for delimiter in (None, "\t", " ", ","):
        try:
            values = np.loadtxt(path, dtype=float, delimiter=delimiter, ndmin=1)
        except ValueError:
            continue
        if values.size == 0:
            continue
        return float(np.asarray(values).reshape(-1)[0])

    raise ValueError(
        f"Could not parse voltage file: {path}. "
        "Expected a single power value or a comma-delimited repeated-value file "
        "like '102.1,5.0,102.1,5.0,...'."
    )


def _fixed_time_array(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Data length must be positive")
    time = np.zeros(length, dtype=float)
    for i in range(1, length):
        interval = 0.710e-3 if i % 2 == 0 else 0.720e-3
        time[i] = time[i - 1] + interval
    return time


def load_position_data(filepath: str | os.PathLike[str]) -> List[np.ndarray]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    load_errors: list[ValueError] = []
    for delimiter in (None, "\t", " ", ","):
        try:
            data = np.loadtxt(path, dtype=float, delimiter=delimiter, ndmin=2)
        except ValueError as exc:
            load_errors.append(ValueError(str(exc)))
            continue

        if data.size == 0:
            raise ValueError(f"No data found in file: {path}")

        if data.shape[1] < 2:
            raise ValueError(
                "Expected at least 2 columns: time and one position channel. "
                f"Detected {data.shape[1]} column(s) in {path}."
            )

        time = data[:, 0]
        if time.shape[0] < 2:
            raise ValueError("At least two samples are required for calibration.")

        position_columns = data[:, 1:]
        if position_columns.shape[1] == 1:
            return [position_columns[:, 0]]

        if position_columns.shape[1] == 2:
            return [(position_columns[:, 0] + position_columns[:, 1]) / 2.0]

        if position_columns.shape[1] == 6:
            # MATLAB-style 7-column layout: [time, left1, left2, right1, right2, const, const]
            trap1_center = (position_columns[:, 0] + position_columns[:, 2]) / 2.0
            trap2_center = (position_columns[:, 1] + position_columns[:, 3]) / 2.0
            return [trap1_center, trap2_center]

        if position_columns.shape[1] == 4:
            trap1_center = (position_columns[:, 0] + position_columns[:, 1]) / 2.0
            trap2_center = (position_columns[:, 2] + position_columns[:, 3]) / 2.0
            return [trap1_center, trap2_center]

        if position_columns.shape[1] % 2 == 0:
            return [
                position_columns[:, start : start + 2].mean(axis=1)
                for start in range(0, position_columns.shape[1], 2)
            ]

        raise ValueError(
            f"Unsupported data format in {path}: detected {position_columns.shape[1]} position columns "
            f"after the time column. Expected 1 (center), 2 (left+right), 4 (2 traps), or 6 "
            "(MATLAB 7-column layout). Check that the file is tab/space-delimited and that the "
            "time column is first."
        )

    details = "; ".join(str(exc) for exc in load_errors)
    raise ValueError(
        f"Could not parse data file {path}. Expected tab/space-delimited numeric data with a time column. "
        f"Detected parsing errors: {details}"
    )


def compute_averaged_psd(signal: np.ndarray, blocks: int = DEFAULT_BLOCKS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    if signal.size < blocks:
        raise ValueError(
            f"Not enough samples for {blocks} blocks; got {signal.size}."
        )

    centered = signal - np.mean(signal)
    time = _fixed_time_array(centered.size)
    ndata = centered.size // blocks
    if ndata < 2:
        raise ValueError("Not enough points in each PSD block for fitting.")

    dt = time[-1] / max(1, centered.size - 1)
    df = 1.0 / (ndata * dt)
    nfreq = ndata // 2 + 1
    freq = np.arange(nfreq, dtype=float) * df
    psd = np.zeros((nfreq, blocks), dtype=float)

    for block_index in range(blocks):
        start = block_index * ndata
        stop = start + ndata
        segment = centered[start:stop]
        fft_values = np.fft.fft(segment)
        magnitude = np.abs(fft_values)
        psd[0, block_index] = (magnitude[0] ** 2) / (df * ndata ** 2)
        if nfreq > 1:
            psd[1 : nfreq - 1, block_index] = (
                2.0 * (magnitude[1 : nfreq - 1] ** 2) / (df * ndata ** 2)
            )
        psd[nfreq - 1, block_index] = (
            magnitude[nfreq - 1] ** 2 / (df * ndata ** 2)
            if nfreq > 1
            else psd[0, block_index]
        )

    avg_psd = np.mean(psd, axis=1)
    return freq, avg_psd, centered


def _frequency_filter(freq: np.ndarray, avg_psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if freq.size == 0:
        raise ValueError("No frequency points available for fitting.")

    below_20hz = freq < 20.0
    filter_mask = ((freq > 90.0) & (freq < 120.0)) | below_20hz
    filter_mask[-1] = True
    filtered_freq = freq[~filter_mask]
    filtered_psd = avg_psd[~filter_mask]
    if filtered_freq.size < 5:
        raise ValueError(
            "Insufficient valid frequency points remain after the default filter. "
            "Please adjust the fit bounds or use a longer data set."
        )
    return filtered_freq, filtered_psd


def fit_aliased_psd(freq: np.ndarray, avg_psd: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    freq_filt, avg_psd_filt = _frequency_filter(freq, avg_psd)
    x0 = np.array([2e-5, 100.0], dtype=float)
    lb = np.array([1e-6, 0.0], dtype=float)
    ub = np.array([1e-4, 1000.0], dtype=float)

    def fit_model(xdata: np.ndarray, gamma: float, fc: float) -> np.ndarray:
        return filtpsdaliased(np.array([gamma, fc], dtype=float), xdata)

    try:
        popt, _ = curve_fit(
            fit_model,
            freq_filt,
            avg_psd_filt,
            p0=x0,
            bounds=(lb, ub),
            maxfev=100000,
        )
    except Exception as exc:  # pragma: no cover - fallback error path
        raise RuntimeError(f"PSD fitting failed: {exc}") from exc

    fit_curve = filtpsdaliased(popt, freq)
    stiffness = float(2.0 * np.pi * popt[0] * popt[1])
    return popt, fit_curve, stiffness


def calibrate_file(filepath: str | os.PathLike[str]) -> tuple[list[dict], float]:
    path = Path(filepath)
    tracks = load_position_data(path)
    voltage_path = path.with_name(f"{path.stem}_Voltage.dat")
    if not voltage_path.exists():
        raise FileNotFoundError(
            f"Voltage file not found for {path.name}. Expected {voltage_path.name}."
        )
    trap_power = load_voltage_file(voltage_path)

    results: list[dict] = []
    for index, track in enumerate(tracks, start=1):
        freq, avg_psd, _ = compute_averaged_psd(track, blocks=DEFAULT_BLOCKS)
        params, fit_curve, stiffness = fit_aliased_psd(freq, avg_psd)
        results.append(
            {
                "track_index": index,
                "gamma": float(params[0]),
                "corner_frequency_hz": float(params[1]),
                "stiffness_pn_per_nm": stiffness,
                "stiffness_per_mw": stiffness / trap_power,
                "frequency": freq,
                "psd": avg_psd,
                "fit": fit_curve,
            }
        )

    return results, float(trap_power)


class OpticalTrapCalibrationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Optical Trap Calibration")
        self.root.geometry("900x650")

        frame = ttk.Frame(root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(frame, text="Select Data File", command=self.select_file).pack(anchor="w")
        self.file_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.file_var, wraplength=780).pack(anchor="w", pady=(8, 0))

        self.power_var = tk.StringVar(value="Trap power: --")
        ttk.Label(frame, textvariable=self.power_var).pack(anchor="w", pady=(8, 0))

        self.results_var = tk.StringVar(value="Results: --")
        ttk.Label(frame, textvariable=self.results_var, wraplength=780).pack(anchor="w", pady=(8, 0))

        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(12, 0))

    def select_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select optical trap tracking file",
            filetypes=[("Data files", "*.dat *.txt *.csv *.tsv"), ("All files", "*.*")],
        )
        if not filename:
            return

        self.file_var.set(filename)
        try:
            results, trap_power = calibrate_file(filename)
        except Exception as exc:  # pragma: no cover - GUI fallback path
            messagebox.showerror("Calibration failed", str(exc))
            return

        self.power_var.set(f"Trap power: {trap_power} mW")

        lines = []
        for result in results:
            lines.append(
                f"Trap {result['track_index']}: gamma = {result['gamma']:.3e}, "
                f"corner freq (fc) = {result['corner_frequency_hz']:.2f} Hz, "
                f"stiffness k = {result['stiffness_pn_per_nm']:.4f} pN/nm, "
                f"k/mW = {result['stiffness_per_mw']:.3e} pN/nm/mW"
            )
        self.results_var.set("\n".join(lines))

        self.axes.clear()
        for result in results:
            self.axes.loglog(result["frequency"], result["psd"], label=f"Track {result['track_index']}")
            self.axes.loglog(result["frequency"], result["fit"], linestyle="--", linewidth=1.5)
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("PSD")
        self.axes.set_title("Averaged PSD and fitted aliased Lorentzian")
        self.axes.grid(True, which="both", linestyle="--", alpha=0.4)
        self.axes.legend(loc="best")
        self.canvas.draw()


def run_gui() -> None:
    if tk is None:
        raise RuntimeError("Tkinter is required for the GUI. Install it or run in CLI mode with --file.")
    root = tk.Tk()
    OpticalTrapCalibrationApp(root)
    root.mainloop()


def _print_calibration_results(results: list[dict], trap_power: float) -> None:
    print(f"Trap power: {trap_power} mW")
    for result in results:
        print(
            f"Trap {result['track_index']}: gamma = {result['gamma']:.3e}, "
            f"corner freq (fc) = {result['corner_frequency_hz']:.2f} Hz, "
            f"stiffness k = {result['stiffness_pn_per_nm']:.4f} pN/nm, "
            f"k/mW = {result['stiffness_per_mw']:.3e} pN/nm/mW"
        )


def run_cli(path: str, show_plot: bool = False) -> None:
    results, trap_power = calibrate_file(path)
    _print_calibration_results(results, trap_power)

    if show_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(results), squeeze=False)
        axes = axes.reshape(-1)
        for idx, result in enumerate(results):
            axes[idx].loglog(result["frequency"], result["psd"], label=f"Track {result['track_index']}")
            axes[idx].loglog(result["frequency"], result["fit"], linestyle="--", linewidth=1.5)
            axes[idx].set_title(f"Track {result['track_index']}")
            axes[idx].set_xlabel("Frequency (Hz)")
            axes[idx].set_ylabel("PSD")
            axes[idx].grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optical trap calibration with aliased PSD fitting.")
    parser.add_argument("--file", type=str, help="Calibration data file to analyze.")
    parser.add_argument("--plot", action="store_true", help="Show the fitted PSD after analysis.")
    parser.add_argument("--no-gui", action="store_true", help="Run in command-line mode without the Tk window.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.file:
        run_cli(args.file, show_plot=args.plot)
        return

    if args.no_gui:
        parser.error("--no-gui requires --file to be provided.")

    run_gui()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
