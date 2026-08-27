#!/usr/bin/env python3
"""Thin web view for the optical trap calibration tool.

Reuses the calibration functions from ``optical_trap_calibration.py`` unchanged.
The original Tkinter desktop GUI is still available locally via
``python optical_trap_calibration.py``; this module only adds a browser-facing
view so the tool can be used inside the Base44 preview.
"""
from __future__ import annotations

import base64
import io
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from optical_trap_calibration import calibrate_file

app = Flask(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIRS = [REPO_ROOT / "example_data", REPO_ROOT]


def list_calibration_files() -> dict[str, Path]:
    """Map file basename -> path for every data file that has a companion _Voltage.dat."""
    found: dict[str, Path] = {}
    for d in DATA_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.dat")):
            if p.name.endswith("_Voltage.dat") or p.name.endswith("_params.dat"):
                continue
            if p.with_name(f"{p.stem}_Voltage.dat").exists():
                found.setdefault(p.name, p)
    return found


def render_plot(results: list[dict]) -> str:
    n = len(results)
    fig, axes = plt.subplots(1, n, squeeze=False, figsize=(6.8 * n, 4.4))
    axes = axes.reshape(-1)
    for idx, r in enumerate(results):
        ax = axes[idx]
        ax.loglog(r["frequency"], r["psd"], label=f"Track {r['track_index']} data")
        ax.loglog(r["frequency"], r["fit"], "--", linewidth=1.5, label="aliased fit")
        ax.set_title(f"Track {r['track_index']}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="best")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    files = list_calibration_files()
    selected = request.args.get("file", "")
    results = None
    trap_power = None
    error = None
    plot_b64 = None

    if selected and selected in files:
        try:
            results, trap_power = calibrate_file(files[selected])
            plot_b64 = render_plot(results)
        except Exception as exc:  # surface calibration errors to the page
            error = str(exc)
    elif selected:
        error = f"File not found: {selected}"

    return _render_results_page(files, selected, results, trap_power, error, plot_b64)


def _render_results_page(files, selected, results, trap_power, error, plot_b64):
    return render_template(
        "index.html",
        files=files,
        selected=selected,
        results=results,
        trap_power=trap_power,
        error=error,
        plot_b64=plot_b64,
    )


@app.route("/upload", methods=["POST"])
def upload():
    files = list_calibration_files()
    data_storage = request.files.get("data")
    voltage_storage = request.files.get("voltage")
    results = None
    trap_power = None
    error = None
    plot_b64 = None
    selected = ""

    if not data_storage or not data_storage.filename:
        error = "Please choose a data file."
    elif not voltage_storage or not voltage_storage.filename:
        error = "Please choose the companion _Voltage.dat file as well."
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="otc_"))
        try:
            data_name = secure_filename(data_storage.filename)
            data_path = tmpdir / data_name
            data_storage.save(str(data_path))
            # Save the voltage file under the companion name calibrate_file() expects.
            voltage_path = tmpdir / f"{data_path.stem}_Voltage.dat"
            voltage_storage.save(str(voltage_path))
            selected = data_name
            try:
                results, trap_power = calibrate_file(data_path)
                plot_b64 = render_plot(results)
            except Exception as exc:  # surface calibration errors to the page
                error = str(exc)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return _render_results_page(files, selected, results, trap_power, error, plot_b64)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True, use_reloader=True)
