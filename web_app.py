#!/usr/bin/env python3
"""FastAPI web app for optical trap calibration with adjustable parameters.

Reuses the calibration functions from ``optical_trap_calibration.py`` unchanged.
The original Tkinter desktop GUI is still available locally via
``python optical_trap_calibration.py``.
"""
from __future__ import annotations

import base64
import io
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from optical_trap_calibration import calibrate_file

app = FastAPI(title="Optical Trap Calibration")
templates = Jinja2Templates(directory="templates")

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIRS = [REPO_ROOT / "example_data", REPO_ROOT]

DEFAULT_PARAMS = {
    "blocks": 64,
    "low_freq_cutoff": 20.0,
    "notch_low": 90.0,
    "notch_high": 120.0,
    "x0_0": 2e-5,
    "x0_1": 100.0,
    "lb_0": 1e-6,
    "lb_1": 0.0,
    "ub_0": 1e-4,
    "ub_1": 1000.0,
}


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


def _calibration_params(
    blocks: int, low_freq_cutoff: float, notch_low: float, notch_high: float,
    x0_0: float, x0_1: float, lb_0: float, lb_1: float, ub_0: float, ub_1: float,
) -> dict:
    """Build the kwargs dict for calibrate_file from individual form fields."""
    return {
        "blocks": blocks,
        "low_freq_cutoff": low_freq_cutoff,
        "notch_low": notch_low,
        "notch_high": notch_high,
        "x0": np.array([x0_0, x0_1], dtype=float),
        "lb": np.array([lb_0, lb_1], dtype=float),
        "ub": np.array([ub_0, ub_1], dtype=float),
    }


def _template_params(
    blocks: int, low_freq_cutoff: float, notch_low: float, notch_high: float,
    x0_0: float, x0_1: float, lb_0: float, lb_1: float, ub_0: float, ub_1: float,
) -> dict:
    """Build the params dict for template rendering (flat, no numpy arrays)."""
    return {
        "blocks": blocks,
        "low_freq_cutoff": low_freq_cutoff,
        "notch_low": notch_low,
        "notch_high": notch_high,
        "x0_0": x0_0,
        "x0_1": x0_1,
        "lb_0": lb_0,
        "lb_1": lb_1,
        "ub_0": ub_0,
        "ub_1": ub_1,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    files = list_calibration_files()
    return templates.TemplateResponse(request, "index.html", {
        "files": files,
        "params": DEFAULT_PARAMS,
        "selected": "",
    })


@app.post("/calibrate", response_class=HTMLResponse)
async def calibrate(
    request: Request,
    example_file: str = Form(""),
    data: Optional[UploadFile] = File(None),
    voltage: Optional[UploadFile] = File(None),
    blocks: int = Form(64),
    low_freq_cutoff: float = Form(20.0),
    notch_low: float = Form(90.0),
    notch_high: float = Form(120.0),
    x0_0: float = Form(2e-5),
    x0_1: float = Form(100.0),
    lb_0: float = Form(1e-6),
    lb_1: float = Form(0.0),
    ub_0: float = Form(1e-4),
    ub_1: float = Form(1000.0),
):
    files = list_calibration_files()
    cal_params = _calibration_params(
        blocks, low_freq_cutoff, notch_low, notch_high,
        x0_0, x0_1, lb_0, lb_1, ub_0, ub_1,
    )
    tmpl_params = _template_params(
        blocks, low_freq_cutoff, notch_low, notch_high,
        x0_0, x0_1, lb_0, lb_1, ub_0, ub_1,
    )
    results = None
    trap_power = None
    error = None
    plot_b64 = None
    selected = ""

    use_upload = data is not None and data.filename and voltage is not None and voltage.filename

    if use_upload:
        tmpdir = Path(tempfile.mkdtemp(prefix="otc_"))
        try:
            data_name = Path(data.filename).name
            data_path = tmpdir / data_name
            data_path.write_bytes(await data.read())
            voltage_path = tmpdir / f"{data_path.stem}_Voltage.dat"
            voltage_path.write_bytes(await voltage.read())
            selected = data_name
            try:
                results, trap_power = calibrate_file(data_path, **cal_params)
                plot_b64 = render_plot(results)
            except Exception as exc:
                error = str(exc)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    elif example_file and example_file in files:
        selected = example_file
        try:
            results, trap_power = calibrate_file(files[example_file], **cal_params)
            plot_b64 = render_plot(results)
        except Exception as exc:
            error = str(exc)
    elif example_file:
        error = f"File not found: {example_file}"
    else:
        error = "Please select an example file or upload your own data and voltage files."

    return templates.TemplateResponse(request, "index.html", {
        "files": files,
        "params": tmpl_params,
        "selected": selected,
        "results": results,
        "trap_power": trap_power,
        "error": error,
        "plot_b64": plot_b64,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=3000, reload=True)
